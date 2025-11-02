"""
Cache Manager Utility
====================

Provides caching functionality for the trip planner application.
Handles in-memory caching with TTL, size management, and optional persistence.

Key Features:
- In-memory caching with configurable TTL
- LRU eviction when cache size limit is reached
- Optional file-based persistence for cache data
- Cache statistics and performance monitoring
- Thread-safe operations for concurrent access
- Automatic cleanup of expired entries

Classes:
    CacheManager: Main caching interface
    CacheEntry: Internal cache entry with metadata
    CacheStats: Cache performance statistics

Author: Hybrid Trip Planner Team
"""

import logging
import time
import json
import pickle
import os
from typing import Any, Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from collections import OrderedDict

# Import configuration
from config import config


@dataclass
class CacheEntry:
    """
    Internal cache entry with metadata
    
    Attributes:
        data (Any): Cached data
        timestamp (float): When entry was created (Unix timestamp)
        ttl (int): Time-to-live in seconds
        access_count (int): Number of times accessed
        last_access (float): Last access timestamp
    """
    data: Any
    timestamp: float
    ttl: int
    access_count: int = 0
    last_access: float = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl <= 0:  # No expiration
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self) -> None:
        """Update access statistics"""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class CacheStats:
    """
    Cache performance statistics
    
    Attributes:
        hits (int): Number of cache hits
        misses (int): Number of cache misses
        sets (int): Number of cache sets
        evictions (int): Number of evicted entries
        expired (int): Number of expired entries cleaned
        total_size (int): Current cache size
        max_size (int): Maximum cache size
    """
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    expired: int = 0
    total_size: int = 0
    max_size: int = 0
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert stats to dictionary"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "expired": self.expired,
            "total_size": self.total_size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate()
        }


class CacheManager:
    """
    Main caching interface with in-memory storage and optional persistence
    Thread-safe implementation with LRU eviction and TTL support
    """
    
    def __init__(self, max_size: int = None, default_ttl: int = None, 
                 cache_dir: str = None, enable_persistence: bool = False):
        """
        Initialize Cache Manager
        
        Args:
            max_size (int): Maximum number of cache entries (default from config)
            default_ttl (int): Default TTL in seconds (default from config)
            cache_dir (str): Directory for persistent cache files
            enable_persistence (bool): Whether to enable file persistence
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_size = max_size or config.CACHE_MAX_SIZE
        self.default_ttl = default_ttl or config.CACHE_TTL
        self.cache_dir = cache_dir or config.OS_CACHE_DIR
        self.enable_persistence = enable_persistence
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Cache storage (using OrderedDict for LRU)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.stats = CacheStats(max_size=self.max_size)
        
        # Ensure cache directory exists if persistence enabled
        if self.enable_persistence and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Start cleanup thread for expired entries
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self._cleanup_running = True
        self._cleanup_thread.start()
        
        self.logger.info(f"Cache Manager initialized (max_size={self.max_size}, ttl={self.default_ttl})")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key (str): Cache key
            
        Returns:
            Any: Cached value or None if not found/expired
        """
        with self._lock:
            # Check if key exists
            if key not in self._cache:
                self.stats.misses += 1
                self.logger.debug(f"Cache miss: {key}")
                
                # Try to load from persistence
                if self.enable_persistence:
                    data = self._load_from_file(key)
                    if data is not None:
                        self.logger.debug(f"Loaded from persistent cache: {key}")
                        return data
                
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired():
                self._remove_entry(key)
                self.stats.misses += 1
                self.stats.expired += 1
                self.logger.debug(f"Cache expired: {key}")
                return None
            
            # Update access info and move to end (LRU)
            entry.touch()
            self._cache.move_to_end(key)
            
            self.stats.hits += 1
            self.logger.debug(f"Cache hit: {key}")
            
            return entry.data
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """
        Set value in cache
        
        Args:
            key (str): Cache key
            value (Any): Value to cache
            ttl (int): Time-to-live in seconds (default: use default_ttl)
        """
        with self._lock:
            ttl = ttl if ttl is not None else self.default_ttl
            
            # Create cache entry
            entry = CacheEntry(
                data=value,
                timestamp=time.time(),
                ttl=ttl
            )
            
            # Check if key already exists
            if key in self._cache:
                # Update existing entry
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = entry
                self.stats.total_size += 1
                
                # Check size limit and evict if necessary
                if len(self._cache) > self.max_size:
                    self._evict_lru()
            
            self.stats.sets += 1
            self.logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            
            # Save to persistence if enabled
            if self.enable_persistence:
                self._save_to_file(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                
                # Remove from persistence
                if self.enable_persistence:
                    self._delete_file(key)
                
                self.logger.debug(f"Cache deleted: {key}")
                return True
            
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self.stats = CacheStats(max_size=self.max_size)
            
            # Clear persistent cache
            if self.enable_persistence and self.cache_dir:
                try:
                    for filename in os.listdir(self.cache_dir):
                        if filename.endswith('.cache'):
                            os.remove(os.path.join(self.cache_dir, filename))
                except Exception as e:
                    self.logger.warning(f"Error clearing persistent cache: {e}")
            
            self.logger.info("Cache cleared")
    
    def has_key(self, key: str) -> bool:
        """
        Check if key exists in cache (and is not expired)
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: True if key exists and is valid
        """
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if entry.is_expired():
                self._remove_entry(key)
                return False
            
            return True
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            self.stats.total_size = len(self._cache)
            return self.stats
    
    def get_keys(self) -> list:
        """Get list of all cache keys"""
        with self._lock:
            return list(self._cache.keys())
    
    def cleanup_expired(self) -> int:
        """
        Manually cleanup expired entries
        
        Returns:
            int: Number of entries cleaned
        """
        with self._lock:
            expired_keys = []
            current_time = time.time()
            
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
                self.stats.expired += 1
            
            if expired_keys:
                self.logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self._cache:
            return
        
        # Remove the first item (least recently used)
        lru_key = next(iter(self._cache))
        self._remove_entry(lru_key)
        self.stats.evictions += 1
        
        self.logger.debug(f"Evicted LRU entry: {lru_key}")
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache"""
        if key in self._cache:
            del self._cache[key]
            self.stats.total_size = len(self._cache)
    
    def _cleanup_expired(self) -> None:
        """Background thread to cleanup expired entries"""
        while self._cleanup_running:
            try:
                time.sleep(60)  # Check every minute
                self.cleanup_expired()
            except Exception as e:
                self.logger.error(f"Error in cache cleanup thread: {e}")
    
    def _save_to_file(self, key: str, value: Any, ttl: int) -> None:
        """Save cache entry to file"""
        if not self.cache_dir:
            return
        
        try:
            filename = f"{key.replace('/', '_').replace(':', '_')}.cache"
            filepath = os.path.join(self.cache_dir, filename)
            
            cache_data = {
                "data": value,
                "timestamp": time.time(),
                "ttl": ttl
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            self.logger.warning(f"Failed to save cache to file {key}: {e}")
    
    def _load_from_file(self, key: str) -> Optional[Any]:
        """Load cache entry from file"""
        if not self.cache_dir:
            return None
        
        try:
            filename = f"{key.replace('/', '_').replace(':', '_')}.cache"
            filepath = os.path.join(self.cache_dir, filename)
            
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if expired
            if cache_data["ttl"] > 0:
                age = time.time() - cache_data["timestamp"]
                if age > cache_data["ttl"]:
                    os.remove(filepath)  # Remove expired file
                    return None
            
            # Add back to memory cache
            self.set(key, cache_data["data"], cache_data["ttl"])
            
            return cache_data["data"]
            
        except Exception as e:
            self.logger.warning(f"Failed to load cache from file {key}: {e}")
            return None
    
    def _delete_file(self, key: str) -> None:
        """Delete cache file"""
        if not self.cache_dir:
            return
        
        try:
            filename = f"{key.replace('/', '_').replace(':', '_')}.cache"
            filepath = os.path.join(self.cache_dir, filename)
            
            if os.path.exists(filepath):
                os.remove(filepath)
                
        except Exception as e:
            self.logger.warning(f"Failed to delete cache file {key}: {e}")
    
    def shutdown(self) -> None:
        """Shutdown cache manager and cleanup resources"""
        self._cleanup_running = False
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        self.logger.info("Cache Manager shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during destruction