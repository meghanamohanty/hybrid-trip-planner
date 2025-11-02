"""
Performance Monitoring Utilities
===============================

Performance monitoring and profiling tools for the trip planner application.
Provides timing decorators, resource monitoring, and performance analysis.

Key Features:
- Function execution timing with decorators
- Memory and CPU usage monitoring
- Performance statistics collection and reporting
- Bottleneck identification and analysis
- Configurable performance thresholds and alerts
- Performance data export and visualization

Classes:
    PerformanceMonitor: Main performance monitoring interface
    TimingStats: Statistics for function execution times
    ResourceMonitor: System resource usage monitoring
    PerformanceReport: Performance analysis report

Functions:
    measure_time: Decorator for measuring function execution time
    monitor_resources: Decorator for monitoring resource usage
    profile_function: Detailed profiling of function execution

Author: Hybrid Trip Planner Team
"""

import time
import logging
import functools
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import os

# Import configuration
from config import config


@dataclass
class TimingStats:
    """
    Statistics for function execution times
    
    Attributes:
        function_name (str): Name of the function
        call_count (int): Number of times function was called
        total_time (float): Total execution time in seconds
        min_time (float): Minimum execution time
        max_time (float): Maximum execution time
        avg_time (float): Average execution time
        recent_times (deque): Recent execution times (sliding window)
        last_called (datetime): When function was last called
    """
    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_called: Optional[datetime] = None
    
    @property
    def avg_time(self) -> float:
        """Calculate average execution time"""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    @property
    def recent_avg_time(self) -> float:
        """Calculate average of recent execution times"""
        return sum(self.recent_times) / len(self.recent_times) if self.recent_times else 0.0
    
    def add_timing(self, execution_time: float) -> None:
        """Add a new timing measurement"""
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.recent_times.append(execution_time)
        self.last_called = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "function_name": self.function_name,
            "call_count": self.call_count,
            "total_time": round(self.total_time, 4),
            "min_time": round(self.min_time, 4) if self.min_time != float('inf') else 0,
            "max_time": round(self.max_time, 4),
            "avg_time": round(self.avg_time, 4),
            "recent_avg_time": round(self.recent_avg_time, 4),
            "last_called": self.last_called.isoformat() if self.last_called else None
        }


@dataclass
class ResourceUsage:
    """
    System resource usage snapshot
    
    Attributes:
        timestamp (datetime): When measurement was taken
        cpu_percent (float): CPU usage percentage
        memory_mb (float): Memory usage in MB
        memory_percent (float): Memory usage percentage
        disk_io_read (int): Disk read bytes
        disk_io_write (int): Disk write bytes
        network_sent (int): Network bytes sent
        network_recv (int): Network bytes received
    """
    timestamp: datetime
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    disk_io_read: int = 0
    disk_io_write: int = 0
    network_sent: int = 0
    network_recv: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": round(self.cpu_percent, 2),
            "memory_mb": round(self.memory_mb, 2),
            "memory_percent": round(self.memory_percent, 2),
            "disk_io_read": self.disk_io_read,
            "disk_io_write": self.disk_io_write,
            "network_sent": self.network_sent,
            "network_recv": self.network_recv
        }


class ResourceMonitor:
    """
    System resource usage monitoring
    """
    
    def __init__(self, sample_interval: float = 1.0, max_samples: int = 1000):
        """
        Initialize Resource Monitor
        
        Args:
            sample_interval (float): Interval between samples in seconds
            max_samples (int): Maximum number of samples to keep
        """
        self.logger = logging.getLogger(__name__)
        self.sample_interval = sample_interval
        self.max_samples = max_samples
        
        # Resource usage history
        self.usage_history: deque = deque(maxlen=max_samples)
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Process reference for monitoring
        self.process = psutil.Process()
        
        # Initial I/O and network counters
        try:
            self._initial_io = psutil.disk_io_counters()
            self._initial_net = psutil.net_io_counters()
        except:
            self._initial_io = None
            self._initial_net = None
    
    def start_monitoring(self) -> None:
        """Start background resource monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background resource monitoring"""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        self.logger.info("Resource monitoring stopped")
    
    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage snapshot"""
        try:
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert bytes to MB
            memory_percent = self.process.memory_percent()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read = disk_io.read_bytes if disk_io else 0
            disk_write = disk_io.write_bytes if disk_io else 0
            
            # Network I/O
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent if net_io else 0
            net_recv = net_io.bytes_recv if net_io else 0
            
            return ResourceUsage(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                disk_io_read=disk_read,
                disk_io_write=disk_write,
                network_sent=net_sent,
                network_recv=net_recv
            )
            
        except Exception as e:
            self.logger.warning(f"Error getting resource usage: {e}")
            return ResourceUsage(timestamp=datetime.now())
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while self._monitoring:
            try:
                usage = self.get_current_usage()
                self.usage_history.append(usage)
                time.sleep(self.sample_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sample_interval)
    
    def get_usage_summary(self, duration_minutes: int = 10) -> Dict:
        """
        Get resource usage summary for specified duration
        
        Args:
            duration_minutes (int): Duration in minutes to analyze
            
        Returns:
            Dict: Usage summary statistics
        """
        if not self.usage_history:
            return {}
        
        # Filter samples by time
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_samples = [usage for usage in self.usage_history 
                         if usage.timestamp >= cutoff_time]
        
        if not recent_samples:
            return {}
        
        # Calculate statistics
        cpu_values = [usage.cpu_percent for usage in recent_samples]
        memory_values = [usage.memory_mb for usage in recent_samples]
        memory_percent_values = [usage.memory_percent for usage in recent_samples]
        
        return {
            "duration_minutes": duration_minutes,
            "sample_count": len(recent_samples),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values)
            },
            "memory_mb": {
                "avg": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values),
                "current": memory_values[-1] if memory_values else 0
            },
            "memory_percent": {
                "avg": sum(memory_percent_values) / len(memory_percent_values),
                "min": min(memory_percent_values),
                "max": max(memory_percent_values),
                "current": memory_percent_values[-1] if memory_percent_values else 0
            },
            "time_range": {
                "start": recent_samples[0].timestamp.isoformat(),
                "end": recent_samples[-1].timestamp.isoformat()
            }
        }


class PerformanceMonitor:
    """
    Main performance monitoring interface
    Collects timing statistics and manages performance analysis
    """
    
    def __init__(self):
        """Initialize Performance Monitor"""
        self.logger = logging.getLogger(__name__)
        
        # Timing statistics
        self._timing_stats: Dict[str, TimingStats] = {}
        self._lock = threading.RLock()
        
        # Resource monitor
        self.resource_monitor = ResourceMonitor()
        
        # Performance thresholds (in seconds)
        self.slow_function_threshold = 5.0
        self.very_slow_function_threshold = 10.0
        
        # Enable resource monitoring by default
        self.resource_monitor.start_monitoring()
        
        self.logger.info("Performance Monitor initialized")
    
    def record_timing(self, function_name: str, execution_time: float) -> None:
        """
        Record timing for a function
        
        Args:
            function_name (str): Name of the function
            execution_time (float): Execution time in seconds
        """
        with self._lock:
            if function_name not in self._timing_stats:
                self._timing_stats[function_name] = TimingStats(function_name)
            
            self._timing_stats[function_name].add_timing(execution_time)
            
            # Log slow functions
            if execution_time > self.very_slow_function_threshold:
                self.logger.warning(f"Very slow function: {function_name} took {execution_time:.2f}s")
            elif execution_time > self.slow_function_threshold:
                self.logger.info(f"Slow function: {function_name} took {execution_time:.2f}s")
    
    def get_timing_stats(self, function_name: str = None) -> Dict:
        """
        Get timing statistics
        
        Args:
            function_name (str): Specific function name, or None for all
            
        Returns:
            Dict: Timing statistics
        """
        with self._lock:
            if function_name:
                stats = self._timing_stats.get(function_name)
                return stats.to_dict() if stats else {}
            else:
                return {name: stats.to_dict() 
                       for name, stats in self._timing_stats.items()}
    
    def get_slowest_functions(self, limit: int = 10) -> List[Dict]:
        """
        Get list of slowest functions by average execution time
        
        Args:
            limit (int): Maximum number of functions to return
            
        Returns:
            List[Dict]: Sorted list of function statistics
        """
        with self._lock:
            sorted_stats = sorted(
                self._timing_stats.values(),
                key=lambda x: x.avg_time,
                reverse=True
            )
            
            return [stats.to_dict() for stats in sorted_stats[:limit]]
    
    def get_most_called_functions(self, limit: int = 10) -> List[Dict]:
        """
        Get list of most frequently called functions
        
        Args:
            limit (int): Maximum number of functions to return
            
        Returns:
            List[Dict]: Sorted list of function statistics
        """
        with self._lock:
            sorted_stats = sorted(
                self._timing_stats.values(),
                key=lambda x: x.call_count,
                reverse=True
            )
            
            return [stats.to_dict() for stats in sorted_stats[:limit]]
    
    def reset_stats(self) -> None:
        """Reset all performance statistics"""
        with self._lock:
            self._timing_stats.clear()
            self.logger.info("Performance statistics reset")
    
    def generate_performance_report(self) -> Dict:
        """
        Generate comprehensive performance report
        
        Returns:
            Dict: Performance report with timing and resource data
        """
        with self._lock:
            # Function timing analysis
            total_functions = len(self._timing_stats)
            total_calls = sum(stats.call_count for stats in self._timing_stats.values())
            total_time = sum(stats.total_time for stats in self._timing_stats.values())
            
            # Resource usage summary
            resource_summary = self.resource_monitor.get_usage_summary(duration_minutes=10)
            
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_functions_monitored": total_functions,
                    "total_function_calls": total_calls,
                    "total_execution_time_seconds": round(total_time, 2),
                    "average_call_time": round(total_time / total_calls, 4) if total_calls > 0 else 0
                },
                "slowest_functions": self.get_slowest_functions(limit=5),
                "most_called_functions": self.get_most_called_functions(limit=5),
                "resource_usage": resource_summary,
                "performance_alerts": self._generate_performance_alerts()
            }
            
            return report
    
    def _generate_performance_alerts(self) -> List[Dict]:
        """Generate performance alerts based on thresholds"""
        alerts = []
        
        with self._lock:
            for stats in self._timing_stats.values():
                # Alert for consistently slow functions
                if stats.avg_time > self.slow_function_threshold:
                    alerts.append({
                        "type": "slow_function",
                        "function": stats.function_name,
                        "avg_time": round(stats.avg_time, 2),
                        "call_count": stats.call_count,
                        "message": f"Function {stats.function_name} averages {stats.avg_time:.2f}s per call"
                    })
                
                # Alert for functions with high variance
                if len(stats.recent_times) >= 10:
                    recent_avg = stats.recent_avg_time
                    if recent_avg > stats.avg_time * 1.5:  # 50% slower than average
                        alerts.append({
                            "type": "performance_degradation",
                            "function": stats.function_name,
                            "recent_avg": round(recent_avg, 2),
                            "overall_avg": round(stats.avg_time, 2),
                            "message": f"Function {stats.function_name} performance has degraded"
                        })
        
        return alerts
    
    def shutdown(self) -> None:
        """Shutdown performance monitor"""
        self.resource_monitor.stop_monitoring()
        self.logger.info("Performance Monitor shutdown")


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def measure_time(func: Callable = None, *, category: str = None) -> Callable:
    """
    Decorator to measure function execution time
    
    Args:
        func (Callable): Function to decorate
        category (str): Optional category for grouping functions
        
    Returns:
        Callable: Decorated function
        
    Examples:
        @measure_time
        def slow_function():
            time.sleep(1)
        
        @measure_time(category="api_calls")
        def api_call():
            pass
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Create function name with category if provided
                function_name = f"{category}.{f.__name__}" if category else f.__name__
                
                # Record timing
                _performance_monitor.record_timing(function_name, execution_time)
        
        return wrapper
    
    # Handle both @measure_time and @measure_time() usage
    if func is None:
        return decorator
    else:
        return decorator(func)


def monitor_resources(func: Callable) -> Callable:
    """
    Decorator to monitor resource usage during function execution
    
    Args:
        func (Callable): Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get resource usage before
        monitor = _performance_monitor.resource_monitor
        before_usage = monitor.get_current_usage()
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            after_usage = monitor.get_current_usage()
            
            # Calculate resource deltas
            memory_delta = after_usage.memory_mb - before_usage.memory_mb
            execution_time = end_time - start_time
            
            # Log significant resource usage
            if memory_delta > 10:  # More than 10MB increase
                logger = logging.getLogger(__name__)
                logger.info(f"Function {func.__name__} used {memory_delta:.1f}MB memory in {execution_time:.2f}s")
    
    return wrapper


def profile_function(func: Callable) -> Callable:
    """
    Decorator for detailed function profiling
    
    Args:
        func (Callable): Function to profile
        
    Returns:
        Callable: Decorated function with profiling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import cProfile
        import pstats
        import io
        
        # Create profiler
        profiler = cProfile.Profile()
        
        # Profile the function
        profiler.enable()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.disable()
            
            # Get profiling results
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)  # Top 10 functions
            
            # Log profiling results
            logger = logging.getLogger(__name__)
            logger.debug(f"Profiling results for {func.__name__}:\n{s.getvalue()}")
    
    return wrapper


def get_performance_stats() -> Dict:
    """Get current performance statistics"""
    return _performance_monitor.get_timing_stats()


def get_performance_report() -> Dict:
    """Generate performance report"""
    return _performance_monitor.generate_performance_report()


def reset_performance_stats() -> None:
    """Reset all performance statistics"""
    _performance_monitor.reset_stats()