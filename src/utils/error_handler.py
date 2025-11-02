"""
Error Handler Utility
====================

Provides centralized error handling and logging for the trip planner application.
Handles different error types, retry logic, and structured error reporting.

Key Features:
- Centralized error categorization and handling
- Automatic retry logic for transient failures
- Structured error logging with context information
- Error recovery strategies and fallback mechanisms
- Integration with application logging system
- Error statistics and monitoring

Classes:
    ErrorHandler: Main error handling interface
    ErrorCategory: Enumeration of error categories
    ErrorContext: Context information for errors
    RetryConfig: Configuration for retry behavior

Author: Hybrid Trip Planner Team
"""

import logging
import time
import traceback
from typing import Any, Optional, Dict, Callable, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import functools

# Import configuration
from config import config


class ErrorCategory(Enum):
    """
    Error categories for classification
    """
    # Data-related errors
    DATA_VALIDATION = "data_validation"
    DATA_NOT_FOUND = "data_not_found"
    DATA_CORRUPTION = "data_corruption"
    
    # API-related errors
    API_CONNECTION = "api_connection"
    API_TIMEOUT = "api_timeout"
    API_RATE_LIMIT = "api_rate_limit"
    API_AUTHENTICATION = "api_authentication"
    API_QUOTA_EXCEEDED = "api_quota_exceeded"
    
    # Configuration errors
    CONFIG_MISSING = "config_missing"
    CONFIG_INVALID = "config_invalid"
    ENV_VAR_MISSING = "env_var_missing"
    
    # Processing errors
    CLUSTERING_FAILED = "clustering_failed"
    ROUTING_FAILED = "routing_failed"
    AI_GENERATION_FAILED = "ai_generation_failed"
    
    # System errors
    NETWORK_ERROR = "network_error"
    FILE_SYSTEM_ERROR = "file_system_error"
    MEMORY_ERROR = "memory_error"
    
    # Unknown errors
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """
    Error severity levels
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """
    Context information for errors
    
    Attributes:
        module (str): Module where error occurred
        function (str): Function where error occurred
        user_input (Dict): User input that caused the error
        system_state (Dict): Relevant system state
        timestamp (datetime): When error occurred
        request_id (str): Unique identifier for the request
    """
    module: str
    function: str
    user_input: Optional[Dict] = None
    system_state: Optional[Dict] = None
    timestamp: Optional[datetime] = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior
    
    Attributes:
        max_attempts (int): Maximum number of retry attempts
        base_delay (float): Base delay between retries in seconds
        max_delay (float): Maximum delay between retries
        exponential_backoff (bool): Whether to use exponential backoff
        jitter (bool): Whether to add random jitter to delays
        retryable_errors (List[ErrorCategory]): Error categories that should trigger retry
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_backoff: bool = True
    jitter: bool = True
    retryable_errors: List[ErrorCategory] = None
    
    def __post_init__(self):
        if self.retryable_errors is None:
            self.retryable_errors = [
                ErrorCategory.API_CONNECTION,
                ErrorCategory.API_TIMEOUT,
                ErrorCategory.API_RATE_LIMIT,
                ErrorCategory.NETWORK_ERROR
            ]


@dataclass
class ErrorReport:
    """
    Structured error report
    
    Attributes:
        category (ErrorCategory): Error category
        severity (ErrorSeverity): Error severity
        message (str): Human-readable error message
        technical_details (str): Technical error details
        context (ErrorContext): Error context information
        stack_trace (str): Stack trace if available
        recovery_suggestions (List[str]): Suggested recovery actions
        occurred_at (datetime): When error occurred
    """
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    technical_details: str
    context: ErrorContext
    stack_trace: Optional[str] = None
    recovery_suggestions: Optional[List[str]] = None
    occurred_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.occurred_at is None:
            self.occurred_at = datetime.now()
        if self.recovery_suggestions is None:
            self.recovery_suggestions = []


class ErrorHandler:
    """
    Main error handling interface
    Provides centralized error management with logging, retry logic, and reporting
    """
    
    def __init__(self):
        """Initialize Error Handler"""
        self.logger = logging.getLogger(__name__)
        
        # Error statistics
        self._error_counts = {}
        self._total_errors = 0
        
        # Default retry configuration
        self.default_retry_config = RetryConfig()
        
        self.logger.info("Error Handler initialized")
    
    def handle_error(self, message: str, exception: Exception = None, 
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: ErrorContext = None,
                    raise_exception: bool = False) -> ErrorReport:
        """
        Handle an error with logging and reporting
        
        Args:
            message (str): Human-readable error message
            exception (Exception): Original exception if available
            category (ErrorCategory): Error category
            severity (ErrorSeverity): Error severity
            context (ErrorContext): Error context
            raise_exception (bool): Whether to re-raise the exception
            
        Returns:
            ErrorReport: Structured error report
        """
        # Update statistics
        self._total_errors += 1
        self._error_counts[category] = self._error_counts.get(category, 0) + 1
        
        # Get technical details and stack trace
        technical_details = str(exception) if exception else "No exception details"
        stack_trace = traceback.format_exc() if exception else None
        
        # Create error report
        error_report = ErrorReport(
            category=category,
            severity=severity,
            message=message,
            technical_details=technical_details,
            context=context or ErrorContext(module="unknown", function="unknown"),
            stack_trace=stack_trace,
            recovery_suggestions=self._get_recovery_suggestions(category)
        )
        
        # Log the error
        self._log_error(error_report)
        
        # Re-raise if requested
        if raise_exception and exception:
            raise exception
        
        return error_report
    
    def handle_api_error(self, api_name: str, endpoint: str, status_code: int = None,
                        response_text: str = None, exception: Exception = None) -> ErrorReport:
        """
        Handle API-specific errors
        
        Args:
            api_name (str): Name of the API
            endpoint (str): API endpoint
            status_code (int): HTTP status code
            response_text (str): Response text
            exception (Exception): Original exception
            
        Returns:
            ErrorReport: Structured error report
        """
        # Determine error category based on status code
        if status_code:
            if status_code == 401:
                category = ErrorCategory.API_AUTHENTICATION
            elif status_code == 403:
                category = ErrorCategory.API_QUOTA_EXCEEDED
            elif status_code == 429:
                category = ErrorCategory.API_RATE_LIMIT
            elif 500 <= status_code < 600:
                category = ErrorCategory.API_CONNECTION
            else:
                category = ErrorCategory.API_CONNECTION
        else:
            category = ErrorCategory.API_CONNECTION
        
        # Create context
        context = ErrorContext(
            module="api_client",
            function=f"{api_name}_request",
            system_state={
                "api_name": api_name,
                "endpoint": endpoint,
                "status_code": status_code,
                "response_text": response_text[:500] if response_text else None
            }
        )
        
        # Create message
        message = f"{api_name} API error"
        if status_code:
            message += f" (HTTP {status_code})"
        
        return self.handle_error(
            message=message,
            exception=exception,
            category=category,
            severity=ErrorSeverity.HIGH if status_code and status_code >= 500 else ErrorSeverity.MEDIUM,
            context=context
        )
    
    def handle_data_error(self, data_type: str, issue: str, 
                         data_sample: Any = None) -> ErrorReport:
        """
        Handle data validation and processing errors
        
        Args:
            data_type (str): Type of data (e.g., "POI", "weather")
            issue (str): Description of the issue
            data_sample (Any): Sample of problematic data
            
        Returns:
            ErrorReport: Structured error report
        """
        context = ErrorContext(
            module="data_processing",
            function="data_validation",
            system_state={
                "data_type": data_type,
                "data_sample": str(data_sample)[:200] if data_sample else None
            }
        )
        
        return self.handle_error(
            message=f"Data validation error for {data_type}: {issue}",
            category=ErrorCategory.DATA_VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )
    
    def retry_on_error(self, func: Callable, *args, 
                      retry_config: RetryConfig = None, **kwargs) -> Any:
        """
        Execute function with retry logic
        
        Args:
            func (Callable): Function to execute
            *args: Function arguments
            retry_config (RetryConfig): Retry configuration
            **kwargs: Function keyword arguments
            
        Returns:
            Any: Function result
            
        Raises:
            Exception: If all retry attempts fail
        """
        config = retry_config or self.default_retry_config
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                result = func(*args, **kwargs)
                
                # Log successful retry if not first attempt
                if attempt > 0:
                    self.logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Determine if this error should trigger retry
                error_category = self._categorize_exception(e)
                if error_category not in config.retryable_errors:
                    self.logger.warning(f"Error {error_category} not retryable, failing immediately")
                    break
                
                # Check if this is the last attempt
                if attempt == config.max_attempts - 1:
                    break
                
                # Calculate delay
                delay = self._calculate_retry_delay(attempt, config)
                
                self.logger.warning(
                    f"Function {func.__name__} failed on attempt {attempt + 1}, "
                    f"retrying in {delay:.1f}s: {str(e)}"
                )
                
                time.sleep(delay)
        
        # All attempts failed
        self.handle_error(
            message=f"Function {func.__name__} failed after {config.max_attempts} attempts",
            exception=last_exception,
            category=self._categorize_exception(last_exception),
            severity=ErrorSeverity.HIGH
        )
        
        raise last_exception
    
    def _log_error(self, error_report: ErrorReport) -> None:
        """Log error report"""
        log_message = (
            f"[{error_report.category.value}] {error_report.message}\n"
            f"Severity: {error_report.severity.value}\n"
            f"Technical: {error_report.technical_details}\n"
            f"Module: {error_report.context.module}.{error_report.context.function}"
        )
        
        if error_report.context.system_state:
            log_message += f"\nState: {error_report.context.system_state}"
        
        if error_report.recovery_suggestions:
            log_message += f"\nSuggestions: {', '.join(error_report.recovery_suggestions)}"
        
        # Log based on severity
        if error_report.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_report.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_report.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Log stack trace for high severity errors
        if (error_report.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] 
            and error_report.stack_trace):
            self.logger.debug(f"Stack trace:\n{error_report.stack_trace}")
    
    def _categorize_exception(self, exception: Exception) -> ErrorCategory:
        """Categorize exception into error category"""
        if isinstance(exception, ConnectionError):
            return ErrorCategory.NETWORK_ERROR
        elif isinstance(exception, TimeoutError):
            return ErrorCategory.API_TIMEOUT
        elif isinstance(exception, ValueError):
            return ErrorCategory.DATA_VALIDATION
        elif isinstance(exception, FileNotFoundError):
            return ErrorCategory.FILE_SYSTEM_ERROR
        elif isinstance(exception, MemoryError):
            return ErrorCategory.MEMORY_ERROR
        else:
            return ErrorCategory.UNKNOWN
    
    def _calculate_retry_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay before next retry attempt"""
        if config.exponential_backoff:
            delay = config.base_delay * (2 ** attempt)
        else:
            delay = config.base_delay
        
        # Apply maximum delay
        delay = min(delay, config.max_delay)
        
        # Add jitter if enabled
        if config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
        
        return delay
    
    def _get_recovery_suggestions(self, category: ErrorCategory) -> List[str]:
        """Get recovery suggestions for error category"""
        suggestions = {
            ErrorCategory.API_CONNECTION: [
                "Check internet connection",
                "Verify API endpoint is accessible",
                "Try again in a few minutes"
            ],
            ErrorCategory.API_TIMEOUT: [
                "Increase timeout settings",
                "Check network latency",
                "Retry with smaller request size"
            ],
            ErrorCategory.API_RATE_LIMIT: [
                "Wait before making more requests",
                "Implement request throttling",
                "Consider upgrading API plan"
            ],
            ErrorCategory.API_AUTHENTICATION: [
                "Check API key validity",
                "Verify API key permissions",
                "Regenerate API key if needed"
            ],
            ErrorCategory.API_QUOTA_EXCEEDED: [
                "Check API usage limits",
                "Wait for quota reset",
                "Consider upgrading API plan"
            ],
            ErrorCategory.DATA_VALIDATION: [
                "Check input data format",
                "Verify data completeness",
                "Review validation rules"
            ],
            ErrorCategory.CONFIG_MISSING: [
                "Check configuration file exists",
                "Verify environment variables",
                "Review required settings"
            ]
        }
        
        return suggestions.get(category, ["Review error details", "Contact support if issue persists"])
    
    def get_error_statistics(self) -> Dict:
        """Get error statistics"""
        return {
            "total_errors": self._total_errors,
            "errors_by_category": dict(self._error_counts),
            "most_common_error": max(self._error_counts, key=self._error_counts.get) if self._error_counts else None
        }
    
    def reset_statistics(self) -> None:
        """Reset error statistics"""
        self._error_counts.clear()
        self._total_errors = 0
        self.logger.info("Error statistics reset")


def retry_on_failure(max_attempts: int = 3, base_delay: float = 1.0,
                    retryable_errors: List[ErrorCategory] = None):
    """
    Decorator for automatic retry on function failure
    
    Args:
        max_attempts (int): Maximum retry attempts
        base_delay (float): Base delay between retries
        retryable_errors (List[ErrorCategory]): Error categories to retry on
        
    Returns:
        Function decorator
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            retry_config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                retryable_errors=retryable_errors
            )
            return error_handler.retry_on_error(func, *args, retry_config=retry_config, **kwargs)
        return wrapper
    return decorator


def handle_exceptions(category: ErrorCategory = ErrorCategory.UNKNOWN,
                     severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     raise_exception: bool = False):
    """
    Decorator for automatic exception handling
    
    Args:
        category (ErrorCategory): Error category
        severity (ErrorSeverity): Error severity
        raise_exception (bool): Whether to re-raise exceptions
        
    Returns:
        Function decorator
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = ErrorHandler()
                context = ErrorContext(
                    module=func.__module__,
                    function=func.__name__
                )
                error_handler.handle_error(
                    message=f"Exception in {func.__name__}",
                    exception=e,
                    category=category,
                    severity=severity,
                    context=context,
                    raise_exception=raise_exception
                )
                return None  # Return None if not re-raising
        return wrapper
    return decorator