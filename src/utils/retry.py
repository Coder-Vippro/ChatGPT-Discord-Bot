"""
Retry utilities with exponential backoff for API calls.

This module provides robust retry logic for external API calls
to handle transient failures gracefully.
"""

import asyncio
import logging
import random
from typing import TypeVar, Callable, Optional, Any, Type, Tuple
from functools import wraps

T = TypeVar('T')

# Default configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds
DEFAULT_EXPONENTIAL_BASE = 2


class RetryError(Exception):
    """Raised when all retry attempts have been exhausted."""
    
    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


async def async_retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    jitter: bool = True,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    **kwargs
) -> Any:
    """
    Execute an async function with exponential backoff retry.
    
    Args:
        func: The async function to execute
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        retryable_exceptions: Tuple of exception types that should trigger retry
        jitter: Whether to add randomness to delay
        on_retry: Optional callback called on each retry with (attempt, exception)
        **kwargs: Keyword arguments for the function
        
    Returns:
        The return value of the function
        
    Raises:
        RetryError: When all retries are exhausted
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e
            
            if attempt == max_retries:
                logging.error(f"All {max_retries} retries exhausted for {func.__name__}: {e}")
                raise RetryError(
                    f"Failed after {max_retries} retries: {str(e)}",
                    last_exception=e
                )
            
            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            
            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random())
            
            logging.warning(
                f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                f"after {delay:.2f}s delay. Error: {e}"
            )
            
            if on_retry:
                try:
                    on_retry(attempt + 1, e)
                except Exception as callback_error:
                    logging.warning(f"on_retry callback failed: {callback_error}")
            
            await asyncio.sleep(delay)
    
    # Should not reach here, but just in case
    raise RetryError("Unexpected retry loop exit", last_exception=last_exception)


def retry_decorator(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    jitter: bool = True
):
    """
    Decorator for adding retry logic to async functions.
    
    Usage:
        @retry_decorator(max_retries=3, base_delay=1.0)
        async def my_api_call():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await async_retry_with_backoff(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                retryable_exceptions=retryable_exceptions,
                jitter=jitter,
                **kwargs
            )
        return wrapper
    return decorator


# Common exception sets for different APIs
OPENAI_RETRYABLE_EXCEPTIONS = (
    # Add specific OpenAI exceptions as needed
    TimeoutError,
    ConnectionError,
)

DISCORD_RETRYABLE_EXCEPTIONS = (
    # Add specific Discord exceptions as needed
    TimeoutError,
    ConnectionError,
)

HTTP_RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    ConnectionResetError,
)


class RateLimiter:
    """
    Simple rate limiter for API calls.
    
    Usage:
        limiter = RateLimiter(calls_per_second=1)
        async with limiter:
            await make_api_call()
    """
    
    def __init__(self, calls_per_second: float = 1.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        async with self._lock:
            import time
            now = time.monotonic()
            time_since_last = now - self.last_call
            
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            
            self.last_call = time.monotonic()
            return self
    
    async def __aexit__(self, *args):
        pass


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascade failures.
    
    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Too many failures, requests are rejected immediately
        - HALF_OPEN: Testing if service recovered
    """
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_successes = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.
        
        Args:
            func: The async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            The function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        async with self._lock:
            await self._check_state()
            
            if self.state == self.OPEN:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _check_state(self):
        """Check and potentially update circuit state."""
        import time
        
        if self.state == self.OPEN:
            if time.monotonic() - self.last_failure_time >= self.recovery_timeout:
                logging.info("Circuit breaker transitioning to HALF_OPEN")
                self.state = self.HALF_OPEN
                self.half_open_successes = 0
    
    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            if self.state == self.HALF_OPEN:
                self.half_open_successes += 1
                if self.half_open_successes >= self.half_open_requests:
                    logging.info("Circuit breaker transitioning to CLOSED")
                    self.state = self.CLOSED
                    self.failure_count = 0
            elif self.state == self.CLOSED:
                self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call."""
        import time
        
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            
            if self.state == self.HALF_OPEN:
                logging.warning("Circuit breaker transitioning to OPEN (half-open failure)")
                self.state = self.OPEN
            elif self.failure_count >= self.failure_threshold:
                logging.warning(f"Circuit breaker transitioning to OPEN ({self.failure_count} failures)")
                self.state = self.OPEN
