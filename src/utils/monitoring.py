"""
Monitoring and observability utilities.

This module provides structured logging, error tracking with Sentry,
and performance monitoring for the Discord bot.
"""

import os
import logging
import time
import asyncio
from typing import Any, Dict, Optional, Callable
from functools import wraps
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Try to import Sentry
try:
    import sentry_sdk
    from sentry_sdk.integrations.asyncio import AsyncioIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    sentry_sdk = None

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

@dataclass
class MonitoringConfig:
    """Configuration for monitoring features."""
    sentry_dsn: Optional[str] = None
    environment: str = "development"
    sample_rate: float = 1.0  # 100% of events
    traces_sample_rate: float = 0.1  # 10% of transactions
    log_level: str = "INFO"
    structured_logging: bool = True


def setup_monitoring(config: Optional[MonitoringConfig] = None) -> None:
    """
    Initialize monitoring with optional Sentry integration.
    
    Args:
        config: Monitoring configuration, uses env vars if not provided
    """
    if config is None:
        config = MonitoringConfig(
            sentry_dsn=os.environ.get("SENTRY_DSN"),
            environment=os.environ.get("ENVIRONMENT", "development"),
            sample_rate=float(os.environ.get("SENTRY_SAMPLE_RATE", "1.0")),
            traces_sample_rate=float(os.environ.get("SENTRY_TRACES_RATE", "0.1")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )
    
    # Setup logging
    setup_structured_logging(
        level=config.log_level,
        structured=config.structured_logging
    )
    
    # Setup Sentry if available and configured
    if SENTRY_AVAILABLE and config.sentry_dsn:
        sentry_sdk.init(
            dsn=config.sentry_dsn,
            environment=config.environment,
            sample_rate=config.sample_rate,
            traces_sample_rate=config.traces_sample_rate,
            integrations=[AsyncioIntegration()],
            before_send=before_send_filter,
        )
        logger.info(f"Sentry initialized for environment: {config.environment}")
    else:
        if config.sentry_dsn and not SENTRY_AVAILABLE:
            logger.warning("Sentry DSN provided but sentry_sdk not installed")
        logger.info("Running without Sentry error tracking")


def before_send_filter(event: Dict, hint: Dict) -> Optional[Dict]:
    """Filter events before sending to Sentry."""
    # Don't send events for expected/handled errors
    if "exc_info" in hint:
        exc_type, exc_value, _ = hint["exc_info"]
        
        # Skip common non-critical errors
        if exc_type.__name__ in [
            "NotFound",  # Discord 404
            "Forbidden",  # Discord 403
            "RateLimited",  # Discord rate limit
        ]:
            return None
    
    return event


# ============================================================
# Structured Logging
# ============================================================

class StructuredFormatter(logging.Formatter):
    """JSON-like structured log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured message."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        if hasattr(record, "guild_id"):
            log_entry["guild_id"] = record.guild_id
        if hasattr(record, "command"):
            log_entry["command"] = record.command
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms
        if hasattr(record, "model"):
            log_entry["model"] = record.model
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Format as key=value pairs for easy parsing
        parts = [f"{k}={v!r}" for k, v in log_entry.items()]
        return " ".join(parts)


def setup_structured_logging(
    level: str = "INFO",
    structured: bool = True
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        structured: Use structured formatting
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create handler
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    
    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [handler]


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


# ============================================================
# Error Tracking
# ============================================================

def capture_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Capture and report an exception.
    
    Args:
        exception: The exception to capture
        context: Additional context to attach
        
    Returns:
        Event ID if sent to Sentry, None otherwise
    """
    logger.exception(f"Captured exception: {exception}")
    
    if SENTRY_AVAILABLE and sentry_sdk.Hub.current.client:
        with sentry_sdk.push_scope() as scope:
            if context:
                for key, value in context.items():
                    scope.set_extra(key, value)
            return sentry_sdk.capture_exception(exception)
    
    return None


def capture_message(
    message: str,
    level: str = "info",
    context: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Capture and report a message.
    
    Args:
        message: The message to capture
        level: Severity level (debug, info, warning, error, fatal)
        context: Additional context to attach
        
    Returns:
        Event ID if sent to Sentry, None otherwise
    """
    log_method = getattr(logger, level, logger.info)
    log_method(message)
    
    if SENTRY_AVAILABLE and sentry_sdk.Hub.current.client:
        with sentry_sdk.push_scope() as scope:
            if context:
                for key, value in context.items():
                    scope.set_extra(key, value)
            return sentry_sdk.capture_message(message, level=level)
    
    return None


def set_user_context(
    user_id: int,
    username: Optional[str] = None,
    guild_id: Optional[int] = None
) -> None:
    """
    Set user context for error tracking.
    
    Args:
        user_id: Discord user ID
        username: Discord username
        guild_id: Discord guild ID
    """
    if SENTRY_AVAILABLE and sentry_sdk.Hub.current.client:
        sentry_sdk.set_user({
            "id": str(user_id),
            "username": username,
        })
        if guild_id:
            sentry_sdk.set_tag("guild_id", str(guild_id))


# ============================================================
# Performance Monitoring
# ============================================================

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        end = self.end_time or time.perf_counter()
        return (end - self.start_time) * 1000
    
    def finish(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark the operation as finished."""
        self.end_time = time.perf_counter()
        self.success = success
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
            "error": self.error,
            **self.metadata
        }


@contextmanager
def measure_sync(name: str, **metadata):
    """
    Context manager to measure synchronous operation performance.
    
    Usage:
        with measure_sync("database_query", table="users"):
            result = db.query(...)
    """
    metrics = PerformanceMetrics(name=name, metadata=metadata)
    
    try:
        yield metrics
        metrics.finish(success=True)
    except Exception as e:
        metrics.finish(success=False, error=str(e))
        raise
    finally:
        logger.info(
            f"Performance: {metrics.name}",
            extra={"duration_ms": metrics.duration_ms, **metrics.metadata}
        )


@asynccontextmanager
async def measure_async(name: str, **metadata):
    """
    Async context manager to measure async operation performance.
    
    Usage:
        async with measure_async("api_call", endpoint="chat"):
            result = await api.call(...)
    """
    metrics = PerformanceMetrics(name=name, metadata=metadata)
    
    # Start Sentry transaction if available
    transaction = None
    if SENTRY_AVAILABLE and sentry_sdk.Hub.current.client:
        transaction = sentry_sdk.start_transaction(
            op="task",
            name=name
        )
    
    try:
        yield metrics
        metrics.finish(success=True)
    except Exception as e:
        metrics.finish(success=False, error=str(e))
        raise
    finally:
        if transaction:
            transaction.set_status("ok" if metrics.success else "internal_error")
            transaction.finish()
        
        logger.info(
            f"Performance: {metrics.name}",
            extra={"duration_ms": metrics.duration_ms, **metrics.metadata}
        )


def track_performance(name: Optional[str] = None):
    """
    Decorator to track async function performance.
    
    Args:
        name: Operation name (defaults to function name)
        
    Usage:
        @track_performance("process_message")
        async def handle_message(message):
            ...
    """
    def decorator(func: Callable):
        op_name = name or func.__name__
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with measure_async(op_name):
                return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# ============================================================
# Health Check
# ============================================================

@dataclass
class HealthStatus:
    """Health check status."""
    healthy: bool
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    
    def add_check(
        self,
        name: str,
        healthy: bool,
        message: str = "",
        details: Optional[Dict] = None
    ) -> None:
        """Add a health check result."""
        self.checks[name] = {
            "healthy": healthy,
            "message": message,
            **(details or {})
        }
        if not healthy:
            self.healthy = False


async def check_health(
    db_handler=None,
    openai_client=None
) -> HealthStatus:
    """
    Perform health checks on bot dependencies.
    
    Args:
        db_handler: Database handler to check
        openai_client: OpenAI client to check
        
    Returns:
        HealthStatus with check results
    """
    status = HealthStatus(healthy=True)
    
    # Check database
    if db_handler:
        try:
            # Simple ping or list operation
            await asyncio.wait_for(
                db_handler.client.admin.command('ping'),
                timeout=5.0
            )
            status.add_check("database", True, "MongoDB connected")
        except Exception as e:
            status.add_check("database", False, f"MongoDB error: {e}")
    
    # Check OpenAI
    if openai_client:
        try:
            # List models as a simple check
            await asyncio.wait_for(
                openai_client.models.list(),
                timeout=10.0
            )
            status.add_check("openai", True, "OpenAI API accessible")
        except Exception as e:
            status.add_check("openai", False, f"OpenAI error: {e}")
    
    return status
