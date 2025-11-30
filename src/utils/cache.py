"""
Simple caching utilities for API responses and frequently accessed data.

This module provides an in-memory LRU cache with optional TTL (time-to-live)
support, designed for caching API responses and reducing redundant calls.
"""

import asyncio
import time
import logging
from typing import Any, Dict, Optional, Callable, TypeVar, Generic
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with value and expiration time."""
    value: T
    expires_at: float
    created_at: float = field(default_factory=time.time)
    hits: int = 0


class LRUCache(Generic[T]):
    """
    Thread-safe LRU (Least Recently Used) cache with TTL support.
    
    Features:
        - Configurable max size with automatic eviction
        - Per-entry TTL (time-to-live)
        - Automatic cleanup of expired entries
        - Hit/miss statistics tracking
        
    Usage:
        cache = LRUCache(max_size=1000, default_ttl=300)  # 5 min TTL
        cache.set("key", "value")
        value = cache.get("key")  # Returns value or None if expired
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 300.0,  # 5 minutes default
        cleanup_interval: float = 60.0
    ):
        """
        Initialize the LRU cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
            cleanup_interval: How often to run cleanup (seconds)
        """
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        self._lock = asyncio.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug("Cache cleanup task started")
    
    async def stop(self) -> None:
        """Stop the background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.debug("Cache cleanup task stopped")
    
    async def _cleanup_loop(self) -> None:
        """Background task to periodically clean up expired entries."""
        while True:
            await asyncio.sleep(self._cleanup_interval)
            await self._cleanup_expired()
    
    async def _cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        now = time.time()
        removed = 0
        
        async with self._lock:
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if entry.expires_at <= now
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
                removed += 1
        
        if removed > 0:
            logger.debug(f"Cache cleanup: removed {removed} expired entries")
        
        return removed
    
    async def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.expires_at <= time.time():
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._hits += 1
            
            return entry.value
    
    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None
    ) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override (uses default if not provided)
        """
        ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.time() + ttl
        
        async with self._lock:
            # Remove oldest entries if at capacity
            while len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Cache evicted oldest entry: {oldest_key}")
            
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=expires_at
            )
            self._cache.move_to_end(key)
    
    async def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was found and deleted
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> int:
        """
        Clear all entries from the cache.
        
        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    async def has(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        return await self.get(key) is not None
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with size, hits, misses, hit_rate
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "default_ttl": self._default_ttl
        }


# Global cache instances for different purposes
_api_response_cache: Optional[LRUCache[Dict[str, Any]]] = None
_user_preference_cache: Optional[LRUCache[Dict[str, Any]]] = None


async def get_api_cache() -> LRUCache[Dict[str, Any]]:
    """Get or create the API response cache."""
    global _api_response_cache
    if _api_response_cache is None:
        _api_response_cache = LRUCache(
            max_size=500,
            default_ttl=300.0  # 5 minutes
        )
        await _api_response_cache.start()
    return _api_response_cache


async def get_user_cache() -> LRUCache[Dict[str, Any]]:
    """Get or create the user preference cache."""
    global _user_preference_cache
    if _user_preference_cache is None:
        _user_preference_cache = LRUCache(
            max_size=1000,
            default_ttl=600.0  # 10 minutes
        )
        await _user_preference_cache.start()
    return _user_preference_cache


def cached(
    cache_key_func: Callable[..., str],
    ttl: Optional[float] = None,
    cache_getter: Callable = get_api_cache
):
    """
    Decorator to cache async function results.
    
    Args:
        cache_key_func: Function to generate cache key from args
        ttl: Optional TTL override
        cache_getter: Function to get the cache instance
        
    Usage:
        @cached(
            cache_key_func=lambda user_id: f"user:{user_id}",
            ttl=300
        )
        async def get_user_data(user_id: int) -> dict:
            # Expensive operation
            return await fetch_from_api(user_id)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = await cache_getter()
            key = cache_key_func(*args, **kwargs)
            
            # Try to get from cache
            cached_value = await cache.get(key)
            if cached_value is not None:
                logger.debug(f"Cache hit for key: {key}")
                return cached_value
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(key, result, ttl=ttl)
            logger.debug(f"Cached result for key: {key}")
            
            return result
        
        return wrapper
    return decorator


def invalidate_on_update(
    cache_key_func: Callable[..., str],
    cache_getter: Callable = get_api_cache
):
    """
    Decorator to invalidate cache when a function (update operation) is called.
    
    Args:
        cache_key_func: Function to generate cache key to invalidate
        cache_getter: Function to get the cache instance
        
    Usage:
        @invalidate_on_update(
            cache_key_func=lambda user_id, **_: f"user:{user_id}"
        )
        async def update_user_data(user_id: int, data: dict) -> None:
            await save_to_db(user_id, data)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidate cache after update
            cache = await cache_getter()
            key = cache_key_func(*args, **kwargs)
            await cache.delete(key)
            logger.debug(f"Invalidated cache for key: {key}")
            
            return result
        
        return wrapper
    return decorator


# Convenience functions for common caching patterns

async def cache_user_model(user_id: int, model: str) -> None:
    """Cache user's selected model."""
    cache = await get_user_cache()
    await cache.set(f"user_model:{user_id}", {"model": model})


async def get_cached_user_model(user_id: int) -> Optional[str]:
    """Get user's cached model selection."""
    cache = await get_user_cache()
    result = await cache.get(f"user_model:{user_id}")
    return result["model"] if result else None


async def invalidate_user_cache(user_id: int) -> None:
    """Invalidate all cached data for a user."""
    cache = await get_user_cache()
    # Clear known user-related keys
    await cache.delete(f"user_model:{user_id}")
    await cache.delete(f"user_history:{user_id}")
    await cache.delete(f"user_stats:{user_id}")
