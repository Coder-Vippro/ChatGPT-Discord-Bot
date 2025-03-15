from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict, Any, Optional
import functools
import asyncio
from datetime import datetime, timedelta

class DatabaseHandler:
    # Class-level cache for database results
    _cache = {}
    _cache_expiry = {}
    _cache_lock = asyncio.Lock()
    
    def __init__(self, mongodb_uri: str):
        """Initialize database connection with optimized settings"""
        # Set up a connection pool with sensible timeouts
        self.client = AsyncIOMotorClient(
            mongodb_uri,
            maxPoolSize=50,
            minPoolSize=10,
            maxIdleTimeMS=45000,
            connectTimeoutMS=2000,
            serverSelectionTimeoutMS=3000,
            waitQueueTimeoutMS=1000,
            retryWrites=True
        )
        self.db = self.client['chatgpt_discord_bot']  # Database name
        
    # Helper for caching results
    async def _get_cached_result(self, cache_key, fetch_func, expiry_seconds=60):
        """Get result from cache or execute fetch_func if not cached/expired"""
        current_time = datetime.now()
        
        # Check if we have a cached result that's still valid
        async with self._cache_lock:
            if (cache_key in self._cache and 
                cache_key in self._cache_expiry and 
                current_time < self._cache_expiry[cache_key]):
                return self._cache[cache_key]
        
        # Not in cache or expired, fetch new result
        result = await fetch_func()
        
        # Cache the new result
        async with self._cache_lock:
            self._cache[cache_key] = result
            self._cache_expiry[cache_key] = current_time + timedelta(seconds=expiry_seconds)
        
        return result
    
    # User history methods
    async def get_history(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user conversation history with caching"""
        cache_key = f"history_{user_id}"
        
        async def fetch_history():
            user_data = await self.db.user_histories.find_one({'user_id': user_id})
            if user_data and 'history' in user_data:
                return user_data['history']
            return []
            
        return await self._get_cached_result(cache_key, fetch_history, 30)  # 30 second cache
    
    async def save_history(self, user_id: int, history: List[Dict[str, Any]]) -> None:
        """Save user conversation history and update cache"""
        await self.db.user_histories.update_one(
            {'user_id': user_id},
            {'$set': {'history': history}},
            upsert=True
        )
        
        # Update the cache
        cache_key = f"history_{user_id}"
        async with self._cache_lock:
            self._cache[cache_key] = history
            self._cache_expiry[cache_key] = datetime.now() + timedelta(seconds=30)
    
    # User model preferences with caching
    async def get_user_model(self, user_id: int) -> Optional[str]:
        """Get user's preferred model with caching"""
        cache_key = f"model_{user_id}"
        
        async def fetch_model():
            user_data = await self.db.user_models.find_one({'user_id': user_id})
            return user_data['model'] if user_data else None
            
        return await self._get_cached_result(cache_key, fetch_model, 300)  # 5 minute cache
    
    async def save_user_model(self, user_id: int, model: str) -> None:
        """Save user's preferred model and update cache"""
        await self.db.user_models.update_one(
            {'user_id': user_id},
            {'$set': {'model': model}},
            upsert=True
        )
        
        # Update the cache
        cache_key = f"model_{user_id}"
        async with self._cache_lock:
            self._cache[cache_key] = model
            self._cache_expiry[cache_key] = datetime.now() + timedelta(seconds=300)
    
    # Admin and permissions management with caching
    async def is_admin(self, user_id: int) -> bool:
        """Check if the user is an admin (no caching for security)"""
        admin_id = str(user_id)  # Convert to string for comparison
        from src.config.config import ADMIN_ID
        return admin_id == ADMIN_ID
    
    async def is_user_whitelisted(self, user_id: int) -> bool:
        """Check if the user is whitelisted with caching"""
        if await self.is_admin(user_id):
            return True
            
        cache_key = f"whitelist_{user_id}"
        
        async def check_whitelist():
            user_data = await self.db.whitelist.find_one({'user_id': user_id})
            return user_data is not None
            
        return await self._get_cached_result(cache_key, check_whitelist, 300)  # 5 minute cache
    
    async def add_user_to_whitelist(self, user_id: int) -> None:
        """Add user to whitelist and update cache"""
        await self.db.whitelist.update_one(
            {'user_id': user_id},
            {'$set': {'user_id': user_id}},
            upsert=True
        )
        
        # Update the cache
        cache_key = f"whitelist_{user_id}"
        async with self._cache_lock:
            self._cache[cache_key] = True
            self._cache_expiry[cache_key] = datetime.now() + timedelta(seconds=300)
    
    async def remove_user_from_whitelist(self, user_id: int) -> bool:
        """Remove user from whitelist and update cache"""
        result = await self.db.whitelist.delete_one({'user_id': user_id})
        
        # Update the cache
        cache_key = f"whitelist_{user_id}"
        async with self._cache_lock:
            self._cache[cache_key] = False
            self._cache_expiry[cache_key] = datetime.now() + timedelta(seconds=300)
            
        return result.deleted_count > 0
    
    async def is_user_blacklisted(self, user_id: int) -> bool:
        """Check if the user is blacklisted with caching"""
        cache_key = f"blacklist_{user_id}"
        
        async def check_blacklist():
            user_data = await self.db.blacklist.find_one({'user_id': user_id})
            return user_data is not None
            
        return await self._get_cached_result(cache_key, check_blacklist, 300)  # 5 minute cache
    
    async def add_user_to_blacklist(self, user_id: int) -> None:
        """Add user to blacklist and update cache"""
        await self.db.blacklist.update_one(
            {'user_id': user_id},
            {'$set': {'user_id': user_id}},
            upsert=True
        )
        
        # Update the cache
        cache_key = f"blacklist_{user_id}"
        async with self._cache_lock:
            self._cache[cache_key] = True
            self._cache_expiry[cache_key] = datetime.now() + timedelta(seconds=300)
    
    async def remove_user_from_blacklist(self, user_id: int) -> bool:
        """Remove user from blacklist and update cache"""
        result = await self.db.blacklist.delete_one({'user_id': user_id})
        
        # Update the cache
        cache_key = f"blacklist_{user_id}"
        async with self._cache_lock:
            self._cache[cache_key] = False
            self._cache_expiry[cache_key] = datetime.now() + timedelta(seconds=300)
            
        return result.deleted_count > 0
            
    # Connection management and cleanup
    async def create_indexes(self):
        """Create indexes for better query performance"""
        await self.db.user_histories.create_index("user_id")
        await self.db.user_models.create_index("user_id") 
        await self.db.whitelist.create_index("user_id")
        await self.db.blacklist.create_index("user_id")
        
    async def close(self):
        """Properly close the database connection"""
        self.client.close()