from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict, Any, Optional
import functools
import asyncio
from datetime import datetime, timedelta
import logging
import re

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
            maxIdleTimeMS=45000,
            connectTimeoutMS=10000,
            serverSelectionTimeoutMS=15000,
            waitQueueTimeoutMS=5000,
            socketTimeoutMS=30000,
            retryWrites=True
        )
        self.db = self.client['chatgpt_discord_bot']  # Database name
        
        # Collections
        self.users_collection = self.db.users
        self.history_collection = self.db.history
        self.admin_collection = self.db.admin
        self.blacklist_collection = self.db.blacklist
        self.whitelist_collection = self.db.whitelist
        self.logs_collection = self.db.logs
        self.reminders_collection = self.db.reminders
        
        logging.info("Database handler initialized")
    
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
        """Get user conversation history with caching and filter expired image links"""
        cache_key = f"history_{user_id}"
        
        async def fetch_history():
            user_data = await self.db.user_histories.find_one({'user_id': user_id})
            if user_data and 'history' in user_data:
                # Filter out expired image links
                filtered_history = self._filter_expired_images(user_data['history'])
                return filtered_history
            return []
            
        return await self._get_cached_result(cache_key, fetch_history, 30)  # 30 second cache
    
    def _filter_expired_images(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out image links that are older than 23 hours"""
        current_time = datetime.now()
        expiration_time = current_time - timedelta(hours=23)
        
        filtered_history = []
        for msg in history:
            # Keep system messages unchanged
            if msg.get('role') == 'system':
                filtered_history.append(msg)
                continue
                
            # Check if message has 'content' field as a list (which may contain image URLs)
            content = msg.get('content')
            if isinstance(content, list):
                # Filter content items
                filtered_content = []
                for item in content:
                    # Keep text items
                    if item.get('type') == 'text':
                        filtered_content.append(item)
                    # Check image items for timestamp
                    elif item.get('type') == 'image_url':
                        # If there's no timestamp or timestamp is newer than expiration time, keep it
                        timestamp = item.get('timestamp')
                        if not timestamp or datetime.fromisoformat(timestamp) > expiration_time:
                            filtered_content.append(item)
                        else:
                            logging.info(f"Filtering out expired image URL (added at {timestamp})")
                
                # Update the message with filtered content
                if filtered_content:
                    new_msg = dict(msg)
                    new_msg['content'] = filtered_content
                    filtered_history.append(new_msg)
                else:
                    # If after filtering there's no content, add a placeholder text
                    new_msg = dict(msg)
                    new_msg['content'] = [{"type": "text", "text": "[Image content expired]"}]
                    filtered_history.append(new_msg)
            else:
                # For string content or other formats, keep as is
                filtered_history.append(msg)
                
        return filtered_history
    
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
        
    async def ensure_reminders_collection(self):
        """
        Ensure the reminders collection exists and create necessary indexes
        """
        # Create the collection if it doesn't exist
        await self.reminders_collection.create_index([("user_id", 1), ("sent", 1)])
        await self.reminders_collection.create_index([("remind_at", 1), ("sent", 1)])
        logging.info("Ensured reminders collection and indexes")

    async def close(self):
        """Properly close the database connection"""
        self.client.close()
        logging.info("Database connection closed")