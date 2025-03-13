from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict, Any, Optional

class DatabaseHandler:
    def __init__(self, mongodb_uri: str):
        """Initialize database connection"""
        self.client = AsyncIOMotorClient(mongodb_uri)
        self.db = self.client['chatgpt_discord_bot']  # Database name
    
    # User history methods
    async def get_history(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user conversation history"""
        user_data = await self.db.user_histories.find_one({'user_id': user_id})
        if user_data and 'history' in user_data:
            return user_data['history']
        return []
    
    async def save_history(self, user_id: int, history: List[Dict[str, Any]]) -> None:
        """Save user conversation history"""
        await self.db.user_histories.update_one(
            {'user_id': user_id},
            {'$set': {'history': history}},
            upsert=True
        )
    
    # User model preferences
    async def get_user_model(self, user_id: int) -> Optional[str]:
        """Get user's preferred model"""
        user_data = await self.db.user_models.find_one({'user_id': user_id})
        return user_data['model'] if user_data else None
    
    async def save_user_model(self, user_id: int, model: str) -> None:
        """Save user's preferred model"""
        await self.db.user_models.update_one(
            {'user_id': user_id},
            {'$set': {'model': model}},
            upsert=True
        )
    
    # Admin and permissions management
    async def is_admin(self, user_id: int) -> bool:
        """Check if the user is an admin"""
        admin_id = str(user_id)  # Convert to string for comparison
        from src.config.config import ADMIN_ID
        return admin_id == ADMIN_ID
    
    async def is_user_whitelisted(self, user_id: int) -> bool:
        """Check if the user is whitelisted"""
        if await self.is_admin(user_id):
            return True
        user_data = await self.db.whitelist.find_one({'user_id': user_id})
        return user_data is not None
    
    async def add_user_to_whitelist(self, user_id: int) -> None:
        """Add user to whitelist"""
        await self.db.whitelist.update_one(
            {'user_id': user_id},
            {'$set': {'user_id': user_id}},
            upsert=True
        )
    
    async def remove_user_from_whitelist(self, user_id: int) -> bool:
        """Remove user from whitelist"""
        result = await self.db.whitelist.delete_one({'user_id': user_id})
        return result.deleted_count > 0
    
    async def is_user_blacklisted(self, user_id: int) -> bool:
        """Check if the user is blacklisted"""
        user_data = await self.db.blacklist.find_one({'user_id': user_id})
        return user_data is not None
    
    async def add_user_to_blacklist(self, user_id: int) -> None:
        """Add user to blacklist"""
        await self.db.blacklist.update_one(
            {'user_id': user_id},
            {'$set': {'user_id': user_id}},
            upsert=True
        )
    
    async def remove_user_from_blacklist(self, user_id: int) -> bool:
        """Remove user from blacklist"""
        result = await self.db.blacklist.delete_one({'user_id': user_id})
        return result.deleted_count > 0