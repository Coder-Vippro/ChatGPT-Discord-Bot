from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict, Any, Optional
import functools
import asyncio
from datetime import datetime, timedelta
import logging
import re
import os

# Configure DNS resolver to be more resilient
try:
    import dns.resolver
    dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
    dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4', '1.1.1.1']
    dns.resolver.default_resolver.lifetime = 15.0  # 15 second timeout for DNS
except ImportError:
    logging.warning("dnspython not installed, using system DNS resolver")
except Exception as e:
    logging.warning(f"Could not configure custom DNS resolver: {e}")

class DatabaseHandler:
    def __init__(self, mongodb_uri: str, max_retries: int = 5):
        """Initialize database connection with optimized settings and retry logic"""
        self.mongodb_uri = mongodb_uri
        self.max_retries = max_retries
        self._connected = False
        self._connection_lock = asyncio.Lock()
        
        # Set up a memory-optimized connection pool with better resilience
        self.client = AsyncIOMotorClient(
            mongodb_uri,
            maxIdleTimeMS=45000,           # Keep connections alive longer
            connectTimeoutMS=20000,        # 20s connect timeout for DNS issues
            serverSelectionTimeoutMS=30000, # 30s for server selection
            waitQueueTimeoutMS=10000,      # Wait longer for available connection
            socketTimeoutMS=45000,         # Socket operations timeout
            maxPoolSize=10,                # Slightly larger pool
            minPoolSize=1,                 # Keep at least 1 connection
            retryWrites=True,
            retryReads=True,               # Also retry reads
            directConnection=False,        # Allow replica set discovery
            appName="ChatGPT-Discord-Bot",
            heartbeatFrequencyMS=30000,    # Reduce heartbeat frequency to avoid DNS issues
            localThresholdMS=30,           # Local threshold for selecting servers
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
        
        logging.info("Database handler initialized with enhanced connection resilience")
    
    async def _retry_operation(self, operation, *args, **kwargs):
        """Execute a database operation with retry logic for transient errors"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Check for transient/retryable errors (expanded list)
                retryable_errors = [
                    'timeout', 'connection', 'socket', 'dns', 'try again', 
                    'network', 'errno -3', 'gaierror', 'nodename', 'servname',
                    'temporary failure', 'name resolution', 'unreachable',
                    'reset by peer', 'broken pipe', 'not connected'
                ]
                if any(err in error_str for err in retryable_errors):
                    wait_time = min((attempt + 1) * 2, 10)  # Exponential backoff: 2s, 4s, 6s, 8s, 10s (max)
                    logging.warning(f"Database operation failed (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    # Non-retryable error, raise immediately
                    raise
        # All retries exhausted
        logging.error(f"Database operation failed after {self.max_retries} attempts: {last_error}")
        raise last_error
    
    async def ensure_connected(self) -> bool:
        """Ensure database connection is established with retry logic"""
        async with self._connection_lock:
            if self._connected:
                return True
            
            for attempt in range(self.max_retries):
                try:
                    await self.client.admin.command('ping')
                    self._connected = True
                    logging.info("Database connection established successfully")
                    return True
                except Exception as e:
                    wait_time = min((attempt + 1) * 2, 10)
                    logging.warning(f"Database connection attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
            
            logging.error("Failed to establish database connection after all retries")
            return False
    
    async def check_connection(self) -> bool:
        """Check if database connection is alive with graceful error handling"""
        try:
            # Use a short timeout for the ping operation
            await asyncio.wait_for(
                self.client.admin.command('ping'),
                timeout=10.0
            )
            self._connected = True
            return True
        except asyncio.TimeoutError:
            logging.warning("Database ping timed out")
            self._connected = False
            return False
        except Exception as e:
            error_str = str(e).lower()
            # Don't log DNS resolution failures as errors (they're often transient)
            if any(err in error_str for err in ['errno -3', 'try again', 'dns', 'gaierror']):
                logging.debug(f"Transient database connection check failed (DNS): {e}")
            else:
                logging.error(f"Database connection check failed: {e}")
            self._connected = False
            return False
    
    # User history methods
    async def get_history(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user conversation history and filter expired image links"""
        async def _get():
            return await self.db.user_histories.find_one({'user_id': user_id})
        
        user_data = await self._retry_operation(_get)
        if user_data and 'history' in user_data:
            # Filter out expired image links
            filtered_history = self._filter_expired_images(user_data['history'])
            
            # Proactive history trimming: Keep only the last 50 messages to prevent excessive token usage
            # Always preserve system messages
            system_messages = [msg for msg in filtered_history if msg.get('role') == 'system']
            conversation_messages = [msg for msg in filtered_history if msg.get('role') != 'system']
            
            # Keep only the last 50 conversation messages
            if len(conversation_messages) > 50:
                conversation_messages = conversation_messages[-50:]
                logging.info(f"Trimmed history for user {user_id}: kept last 50 conversation messages")
            
            # Combine system messages with trimmed conversation
            trimmed_history = system_messages + conversation_messages
            
            # If history was trimmed, save the trimmed version back to DB
            if len(trimmed_history) < len(filtered_history):
                await self.save_history(user_id, trimmed_history)
                logging.info(f"Saved trimmed history for user {user_id}: {len(trimmed_history)} messages")
            
            return trimmed_history
        return []
    
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
        """Save user conversation history"""
        await self.db.user_histories.update_one(
            {'user_id': user_id},
            {'$set': {'history': history}},
            upsert=True
        )
    
    # User model preferences with caching
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
    
    # Tool display preferences
    async def get_user_tool_display(self, user_id: int) -> bool:
        """Get user's tool display preference (default: False - disabled)"""
        user_data = await self.db.user_preferences.find_one({'user_id': user_id})
        return user_data.get('show_tool_execution', False) if user_data else False
    
    async def set_user_tool_display(self, user_id: int, show_tools: bool) -> None:
        """Set user's tool display preference"""
        await self.db.user_preferences.update_one(
            {'user_id': user_id},
            {'$set': {'show_tool_execution': show_tools}},
            upsert=True
        )
    
    # Admin and permissions management with caching
    async def is_admin(self, user_id: int) -> bool:
        """Check if the user is an admin (no caching for security)"""
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
            
    # Connection management and cleanup
    async def create_indexes(self):
        """Create indexes for better query performance"""
        await self.db.user_histories.create_index("user_id")
        await self.db.user_models.create_index("user_id") 
        await self.db.user_preferences.create_index("user_id")
        await self.db.whitelist.create_index("user_id")
        await self.db.blacklist.create_index("user_id")
        await self.db.token_usage.create_index([("user_id", 1), ("timestamp", -1)])
        await self.db.user_token_stats.create_index("user_id")
        
        # User files indexes for code interpreter (48-hour expiration)
        await self.db.user_files.create_index([("user_id", 1), ("expires_at", -1)])
        await self.db.user_files.create_index("file_id", unique=True)
        await self.db.user_files.create_index("expires_at")  # For cleanup queries
        
    async def ensure_reminders_collection(self):
        """
        Ensure the reminders collection exists and create necessary indexes
        """
        # Create the collection if it doesn't exist
        await self.reminders_collection.create_index([("user_id", 1), ("sent", 1)])
        await self.reminders_collection.create_index([("remind_at", 1), ("sent", 1)])
        logging.info("Ensured reminders collection and indexes")

    # Token usage tracking methods
    async def save_token_usage(
        self, 
        user_id: int, 
        model: str, 
        input_tokens: int, 
        output_tokens: int, 
        cost: float,
        text_tokens: int = 0,
        image_tokens: int = 0
    ):
        """Save token usage and cost for a user with detailed breakdown"""
        try:
            usage_data = {
                "user_id": user_id,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "text_tokens": text_tokens,
                "image_tokens": image_tokens,
                "cost": cost,
                "timestamp": datetime.now()
            }
            
            # Insert usage record
            await self.db.token_usage.insert_one(usage_data)
            
            # Escape model name for MongoDB field names (replace dots and other special chars)
            escaped_model = model.replace(".", "_DOT_").replace("/", "_SLASH_").replace("$", "_DOLLAR_")
            
            # Update user's total usage
            await self.db.user_token_stats.update_one(
                {"user_id": user_id},
                {
                    "$inc": {
                        "total_input_tokens": input_tokens,
                        "total_output_tokens": output_tokens,
                        "total_text_tokens": text_tokens,
                        "total_image_tokens": image_tokens,
                        "total_cost": cost,
                        f"models.{escaped_model}.input_tokens": input_tokens,
                        f"models.{escaped_model}.output_tokens": output_tokens,
                        f"models.{escaped_model}.text_tokens": text_tokens,
                        f"models.{escaped_model}.image_tokens": image_tokens,
                        f"models.{escaped_model}.cost": cost,
                        f"models.{escaped_model}.requests": 1
                    },
                    "$set": {"last_updated": datetime.now()}
                },
                upsert=True
            )
            
        except Exception as e:
            logging.error(f"Error saving token usage: {e}")

    async def get_user_token_usage(self, user_id: int) -> Dict[str, Any]:
        """Get total token usage for a user with detailed breakdown"""
        try:
            user_stats = await self.db.user_token_stats.find_one({"user_id": user_id})
            if user_stats:
                return {
                    "total_input_tokens": user_stats.get("total_input_tokens", 0),
                    "total_output_tokens": user_stats.get("total_output_tokens", 0),
                    "total_text_tokens": user_stats.get("total_text_tokens", 0),
                    "total_image_tokens": user_stats.get("total_image_tokens", 0),
                    "total_cost": user_stats.get("total_cost", 0.0)
                }
            return {
                "total_input_tokens": 0, 
                "total_output_tokens": 0, 
                "total_text_tokens": 0,
                "total_image_tokens": 0,
                "total_cost": 0.0
            }
        except Exception as e:
            logging.error(f"Error getting user token usage: {e}")
            return {
                "total_input_tokens": 0, 
                "total_output_tokens": 0,
                "total_text_tokens": 0,
                "total_image_tokens": 0,
                "total_cost": 0.0
            }

    async def get_user_token_usage_by_model(self, user_id: int) -> Dict[str, Dict[str, Any]]:
        """Get token usage breakdown by model for a user with text/image details"""
        try:
            user_stats = await self.db.user_token_stats.find_one({"user_id": user_id})
            if user_stats and "models" in user_stats:
                # Unescape model names for display
                unescaped_models = {}
                for escaped_model, usage in user_stats["models"].items():
                    # Reverse the escaping
                    original_model = escaped_model.replace("_DOT_", ".").replace("_SLASH_", "/").replace("_DOLLAR_", "$")
                    unescaped_models[original_model] = {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "text_tokens": usage.get("text_tokens", 0),
                        "image_tokens": usage.get("image_tokens", 0),
                        "cost": usage.get("cost", 0.0),
                        "requests": usage.get("requests", 0)
                    }
                return unescaped_models
            return {}
        except Exception as e:
            logging.error(f"Error getting user token usage by model: {e}")
            return {}

    async def reset_user_token_stats(self, user_id: int) -> None:
        """Reset all token usage statistics for a user"""
        try:
            # Delete the user's token stats document
            await self.db.user_token_stats.delete_one({"user_id": user_id})
            
            # Optionally, also delete individual usage records
            await self.db.token_usage.delete_many({"user_id": user_id})
            
            logging.info(f"Reset token statistics for user {user_id}")
        except Exception as e:
            logging.error(f"Error resetting user token stats: {e}")

    # User files management methods for code interpreter
    async def get_user_files(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all files for a specific user"""
        try:
            current_time = datetime.now().isoformat()  # Use ISO string for comparison
            files = await self.db.user_files.find({
                "user_id": user_id,
                "$or": [
                    {"expires_at": {"$gt": current_time}},  # Not expired
                    {"expires_at": None}  # Never expires
                ]
            }).to_list(length=1000)
            logging.info(f"[DEBUG] Query returned {len(files)} files for user {user_id}")
            return files
        except Exception as e:
            logging.error(f"Error getting user files: {e}")
            return []
    
    async def save_user_file(self, file_data: Dict[str, Any]) -> None:
        """Save or update a user file record"""
        try:
            await self.db.user_files.update_one(
                {"file_id": file_data["file_id"]},
                {"$set": file_data},
                upsert=True
            )
        except Exception as e:
            logging.error(f"Error saving user file: {e}")
    
    async def delete_user_file(self, file_id: str) -> bool:
        """Delete a specific user file record"""
        try:
            result = await self.db.user_files.delete_one({"file_id": file_id})
            return result.deleted_count > 0
        except Exception as e:
            logging.error(f"Error deleting user file: {e}")
            return False
    
    async def delete_expired_files(self) -> int:
        """Delete all expired file records (called by cleanup task)"""
        try:
            current_time = datetime.now()
            result = await self.db.user_files.delete_many({
                "expires_at": {"$lt": current_time, "$ne": None}
            })
            return result.deleted_count
        except Exception as e:
            logging.error(f"Error deleting expired files: {e}")
            return 0

    async def close(self):
        """Properly close the database connection"""
        self.client.close()
        logging.info("Database connection closed")