import asyncio
import logging
import discord
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, List, Optional, Union

class ReminderManager:
    """
    Manages reminder functionality for Discord users
    """
    def __init__(self, bot, db_handler):
        """
        Initialize ReminderManager
        
        Args:
            bot: Discord bot instance
            db_handler: Database handler instance
        """
        self.bot = bot
        self.db = db_handler
        self.running = False
        self.check_task = None
        
        # Get system timezone to ensure consistency
        self.timezone = datetime.now().astimezone().tzinfo
        logging.info(f"Using server timezone: {self.timezone}")
    
    def start(self):
        """Start periodic reminder check"""
        if not self.running:
            self.running = True
            self.check_task = asyncio.create_task(self._check_reminders_loop())
            logging.info("Reminder manager started")
    
    async def stop(self):
        """Stop the reminder check"""
        if self.running:
            self.running = False
            if self.check_task:
                self.check_task.cancel()
                try:
                    await self.check_task
                except asyncio.CancelledError:
                    pass
                self.check_task = None
            logging.info("Reminder manager stopped")
    
    def get_current_time(self) -> datetime:
        """
        Get the current time with proper timezone
        
        Returns:
            Current datetime with timezone
        """
        return datetime.now().replace(tzinfo=self.timezone)
    
    async def add_reminder(self, user_id: int, content: str, remind_at: datetime) -> Dict[str, Any]:
        """
        Add a new reminder
        
        Args:
            user_id: Discord user ID
            content: Reminder content
            remind_at: When to send the reminder
            
        Returns:
            Information about the added reminder
        """
        try:
            now = self.get_current_time()
            
            reminder = {
                "user_id": user_id,
                "content": content,
                "remind_at": remind_at,
                "created_at": now,
                "sent": False
            }
            
            result = await self.db.reminders_collection.insert_one(reminder)
            reminder["_id"] = result.inserted_id
            
            logging.info(f"Added reminder for user {user_id} at {remind_at} (Server timezone: {self.timezone})")
            return reminder
        except Exception as e:
            logging.error(f"Error adding reminder: {str(e)}")
            raise
    
    async def get_user_reminders(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get a user's reminders
        
        Args:
            user_id: Discord user ID
            
        Returns:
            List of reminders
        """
        try:
            cursor = self.db.reminders_collection.find({
                "user_id": user_id,
                "sent": False
            }).sort("remind_at", 1)
            
            return await cursor.to_list(length=100)
        except Exception as e:
            logging.error(f"Error getting reminders for user {user_id}: {str(e)}")
            return []
    
    async def delete_reminder(self, reminder_id, user_id: int) -> bool:
        """
        Delete a reminder
        
        Args:
            reminder_id: Reminder ID
            user_id: Discord user ID (to verify ownership)
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            from bson.objectid import ObjectId
            
            # Convert reminder_id to ObjectId if needed
            if isinstance(reminder_id, str):
                reminder_id = ObjectId(reminder_id)
                
            result = await self.db.reminders_collection.delete_one({
                "_id": reminder_id,
                "user_id": user_id
            })
            
            return result.deleted_count > 0
        except Exception as e:
            logging.error(f"Error deleting reminder {reminder_id}: {str(e)}")
            return False
    
    async def _check_reminders_loop(self):
        """Loop to check for due reminders"""
        try:
            while self.running:
                try:
                    await self._process_due_reminders()
                    await self._clean_expired_reminders()
                except Exception as e:
                    logging.error(f"Error in reminder check: {str(e)}")
                    
                # Wait 30 seconds before checking again
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            # Handle task cancellation
            logging.info("Reminder check loop was cancelled")
            raise
    
    async def _process_due_reminders(self):
        """Process due reminders and send notifications"""
        now = self.get_current_time()
        
        # Find due reminders
        cursor = self.db.reminders_collection.find({
            "remind_at": {"$lte": now},
            "sent": False
        })
        
        due_reminders = await cursor.to_list(length=100)
        
        for reminder in due_reminders:
            try:
                # Get user information
                user_id = reminder["user_id"]
                user = await self.bot.fetch_user(user_id)
                
                if user:
                    # Format reminder message
                    embed = discord.Embed(
                        title="📅 Reminder",
                        description=reminder["content"],
                        color=discord.Color.blue()
                    )
                    embed.add_field(
                        name="Set on",
                        value=reminder["created_at"].strftime("%Y-%m-%d %H:%M")
                    )
                    embed.set_footer(text="Server time: " + now.strftime("%Y-%m-%d %H:%M"))
                    
                    # Send reminder message with mention
                    try:
                        # Try to send a direct message first
                        await user.send(f"<@{user_id}> Here's your reminder:", embed=embed)
                        logging.info(f"Sent reminder DM to user {user_id}")
                    except Exception as dm_error:
                        logging.error(f"Could not send DM to user {user_id}: {str(dm_error)}")
                        # Could implement fallback method here if needed
                
                # Mark reminder as sent and delete it
                await self.db.reminders_collection.delete_one({"_id": reminder["_id"]})
                logging.info(f"Deleted completed reminder {reminder['_id']} for user {user_id}")
                
            except Exception as e:
                logging.error(f"Error processing reminder {reminder['_id']}: {str(e)}")
    
    async def _clean_expired_reminders(self):
        """Clean up old reminders that were marked as sent but not deleted"""
        try:
            result = await self.db.reminders_collection.delete_many({
                "sent": True
            })
            
            if result.deleted_count > 0:
                logging.info(f"Cleaned up {result.deleted_count} expired reminders")
        except Exception as e:
            logging.error(f"Error cleaning expired reminders: {str(e)}")
    
    async def parse_time(self, time_str: str) -> Optional[datetime]:
        """
        Parse a time string into a datetime object with the server's timezone
        
        Args:
            time_str: Time string (e.g., "30m", "2h", "1d", "tomorrow", "15:00")
            
        Returns:
            Datetime object or None if parsing fails
        """
        now = self.get_current_time()
        time_str = time_str.lower().strip()
        
        try:
            # Handle special keywords
            if time_str == "tomorrow":
                return now.replace(hour=9, minute=0, second=0) + timedelta(days=1)
            elif time_str == "tonight":
                # Use 8 PM (20:00) for "tonight"
                target = now.replace(hour=20, minute=0, second=0)
                # If it's already past 8 PM, schedule for tomorrow night
                if target <= now:
                    target += timedelta(days=1)
                return target
            elif time_str == "noon":
                # Use 12 PM for "noon"
                target = now.replace(hour=12, minute=0, second=0)
                # If it's already past noon, schedule for tomorrow
                if target <= now:
                    target += timedelta(days=1)
                return target
                
            # Handle relative time formats (30m, 2h, 1d)
            if time_str[-1] in ['m', 'h', 'd']:
                value = int(time_str[:-1])
                unit = time_str[-1]
                
                if unit == 'm':  # minutes
                    return now + timedelta(minutes=value)
                elif unit == 'h':  # hours
                    return now + timedelta(hours=value)
                elif unit == 'd':  # days
                    return now + timedelta(days=value)
            
            # Handle specific time format
            # HH:MM format for today or tomorrow
            if ':' in time_str and len(time_str.split(':')) == 2:
                hour, minute = map(int, time_str.split(':'))
                
                # Check if valid time
                if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                    logging.warning(f"Invalid time format: {time_str}")
                    return None
                    
                # Create datetime for the specified time today
                target = now.replace(hour=hour, minute=minute, second=0)
                
                # If the time has already passed today, schedule for tomorrow
                if target <= now:
                    target += timedelta(days=1)
                    
                logging.info(f"Parsed time '{time_str}' to {target} (Server timezone: {self.timezone})")
                return target
                
            return None
        except Exception as e:
            logging.error(f"Error parsing time string '{time_str}': {str(e)}")
            return None