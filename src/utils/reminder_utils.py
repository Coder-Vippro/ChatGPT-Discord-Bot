import asyncio
import logging
import discord
from datetime import datetime, timedelta
import pytz
import time
from typing import Dict, Any, List, Optional, Union
from src.config.config import TIMEZONE  # Import the TIMEZONE from config

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
        
        # Use the timezone from .env file through config
        try:
            self.server_timezone = pytz.timezone(TIMEZONE)
        except pytz.exceptions.UnknownTimeZoneError:
            logging.warning(f"Invalid timezone '{TIMEZONE}' in .env, using UTC instead")
            self.server_timezone = pytz.timezone("UTC")
        
        # Store user timezones (will be populated as users interact)
        self.user_timezones = {}
        
        # Log initial timezone info
        logging.info(f"ReminderManager initialized, server timezone: {self.server_timezone}")
    
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
        Get the current time with proper timezone from the real machine
        
        Returns:
            Current datetime with timezone
        """
        # Always get the current time with the server's timezone
        return datetime.now(self.server_timezone)
    
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
            
            # Ensure remind_at has timezone info
            if remind_at.tzinfo is None:
                # Apply server timezone if no timezone is provided
                remind_at = remind_at.replace(tzinfo=self.server_timezone)
            
            reminder = {
                "user_id": user_id,
                "content": content,
                "remind_at": remind_at,
                "created_at": now,
                "sent": False,
                "user_timezone": self.user_timezones.get(user_id, str(self.server_timezone))
            }
            
            result = await self.db.reminders_collection.insert_one(reminder)
            reminder["_id"] = result.inserted_id
            
            logging.info(f"Added reminder for user {user_id} at {remind_at} (System timezone: {now.tzinfo})")
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
        
        # Find due reminders - convert now to UTC for MongoDB comparison
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
                    # Format reminder message with user's timezone if available
                    user_timezone = reminder.get("user_timezone", str(self.server_timezone))
                    try:
                        tz = pytz.timezone(user_timezone) if isinstance(user_timezone, str) else user_timezone
                    except (pytz.exceptions.UnknownTimeZoneError, TypeError):
                        tz = self.server_timezone
                    
                    # Format datetime in user's preferred timezone
                    reminder_time = reminder["remind_at"]
                    if reminder_time.tzinfo is not None:
                        user_time = reminder_time.astimezone(tz)
                    else:
                        user_time = reminder_time.replace(tzinfo=self.server_timezone).astimezone(tz)
                    
                    current_time = now.astimezone(tz)
                    
                    embed = discord.Embed(
                        title="📅 Reminder",
                        description=reminder["content"],
                        color=discord.Color.blue()
                    )
                    embed.add_field(
                        name="Set on",
                        value=reminder["created_at"].astimezone(tz).strftime("%Y-%m-%d %H:%M")
                    )
                    embed.add_field(
                        name="Your timezone",
                        value=str(tz)
                    )
                    embed.set_footer(text="Current time: " + current_time.strftime("%Y-%m-%d %H:%M"))
                    
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
    
    async def set_user_timezone(self, user_id: int, timezone_str: str) -> bool:
        """
        Set a user's timezone preference
        
        Args:
            user_id: Discord user ID
            timezone_str: Timezone string (e.g. "America/New_York", "Europe/London")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate timezone string
            try:
                tz = pytz.timezone(timezone_str)
                self.user_timezones[user_id] = timezone_str
                logging.info(f"Set timezone for user {user_id} to {timezone_str}")
                return True
            except pytz.exceptions.UnknownTimeZoneError:
                logging.warning(f"Invalid timezone: {timezone_str}")
                return False
        except Exception as e:
            logging.error(f"Error setting user timezone: {str(e)}")
            return False
    
    async def detect_user_timezone(self, user_id: int, guild_id: Optional[int] = None) -> str:
        """
        Try to detect a user's timezone 
        
        Args:
            user_id: Discord user ID
            guild_id: Optional guild ID to check location
            
        Returns:
            Timezone string
        """
        # First check if we already have the user's timezone
        if user_id in self.user_timezones:
            return self.user_timezones[user_id]
        
        # Default to server timezone
        return str(self.server_timezone)
    
    async def parse_time(self, time_str: str, user_id: Optional[int] = None) -> Optional[datetime]:
        """
        Parse a time string into a datetime object with timezone awareness
        
        Args:
            time_str: Time string (e.g., "30m", "2h", "1d", "tomorrow", "15:00")
            user_id: Optional user ID to use their preferred timezone
            
        Returns:
            Datetime object or None if parsing fails
        """
        # Get appropriate timezone
        if user_id and user_id in self.user_timezones:
            try:
                user_tz = pytz.timezone(self.user_timezones[user_id])
            except pytz.exceptions.UnknownTimeZoneError:
                user_tz = self.server_timezone
        else:
            user_tz = self.server_timezone
            
        # Get current time in user's timezone
        now = datetime.now(user_tz)
        time_str = time_str.lower().strip()
        
        try:
            # Handle special keywords
            if time_str == "tomorrow":
                return now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
            elif time_str == "tonight":
                # Use 8 PM (20:00) for "tonight"
                target = now.replace(hour=20, minute=0, second=0, microsecond=0)
                # If it's already past 8 PM, schedule for tomorrow night
                if target <= now:
                    target += timedelta(days=1)
                return target
            elif time_str == "noon":
                # Use 12 PM for "noon"
                target = now.replace(hour=12, minute=0, second=0, microsecond=0)
                # If it's already past noon, schedule for tomorrow
                if target <= now:
                    target += timedelta(days=1)
                return target
                
            # Handle relative time formats (30m, 2h, 1d)
            if len(time_str) >= 2 and time_str[-1] in ['m', 'h', 'd'] and time_str[:-1].isdigit():
                value = int(time_str[:-1])
                unit = time_str[-1]
                
                if unit == 'm':  # minutes
                    return now + timedelta(minutes=value)
                elif unit == 'h':  # hours
                    return now + timedelta(hours=value)
                elif unit == 'd':  # days
                    return now + timedelta(days=value)
              # Handle specific time format
            # Support various time formats: HH:MM, H:MM, H:MM AM/PM, HH:MM AM/PM
            if ':' in time_str:
                # Extract time part and additional words
                time_parts = time_str.split()
                time_part = time_parts[0]  # e.g., "9:00"
                
                # Check for AM/PM
                is_pm = False
                for part in time_parts[1:]:
                    if 'pm' in part.lower():
                        is_pm = True
                        break
                    elif 'am' in part.lower():
                        is_pm = False
                        break
                
                try:
                    if ':' in time_part and len(time_part.split(':')) == 2:
                        hour_str, minute_str = time_part.split(':')
                        
                        # Clean minute string to remove non-digit characters
                        minute_str = ''.join(filter(str.isdigit, minute_str))
                        if not minute_str:
                            minute_str = '0'
                            
                        hour = int(hour_str)
                        minute = int(minute_str)
                        
                        # Handle AM/PM conversion
                        if is_pm and hour != 12:
                            hour += 12
                        elif not is_pm and hour == 12:
                            hour = 0
                        
                        # Check if valid time
                        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                            logging.warning(f"Invalid time format: {time_str}")
                            return None
                            
                        # Create datetime for the specified time today in user's timezone
                        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                        
                        # Check for "tomorrow" keyword
                        if 'tomorrow' in time_str.lower():
                            target += timedelta(days=1)
                        # If the time has already passed today and no "today" keyword, schedule for tomorrow
                        elif target <= now and 'today' not in time_str.lower():
                            target += timedelta(days=1)
                            
                        logging.info(f"Parsed time '{time_str}' to {target} (User timezone: {user_tz})")
                        return target
                        
                except ValueError as ve:
                    logging.error(f"Error parsing time components in '{time_str}': {str(ve)}")
                    return None
                
            return None
        except Exception as e:
            logging.error(f"Error parsing time string '{time_str}': {str(e)}")
            return None