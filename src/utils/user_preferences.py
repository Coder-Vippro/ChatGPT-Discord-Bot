"""
User Preferences System
Manages user-specific preferences and settings for enhanced personalization.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

class UserPreferences:
    """Manages user preferences and settings."""
    
    def __init__(self, db_handler):
        self.db = db_handler
        self.logger = logging.getLogger(__name__)
        
        # Default preferences
        self.default_preferences = {
            "preferred_model": None,  # Let auto-selection work by default
            "auto_model_selection": True,  # Enable smart model selection
            "response_style": "balanced",  # balanced, concise, detailed
            "language": "auto",  # auto-detect or specific language
            "timezone": "UTC",
            "show_model_suggestions": True,  # Show why a model was chosen
            "enable_conversation_summary": True,
            "max_response_length": "medium",  # short, medium, long
            "code_execution_allowed": True,
            "image_generation_style": "default",
            "notification_reminders": True,
            "analytics_opt_in": True,  # Allow usage analytics
            "theme": "default",  # For future UI customization
            "created_at": None,
            "updated_at": None
        }
    
    async def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """
        Get user preferences, creating defaults if none exist.
        
        Args:
            user_id (int): Discord user ID
            
        Returns:
            Dict[str, Any]: User preferences
        """
        try:
            cache_key = f"user_prefs_{user_id}"
            
            async def fetch_preferences():
                user_prefs = await self.db.db.user_preferences.find_one({'user_id': user_id})
                if user_prefs:
                    # Merge with defaults to ensure all keys exist
                    prefs = self.default_preferences.copy()
                    prefs.update(user_prefs.get('preferences', {}))
                    return prefs
                else:
                    # Create default preferences
                    new_prefs = self.default_preferences.copy()
                    new_prefs['created_at'] = datetime.now(timezone.utc)
                    new_prefs['updated_at'] = datetime.now(timezone.utc)
                    
                    await self.db.db.user_preferences.update_one(
                        {'user_id': user_id},
                        {'$set': {'preferences': new_prefs}},
                        upsert=True
                    )
                    return new_prefs
            
            return await self.db._get_cached_result(cache_key, fetch_preferences, 300)  # 5 min cache
            
        except Exception as e:
            self.logger.error(f"Error getting user preferences for {user_id}: {str(e)}")
            return self.default_preferences.copy()
    
    async def update_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences.
        
        Args:
            user_id (int): Discord user ID
            preferences (Dict[str, Any]): Preferences to update
            
        Returns:
            bool: Success status
        """
        try:
            # Get current preferences
            current_prefs = await self.get_user_preferences(user_id)
            
            # Update with new preferences
            current_prefs.update(preferences)
            current_prefs['updated_at'] = datetime.now(timezone.utc)
            
            # Validate preferences
            validated_prefs = self._validate_preferences(current_prefs)
            
            # Save to database
            await self.db.db.user_preferences.update_one(
                {'user_id': user_id},
                {'$set': {'preferences': validated_prefs}},
                upsert=True
            )
            
            # Clear cache
            cache_key = f"user_prefs_{user_id}"
            if cache_key in self.db.cache:
                del self.db.cache[cache_key]
            
            self.logger.info(f"Updated preferences for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating preferences for user {user_id}: {str(e)}")
            return False
    
    def _validate_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize user preferences.
        
        Args:
            preferences (Dict[str, Any]): Raw preferences
            
        Returns:
            Dict[str, Any]: Validated preferences
        """
        validated = {}
        
        # Validate each preference
        for key, value in preferences.items():
            if key == "preferred_model":
                # Validate model exists in available models
                from src.config.config import MODEL_OPTIONS
                if value is None or value in MODEL_OPTIONS:
                    validated[key] = value
                else:
                    validated[key] = None
                    
            elif key == "response_style":
                if value in ["balanced", "concise", "detailed"]:
                    validated[key] = value
                else:
                    validated[key] = "balanced"
                    
            elif key == "max_response_length":
                if value in ["short", "medium", "long"]:
                    validated[key] = value
                else:
                    validated[key] = "medium"
                    
            elif key == "image_generation_style":
                if value in ["default", "artistic", "realistic", "cartoon"]:
                    validated[key] = value
                else:
                    validated[key] = "default"
                    
            elif key in ["auto_model_selection", "show_model_suggestions", "enable_conversation_summary", 
                        "code_execution_allowed", "notification_reminders", "analytics_opt_in"]:
                # Handle string representations of booleans
                if isinstance(value, str):
                    validated[key] = value.lower() in ['true', '1', 'yes', 'on']
                else:
                    validated[key] = bool(value)
                
            elif key in ["language", "timezone", "theme"]:
                validated[key] = str(value) if value else self.default_preferences[key]
                
            elif key in ["created_at", "updated_at"]:
                validated[key] = value  # Keep as-is for datetime objects
                
            else:
                # Unknown preference, keep default
                if key in self.default_preferences:
                    validated[key] = self.default_preferences[key]
        
        # Ensure all default keys exist
        for key, default_value in self.default_preferences.items():
            if key not in validated:
                validated[key] = default_value
        
        return validated
    
    async def get_preference(self, user_id: int, preference_key: str) -> Any:
        """
        Get a specific preference value.
        
        Args:
            user_id (int): Discord user ID
            preference_key (str): Preference key to get
            
        Returns:
            Any: Preference value
        """
        preferences = await self.get_user_preferences(user_id)
        return preferences.get(preference_key, self.default_preferences.get(preference_key))
    
    async def set_preference(self, user_id: int, preference_key: str, value: Any) -> bool:
        """
        Set a specific preference value.
        
        Args:
            user_id (int): Discord user ID
            preference_key (str): Preference key to set
            value (Any): New preference value
            
        Returns:
            bool: Success status
        """
        return await self.update_user_preferences(user_id, {preference_key: value})
    
    async def reset_preferences(self, user_id: int) -> bool:
        """
        Reset user preferences to defaults.
        
        Args:
            user_id (int): Discord user ID
            
        Returns:
            bool: Success status
        """
        try:
            default_prefs = self.default_preferences.copy()
            default_prefs['created_at'] = datetime.now(timezone.utc)
            default_prefs['updated_at'] = datetime.now(timezone.utc)
            
            await self.db.db.user_preferences.update_one(
                {'user_id': user_id},
                {'$set': {'preferences': default_prefs}},
                upsert=True
            )
            
            # Clear cache
            cache_key = f"user_prefs_{user_id}"
            if cache_key in self.db.cache:
                del self.db.cache[cache_key]
            
            self.logger.info(f"Reset preferences for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting preferences for user {user_id}: {str(e)}")
            return False
    
    def format_preferences_display(self, preferences: Dict[str, Any]) -> str:
        """
        Format preferences for display to user.
        
        Args:
            preferences (Dict[str, Any]): User preferences
            
        Returns:
            str: Formatted preference display
        """
        display_lines = [
            "**Your Current Preferences:**",
            "",
            f"🤖 **Model Settings:**",
            f"  • Preferred Model: `{preferences.get('preferred_model', 'Auto-select')}`",
            f"  • Auto Model Selection: `{'✅' if preferences.get('auto_model_selection') else '❌'}`",
            f"  • Show Model Suggestions: `{'✅' if preferences.get('show_model_suggestions') else '❌'}`",
            "",
            f"💬 **Response Settings:**",
            f"  • Response Style: `{preferences.get('response_style', 'balanced').title()}`",
            f"  • Max Response Length: `{preferences.get('max_response_length', 'medium').title()}`",
            f"  • Language: `{preferences.get('language', 'auto')}`",
            "",
            f"🔧 **Feature Settings:**",
            f"  • Code Execution: `{'✅' if preferences.get('code_execution_allowed') else '❌'}`",
            f"  • Conversation Summary: `{'✅' if preferences.get('enable_conversation_summary') else '❌'}`",
            f"  • Reminder Notifications: `{'✅' if preferences.get('notification_reminders') else '❌'}`",
            "",
            f"🎨 **Creative Settings:**",
            f"  • Image Generation Style: `{preferences.get('image_generation_style', 'default').title()}`",
            "",
            f"📊 **Privacy Settings:**",
            f"  • Usage Analytics: `{'✅' if preferences.get('analytics_opt_in') else '❌'}`",
            "",
            f"*Use `/preferences set` to modify these settings*"
        ]
        
        return "\n".join(display_lines)