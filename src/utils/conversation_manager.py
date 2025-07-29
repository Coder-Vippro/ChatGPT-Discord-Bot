"""
Conversation Summarization Utility
Manages conversation context by creating smart summaries when conversations get too long.
"""

import logging
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

class ConversationSummarizer:
    """Handles conversation summarization for better context management."""
    
    def __init__(self, openai_client, db_handler):
        self.client = openai_client
        self.db = db_handler
        self.logger = logging.getLogger(__name__)
        self.encoding = tiktoken.get_encoding("o200k_base")
        
        # Configuration
        self.max_context_tokens = 6000  # When to start summarizing
        self.summary_target_tokens = 2000  # Target size for summary
        self.min_messages_to_summarize = 4  # Minimum messages before summarizing
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(self.encoding.encode(str(text)))
        except Exception:
            # Fallback estimation
            return len(str(text)) // 4
    
    def count_conversation_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count total tokens in conversation history."""
        total_tokens = 0
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, list):
                # Handle multimodal content
                for part in content:
                    if isinstance(part, dict) and 'text' in part:
                        total_tokens += self.count_tokens(part['text'])
            else:
                total_tokens += self.count_tokens(str(content))
        return total_tokens
    
    async def should_summarize(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Determine if conversation should be summarized.
        
        Args:
            messages: Conversation history
            
        Returns:
            bool: Whether to summarize
        """
        if len(messages) < self.min_messages_to_summarize:
            return False
        
        token_count = self.count_conversation_tokens(messages)
        return token_count > self.max_context_tokens
    
    async def create_summary(self, messages: List[Dict[str, Any]], user_id: int) -> Optional[str]:
        """
        Create a summary of the conversation.
        
        Args:
            messages: Conversation history to summarize
            user_id: User ID for context
            
        Returns:
            Optional[str]: Summary of the conversation
        """
        try:
            if len(messages) < 2:
                return None
            
            # Prepare messages for summarization
            conversation_text = self._format_messages_for_summary(messages)
            
            # Create summary prompt
            summary_prompt = """Please create a concise summary of this conversation that preserves:
1. Key topics discussed
2. Important decisions or conclusions reached
3. Ongoing context that might be relevant for future messages
4. User preferences or specific requests mentioned

Keep the summary under 500 words and focus on information that would help continue the conversation naturally.

Conversation to summarize:
""" + conversation_text
            
            # Get user's preferred model for summarization (prefer efficient models)
            user_prefs = await self.db.get_user_model(user_id)
            summary_model = "openai/gpt-4o-mini"  # Use efficient model for summaries
            
            response = await self.client.chat.completions.create(
                model=summary_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, informative conversation summaries."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            summary = response.choices[0].message.content
            
            self.logger.info(f"Created conversation summary for user {user_id} ({len(messages)} messages)")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating conversation summary: {str(e)}")
            return None
    
    def _format_messages_for_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for summarization."""
        formatted_lines = []
        
        for i, message in enumerate(messages):
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            
            # Handle multimodal content
            if isinstance(content, list):
                content_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if 'text' in part:
                            content_parts.append(part['text'])
                        elif 'type' in part:
                            content_parts.append(f"[{part['type']} content]")
                content = " ".join(content_parts)
            
            # Truncate very long messages
            if len(str(content)) > 1000:
                content = str(content)[:1000] + "... [truncated]"
            
            formatted_lines.append(f"{role.upper()}: {content}")
        
        return "\n\n".join(formatted_lines)
    
    async def manage_conversation_length(self, user_id: int, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Manage conversation length by summarizing when needed.
        
        Args:
            user_id: User ID
            messages: Current conversation history
            
        Returns:
            List[Dict[str, Any]]: Managed conversation history
        """
        try:
            # Check if summarization is needed
            if not await self.should_summarize(messages):
                return messages
            
            # Check user preferences
            try:
                from src.utils.user_preferences import UserPreferences
                user_prefs_manager = UserPreferences(self.db)
                prefs = await user_prefs_manager.get_user_preferences(user_id)
                
                if not prefs.get('enable_conversation_summary', True):
                    # User disabled summarization, just trim older messages
                    return self._trim_messages(messages)
            except Exception:
                # If preferences system fails, continue with summarization
                pass
            
            # Find split point (keep recent messages, summarize older ones)
            recent_tokens = 0
            split_index = len(messages)
            
            # Work backwards to find where to split
            for i in range(len(messages) - 1, -1, -1):
                message_tokens = self.count_tokens(str(messages[i].get('content', '')))
                if recent_tokens + message_tokens > self.summary_target_tokens:
                    split_index = i + 1
                    break
                recent_tokens += message_tokens
            
            # Don't summarize if we'd only be summarizing a few messages
            if split_index >= len(messages) - 2:
                return self._trim_messages(messages)
            
            # Split conversation
            messages_to_summarize = messages[:split_index]
            recent_messages = messages[split_index:]
            
            # Create summary
            summary = await self.create_summary(messages_to_summarize, user_id)
            
            if summary:
                # Create new conversation starting with summary
                summary_message = {
                    "role": "system",
                    "content": f"[Conversation Summary] {summary}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": "summary"
                }
                
                managed_messages = [summary_message] + recent_messages
                
                self.logger.info(f"Summarized {len(messages_to_summarize)} messages into summary for user {user_id}")
                return managed_messages
            else:
                # Fallback to simple trimming if summarization fails
                return self._trim_messages(messages)
                
        except Exception as e:
            self.logger.error(f"Error managing conversation length: {str(e)}")
            return self._trim_messages(messages)
    
    def _trim_messages(self, messages: List[Dict[str, Any]], max_messages: int = 20) -> List[Dict[str, Any]]:
        """
        Simple fallback: trim to recent messages.
        
        Args:
            messages: Messages to trim
            max_messages: Maximum number of messages to keep
            
        Returns:
            List[Dict[str, Any]]: Trimmed messages
        """
        if len(messages) <= max_messages:
            return messages
        
        # Keep the most recent messages
        return messages[-max_messages:]
    
    async def get_conversation_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the conversation.
        
        Args:
            messages: Conversation messages
            
        Returns:
            Dict[str, Any]: Conversation statistics
        """
        total_messages = len(messages)
        total_tokens = self.count_conversation_tokens(messages)
        
        user_messages = [m for m in messages if m.get('role') == 'user']
        assistant_messages = [m for m in messages if m.get('role') == 'assistant']
        summary_messages = [m for m in messages if m.get('type') == 'summary']
        
        return {
            "total_messages": total_messages,
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "summary_messages": len(summary_messages),
            "total_tokens": total_tokens,
            "needs_summary": total_tokens > self.max_context_tokens,
            "token_limit": self.max_context_tokens
        }