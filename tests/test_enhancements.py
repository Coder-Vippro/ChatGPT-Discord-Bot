"""
Tests for enhanced Discord bot features
"""

import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.model_selector import ModelSelector, model_selector
from src.utils.user_preferences import UserPreferences
from src.utils.conversation_manager import ConversationSummarizer
from src.utils.enhanced_file_processor import EnhancedFileProcessor, enhanced_file_processor


class TestModelSelector(unittest.TestCase):
    """Test smart model selection functionality."""
    
    def setUp(self):
        self.selector = ModelSelector()
    
    def test_analyze_task_type_coding(self):
        """Test detection of coding tasks."""
        coding_prompts = [
            "Write a Python function to calculate fibonacci",
            "Help me debug this JavaScript code",
            "Create a REST API in Node.js"
        ]
        
        for prompt in coding_prompts:
            task_type = self.selector.analyze_task_type(prompt)
            self.assertEqual(task_type, "coding", f"Failed for prompt: {prompt}")
    
    def test_analyze_task_type_reasoning(self):
        """Test detection of reasoning tasks."""
        reasoning_prompts = [
            "Solve this math problem step by step",
            "Analyze the logic behind this algorithm",
            "What is the relationship between these variables?"
        ]
        
        for prompt in reasoning_prompts:
            task_type = self.selector.analyze_task_type(prompt)
            self.assertEqual(task_type, "reasoning", f"Failed for prompt: {prompt}")
    
    def test_analyze_task_type_creative(self):
        """Test detection of creative tasks."""
        creative_prompts = [
            "Write a story about a dragon",
            "Create a marketing slogan for our product",
            "Generate a poem about nature"
        ]
        
        for prompt in creative_prompts:
            task_type = self.selector.analyze_task_type(prompt)
            self.assertEqual(task_type, "creative", f"Failed for prompt: {prompt}")
    
    def test_suggest_model_for_coding(self):
        """Test model suggestion for coding tasks."""
        model, reason = self.selector.suggest_model("Write a Python function to sort a list")
        self.assertIn("openai/gpt-4o", model)
        self.assertIn("coding", reason.lower())
    
    def test_suggest_model_for_reasoning(self):
        """Test model suggestion for reasoning tasks."""
        model, reason = self.selector.suggest_model("Solve this complex mathematical proof step by step")
        self.assertIn("o1", model)  # Should suggest o1 family for reasoning
        self.assertIn("reasoning", reason.lower())
    
    def test_suggest_model_with_preference(self):
        """Test that user preference is respected."""
        preferred_model = "openai/gpt-4o-mini"
        model, reason = self.selector.suggest_model("Any task", preferred_model)
        self.assertEqual(model, preferred_model)
        self.assertIn("preferred", reason.lower())
    
    def test_get_model_explanation(self):
        """Test model explanations."""
        explanation = self.selector.get_model_explanation("openai/gpt-4o")
        self.assertIsInstance(explanation, str)
        self.assertTrue(len(explanation) > 10)


class TestUserPreferences(unittest.IsolatedAsyncioTestCase):
    """Test user preferences system."""
    
    def setUp(self):
        # Mock database handler
        self.mock_db = MagicMock()
        self.mock_db.db = MagicMock()
        self.mock_db.db.user_preferences = MagicMock()
        self.mock_db._get_cached_result = AsyncMock()
        self.mock_db.cache = {}
        
        self.prefs_manager = UserPreferences(self.mock_db)
    
    async def test_get_default_preferences(self):
        """Test getting default preferences for new user."""
        # Mock no existing preferences
        self.mock_db._get_cached_result.return_value = self.prefs_manager.default_preferences.copy()
        
        prefs = await self.prefs_manager.get_user_preferences(12345)
        
        # Should return default preferences
        self.assertEqual(prefs['auto_model_selection'], True)
        self.assertEqual(prefs['response_style'], 'balanced')
        self.assertIsNone(prefs['preferred_model'])
    
    async def test_update_preferences(self):
        """Test updating user preferences."""
        # Mock existing preferences
        self.mock_db._get_cached_result.return_value = self.prefs_manager.default_preferences.copy()
        self.mock_db.db.user_preferences.update_one = AsyncMock()
        
        # Update a preference
        success = await self.prefs_manager.update_user_preferences(12345, {
            'response_style': 'detailed',
            'preferred_model': 'openai/gpt-4o'
        })
        
        self.assertTrue(success)
        self.mock_db.db.user_preferences.update_one.assert_called_once()
    
    async def test_validate_preferences(self):
        """Test preference validation."""
        invalid_prefs = {
            'response_style': 'invalid_style',
            'preferred_model': 'invalid_model',
            'auto_model_selection': 'false'  # String instead of boolean
        }
        
        validated = self.prefs_manager._validate_preferences(invalid_prefs)
        
        # Should fall back to defaults for invalid values
        self.assertEqual(validated['response_style'], 'balanced')
        self.assertIsNone(validated['preferred_model'])
        self.assertFalse(validated['auto_model_selection'])  # String 'false' should become boolean False
    
    def test_format_preferences_display(self):
        """Test preference display formatting."""
        prefs = self.prefs_manager.default_preferences.copy()
        prefs['preferred_model'] = 'openai/gpt-4o'
        
        display = self.prefs_manager.format_preferences_display(prefs)
        
        self.assertIsInstance(display, str)
        self.assertIn('openai/gpt-4o', display)
        self.assertIn('Preferences', display)


class TestConversationSummarizer(unittest.IsolatedAsyncioTestCase):
    """Test conversation summarization functionality."""
    
    def setUp(self):
        # Mock OpenAI client
        self.mock_client = MagicMock()
        self.mock_client.chat = MagicMock()
        self.mock_client.chat.completions = MagicMock()
        self.mock_client.chat.completions.create = AsyncMock()
        
        # Mock database handler
        self.mock_db = MagicMock()
        self.mock_db.get_user_model = AsyncMock(return_value="openai/gpt-4o-mini")
        
        # Mock tiktoken to avoid network calls
        with patch('tiktoken.get_encoding') as mock_encoding:
            mock_encoder = MagicMock()
            mock_encoder.encode = MagicMock(return_value=[1, 2, 3, 4])  # Mock 4 tokens
            mock_encoding.return_value = mock_encoder
            
            self.summarizer = ConversationSummarizer(self.mock_client, self.mock_db)
    
    def test_count_tokens(self):
        """Test token counting."""
        text = "Hello, world! This is a test message."
        tokens = self.summarizer.count_tokens(text)
        self.assertGreater(tokens, 0)
        self.assertIsInstance(tokens, int)
    
    def test_should_summarize_short_conversation(self):
        """Test that short conversations are not summarized."""
        short_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = asyncio.run(self.summarizer.should_summarize(short_messages))
        self.assertFalse(result)
    
    def test_should_summarize_long_conversation(self):
        """Test that long conversations trigger summarization."""
        # Create a long conversation
        long_messages = []
        long_text = "This is a very long message. " * 100  # Make it long
        
        for i in range(10):
            long_messages.append({"role": "user", "content": long_text})
            long_messages.append({"role": "assistant", "content": long_text})
        
        result = asyncio.run(self.summarizer.should_summarize(long_messages))
        self.assertTrue(result)
    
    async def test_create_summary(self):
        """Test summary creation."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a test summary."
        self.mock_client.chat.completions.create.return_value = mock_response
        
        messages = [
            {"role": "user", "content": "Tell me about AI"},
            {"role": "assistant", "content": "AI is artificial intelligence..."},
            {"role": "user", "content": "What about machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI..."}
        ]
        
        summary = await self.summarizer.create_summary(messages, 12345)
        
        self.assertIsInstance(summary, str)
        self.assertEqual(summary, "This is a test summary.")
        self.mock_client.chat.completions.create.assert_called_once()
    
    def test_trim_messages(self):
        """Test message trimming fallback."""
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(30)]
        
        trimmed = self.summarizer._trim_messages(messages, max_messages=10)
        
        self.assertEqual(len(trimmed), 10)
        self.assertEqual(trimmed[-1]["content"], "Message 29")  # Should keep most recent


class TestEnhancedFileProcessor(unittest.IsolatedAsyncioTestCase):
    """Test enhanced file processing functionality."""
    
    def setUp(self):
        self.processor = EnhancedFileProcessor()
    
    def test_get_supported_extensions(self):
        """Test getting supported file extensions."""
        extensions = self.processor.get_supported_extensions()
        self.assertIsInstance(extensions, list)
        self.assertIn('.txt', extensions)
        self.assertIn('.py', extensions)
        self.assertIn('.json', extensions)
    
    def test_is_supported(self):
        """Test file support detection."""
        self.assertTrue(self.processor.is_supported('test.txt'))
        self.assertTrue(self.processor.is_supported('script.py'))
        self.assertTrue(self.processor.is_supported('data.json'))
        self.assertFalse(self.processor.is_supported('image.png'))  # Not in text processors
    
    async def test_process_text_file(self):
        """Test text file processing."""
        # Create a temporary text file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello, world!\nThis is a test file.")
            temp_path = f.name
        
        try:
            result = await self.processor.process_file(temp_path, "test.txt")
            
            self.assertTrue(result['success'])
            self.assertIn("Hello, world!", result['content'])
            self.assertEqual(result['metadata']['type'], 'text')
            self.assertEqual(result['metadata']['lines'], 2)
            
        finally:
            os.unlink(temp_path)
    
    async def test_process_json_file(self):
        """Test JSON file processing."""
        import tempfile
        import json
        
        test_data = {"name": "test", "value": 123, "items": [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            result = await self.processor.process_file(temp_path, "test.json")
            
            self.assertTrue(result['success'])
            self.assertIn('"name": "test"', result['content'])
            self.assertEqual(result['metadata']['type'], 'json')
            self.assertTrue(result['metadata']['is_valid'])
            
        finally:
            os.unlink(temp_path)
    
    async def test_process_unsupported_file(self):
        """Test handling of unsupported file types."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as f:
            f.write(b"some binary data")
            temp_path = f.name
        
        try:
            result = await self.processor.process_file(temp_path, "test.unknown")
            
            self.assertFalse(result['success'])
            self.assertIn("Unsupported file type", result['error'])
            
        finally:
            os.unlink(temp_path)
    
    def test_format_file_size(self):
        """Test file size formatting."""
        self.assertEqual(self.processor._format_file_size(1024), "1.0 KB")
        self.assertEqual(self.processor._format_file_size(1048576), "1.0 MB")
        self.assertEqual(self.processor._format_file_size(500), "500.0 B")


if __name__ == "__main__":
    # Run tests
    unittest.main()