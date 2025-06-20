import asyncio
import unittest
import os
import sys
import json
import io
from unittest.mock import MagicMock, patch, AsyncMock
from dotenv import load_dotenv
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules for testing
from src.database.db_handler import DatabaseHandler
from src.utils.openai_utils import count_tokens, trim_content_to_token_limit, prepare_messages_for_api
from src.utils.code_utils import sanitize_code, extract_code_blocks
from src.utils.web_utils import scrape_web_content
from src.utils.pdf_utils import send_response


class TestDatabaseHandler(unittest.IsolatedAsyncioTestCase):
    """Test database handler functionality"""

    def setUp(self):
        # Load environment variables
        load_dotenv()
        
        # Try to get MongoDB URI from environment
        self.mongodb_uri = os.getenv("MONGODB_URI")
        self.using_real_db = bool(self.mongodb_uri)
        
        if not self.using_real_db:
            # Use mock if no real URI available
            self.mock_client_patcher = patch('motor.motor_asyncio.AsyncIOMotorClient')
            self.mock_client = self.mock_client_patcher.start()
            
            # Setup mock database and collections
            self.mock_db = self.mock_client.return_value.__getitem__.return_value
            self.mock_histories = MagicMock()
            self.mock_models = MagicMock()  # Store mock_models as instance variable
            self.mock_db.__getitem__.side_effect = lambda x: {
                'user_histories': self.mock_histories,
                'user_models': self.mock_models,  # Use the instance variable
                'whitelist': MagicMock(),
                'blacklist': MagicMock()
            }[x]
            
            # Initialize handler with mock connection string
            self.db_handler = DatabaseHandler("mongodb://localhost:27017")
        else:
            # Use real database connection
            print(f"Testing with real MongoDB at: {self.mongodb_uri}")
            self.db_handler = DatabaseHandler(self.mongodb_uri)
            
            # Extract database name from URI for later use
            self.db_name = self._extract_db_name_from_uri(self.mongodb_uri)
    
    async def asyncSetUp(self):
        # No additional async setup needed, but required by IsolatedAsyncioTestCase
        pass
    
    async def asyncTearDown(self):
        if not self.using_real_db:
            self.mock_client_patcher.stop()
        else:
            # Clean up test data if using real database
            await self.cleanup_test_data()
    
    def _extract_db_name_from_uri(self, uri):
        """Extract database name from MongoDB URI more reliably"""
        # Default database name if extraction fails
        default_db_name = 'chatgpt_discord_bot'
        
        try:
            # Handle standard MongoDB URI format
            # mongodb://[username:password@]host1[:port1][,...hostN[:portN]][/database][?options]
            match = re.search(r'\/([^/?]+)(\?|$)', uri)
            if match:
                return match.group(1)
            
            # If no database in URI, return default
            return default_db_name
        except:
            # If any error occurs, return default name
            return default_db_name
            
    async def cleanup_test_data(self):
        """Remove test data from real database"""
        if self.using_real_db:
            try:
                # Use the database name we extracted in setUp
                db = self.db_handler.client.get_database(self.db_name)
                await db.user_histories.delete_one({'user_id': 12345})
                await db.user_models.delete_one({'user_id': 12345})
            except Exception as e:
                print(f"Error during test cleanup: {e}")
        
    async def test_get_history_empty(self):
        if self.using_real_db:
            # Clean up any existing history first
            await self.cleanup_test_data()
            # Test with real database
            result = await self.db_handler.get_history(12345)
            self.assertEqual(result, [])
        else:
            # Mock find_one to return None (no history)
            self.mock_histories.find_one = AsyncMock(return_value=None)
            
            # Test getting non-existent history
            result = await self.db_handler.get_history(12345)
            self.assertEqual(result, [])
            self.mock_histories.find_one.assert_called_once_with({'user_id': 12345})
        
    async def test_get_history_existing(self):
        # Sample history data
        sample_history = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]
        
        if self.using_real_db:
            # Save test history first
            await self.db_handler.save_history(12345, sample_history)
            
            # Test getting existing history
            result = await self.db_handler.get_history(12345)
            self.assertEqual(result, sample_history)
        else:
            # Mock find_one to return existing history
            self.mock_histories.find_one = AsyncMock(return_value={'user_id': 12345, 'history': sample_history})
            
            # Test getting existing history
            result = await self.db_handler.get_history(12345)
            self.assertEqual(result, sample_history)
        
    async def test_save_history(self):
        # Sample history to save
        sample_history = [
            {'role': 'user', 'content': 'Test message'},
            {'role': 'assistant', 'content': 'Test response'}
        ]
        
        if self.using_real_db:
            # Test saving history to real database
            await self.db_handler.save_history(12345, sample_history)
            
            # Verify it was saved
            result = await self.db_handler.get_history(12345)
            self.assertEqual(result, sample_history)
        else:
            # Mock update_one method
            self.mock_histories.update_one = AsyncMock()
            
            # Test saving history
            await self.db_handler.save_history(12345, sample_history)
            
            # Verify update_one was called with correct parameters
            self.mock_histories.update_one.assert_called_once_with(
                {'user_id': 12345},
                {'$set': {'history': sample_history}},
                upsert=True
            )
        
    async def test_user_model_operations(self):
        if self.using_real_db:
            # Save a model and then retrieve it
            await self.db_handler.save_user_model(12345, 'openai/gpt-4o')
            model = await self.db_handler.get_user_model(12345)
            self.assertEqual(model, 'openai/gpt-4o')
            
            # Test updating model
            await self.db_handler.save_user_model(12345, 'openai/gpt-4o-mini')
            updated_model = await self.db_handler.get_user_model(12345)
            self.assertEqual(updated_model, 'openai/gpt-4o-mini')
        else:
            # Setup mock for user_models collection
            # Use self.mock_models instead of creating a new mock
            self.mock_models.find_one = AsyncMock(return_value={'user_id': 12345, 'model': 'openai/gpt-4o'})
            self.mock_models.update_one = AsyncMock()
            
            # Test getting user model
            model = await self.db_handler.get_user_model(12345)
            self.assertEqual(model, 'openai/gpt-4o')
            
            # Test saving user model
            await self.db_handler.save_user_model(12345, 'openai/gpt-4o-mini')
            self.mock_models.update_one.assert_called_once_with(
                {'user_id': 12345},
                {'$set': {'model': 'openai/gpt-4o-mini'}},
                upsert=True
            )


class TestOpenAIUtils(unittest.TestCase):
    """Test OpenAI utility functions"""
    
    def test_count_tokens(self):
        # Test token counting
        self.assertGreater(count_tokens("Hello, world!"), 0)
        self.assertGreater(count_tokens("This is a longer text that should have more tokens."), 
                           count_tokens("Short text"))
                           
    def test_trim_content_to_token_limit(self):
        # Create a long text
        long_text = "This is a test. " * 1000
        
        # Test trimming
        trimmed = trim_content_to_token_limit(long_text, 100)
        self.assertLess(count_tokens(trimmed), count_tokens(long_text))
        self.assertLessEqual(count_tokens(trimmed), 100)
        
        # Test no trimming needed
        short_text = "This is a short text."
        untrimmed = trim_content_to_token_limit(short_text, 100)
        self.assertEqual(untrimmed, short_text)
        
    def test_prepare_messages_for_api(self):
        # Test empty messages
        empty_result = prepare_messages_for_api([])
        self.assertEqual(len(empty_result), 1)  # Should have system message
        self.assertEqual(empty_result[0]["role"], "system")  # Verify it's a system message
        
        # Test regular messages
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        result = prepare_messages_for_api(messages)
        self.assertEqual(len(result), 4)  # Should have system message + 3 original messages
        
        # Test with null content
        messages_with_null = [
            {"role": "user", "content": None},
            {"role": "assistant", "content": "Response"}
        ]
        result_fixed = prepare_messages_for_api(messages_with_null)
        self.assertEqual(len(result_fixed), 2)  # Should have system message + 1 valid message
        # Verify the content is correct (system message + only the assistant message)
        self.assertEqual(result_fixed[0]["role"], "system")
        self.assertEqual(result_fixed[1]["role"], "assistant")
        self.assertEqual(result_fixed[1]["content"], "Response")

class TestCodeUtils(unittest.TestCase):
    """Test code utility functions"""
    
    def test_sanitize_python_code_safe(self):
        # Safe Python code
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
    
print(factorial(5))
"""
        is_safe, sanitized = sanitize_code(code, "python")
        self.assertTrue(is_safe)
        self.assertIn("def factorial", sanitized)
        
    def test_sanitize_python_code_unsafe(self):
        # Unsafe Python code with os.system
        unsafe_code = """
import os
os.system('rm -rf /')
"""
        is_safe, message = sanitize_code(unsafe_code, "python")
        self.assertFalse(is_safe)
        self.assertIn("Forbidden", message)
        
    def test_sanitize_cpp_code_safe(self):
        # Safe C++ code
        code = """
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, world!" << endl;
    return 0;
}
"""
        is_safe, sanitized = sanitize_code(code, "cpp")
        self.assertTrue(is_safe)
        self.assertIn("Hello, world!", sanitized)
        
    def test_sanitize_cpp_code_unsafe(self):
        # Unsafe C++ code with system
        unsafe_code = """
#include <stdlib.h>
int main() {
    system("rm -rf /");
    return 0;
}
"""
        is_safe, message = sanitize_code(unsafe_code, "cpp")
        self.assertFalse(is_safe)
        self.assertIn("Forbidden", message)
        
    def test_extract_code_blocks(self):
        # Test message with code block
        message = """
Here's a Python function to calculate factorial:
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
```
And here's a C++ version:
```cpp
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n-1);
}
```
"""
        blocks = extract_code_blocks(message)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0][0], "python")
        self.assertEqual(blocks[1][0], "cpp")
        
        # Test without language specifier
        message_no_lang = """
Here's some code:
```
print("Hello world")
```
"""
        blocks_no_lang = extract_code_blocks(message_no_lang)
        self.assertEqual(len(blocks_no_lang), 1)


#class TestWebUtils(unittest.TestCase):
#    """Test web utilities"""
#    
#    @patch('requests.get')
#    def test_scrape_web_content(self, mock_get):
#        # Mock the response
#        mock_response = MagicMock()
#        mock_response.text = '<html><body><h1>Test Heading</h1><p>Test paragraph</p></body></html>'
#        mock_response.status_code = 200
#        mock_get.return_value = mock_response
#        
#        # Test scraping
#        content = scrape_web_content("example.com")
#        self.assertIn("Test Heading", content)
#        self.assertIn("Test paragraph", content)
        
class TestPDFUtils(unittest.IsolatedAsyncioTestCase):
    """Test PDF utilities"""
    
    async def test_send_response(self):
        # Create mock channel
        mock_channel = AsyncMock()
        mock_channel.send = AsyncMock()
        
        # Test sending short response
        short_response = "This is a short response"
        await send_response(mock_channel, short_response)
        mock_channel.send.assert_called_once_with(short_response)
        
        # Reset mock
        mock_channel.send.reset_mock()
        
        # Mock for long response (testing would need file operations)
        with patch('builtins.open', new_callable=unittest.mock.mock_open):
            with patch('discord.File', return_value="mocked_file"):
                # Test sending long response
                long_response = "X" * 2500  # Over 2000 character limit
                await send_response(mock_channel, long_response)
                mock_channel.send.assert_called_once()
                # Verify it's called with the file argument
                args, kwargs = mock_channel.send.call_args
                self.assertIn('file', kwargs)


if __name__ == "__main__":
    unittest.main()