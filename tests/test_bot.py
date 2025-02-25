import sys
import os
import unittest
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import bot

class TestBotUtils(unittest.TestCase):
    
    def test_count_tokens(self):
        """Test token counting function"""
        # Basic test
        self.assertGreater(bot.count_tokens("Hello world"), 0)
        # Empty string should return 0 or small value
        self.assertLessEqual(bot.count_tokens(""), 3)
        # Longer text should have more tokens
        short_text = "Hello"
        long_text = "Hello " * 100
        self.assertLess(bot.count_tokens(short_text), bot.count_tokens(long_text))
    

    def test_prepare_messages_for_api(self):
        """Test message preparation for API"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Help me with Python"}
        ]
        prepared = bot.prepare_messages_for_api(messages, max_tokens=1000)
        self.assertIsInstance(prepared, list)
        # Should have role and content
        for msg in prepared:
            self.assertIn("role", msg)
            self.assertIn("content", msg)

class TestCodeSanitization(unittest.TestCase):
    
    def test_python_safe_code(self):
        """Test Python code sanitization with safe code"""
        code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
        
print(factorial(5))
"""
        is_safe, sanitized = bot.sanitize_code(code, "python")
        self.assertTrue(is_safe)
        self.assertIn("signal.alarm(10)", sanitized)
        
    def test_python_unsafe_code(self):
        """Test Python code sanitization with unsafe imports"""
        code = """
import os
print(os.system('ls'))
"""
        is_safe, message = bot.sanitize_code(code, "python")
        self.assertFalse(is_safe)
        self.assertIn("Forbidden module import", message)
        
    def test_cpp_safe_code(self):
        """Test C++ code sanitization with safe code"""
        code = """
#include <iostream>
using namespace std;

int main() {
    cout << "Hello World" << endl;
    return 0;
}
"""
        is_safe, sanitized = bot.sanitize_code(code, "cpp")
        self.assertTrue(is_safe)
        self.assertIn("userMain(", sanitized)
        
    def test_cpp_unsafe_code(self):
        """Test C++ code sanitization with unsafe includes"""
        code = """
#include <iostream>
#include <fstream>
int main() {
    ofstream file("test.txt");
    file << "Hello World";
    return 0;
}
"""
        is_safe, message = bot.sanitize_code(code, "cpp")
        self.assertFalse(is_safe)
        self.assertIn("Forbidden header", message)

    def test_python_eval_exec_detection(self):
        """Test detection of eval/exec in Python code"""
        code = """
print("Hello")
eval("print('This is dangerous')")
"""
        is_safe, message = bot.sanitize_code(code, "python")
        self.assertFalse(is_safe)
        self.assertIn("Forbidden", message)

    def test_python_add_missing_structure(self):
        """Test Python code gets proper safety structure"""
        code = "print('Hello world')"
        is_safe, sanitized = bot.sanitize_code(code, "python")
        self.assertTrue(is_safe)
        self.assertIn("try:", sanitized)
        self.assertIn("except Exception as e:", sanitized)
        self.assertIn("finally:", sanitized)
        self.assertIn("signal.alarm(0)", sanitized)

    def test_cpp_add_missing_iostream(self):
        """Test C++ code gets iostream added when using cout"""
        code = """
int main() {
    cout << "Hello" << endl;
    return 0;
}
"""
        is_safe, sanitized = bot.sanitize_code(code, "cpp")
        self.assertTrue(is_safe)
        self.assertIn("#include <iostream>", sanitized)

    def test_cpp_add_missing_namespace(self):
        """Test C++ code gets namespace std added when needed"""
        code = """
#include <iostream>
int main() {
    cout << "Hello" << endl;
    return 0;
}
"""
        is_safe, sanitized = bot.sanitize_code(code, "cpp")
        self.assertTrue(is_safe)
        self.assertIn("using namespace std;", sanitized)

    def test_edge_case_empty_code(self):
        """Test sanitization with empty code"""
        code = ""
        is_safe, sanitized = bot.sanitize_code(code, "python")
        self.assertTrue(is_safe)
        self.assertNotEqual(sanitized, "")  # Should add safety structure

    def test_multiple_forbidden_imports(self):
        """Test code with multiple forbidden imports"""
        code = """
import os
import sys
import subprocess
print("This is malicious")
"""
        is_safe, message = bot.sanitize_code(code, "python")
        self.assertFalse(is_safe)
        self.assertIn("Forbidden", message)


@pytest.mark.asyncio
class TestAsyncFunctions:
    
    @pytest.fixture
    def mock_channel(self):
        mock = AsyncMock()
        mock.send = AsyncMock()
        return mock
        
    @patch('bot.client.chat.completions.create')
    async def test_process_batch(self, mock_create, mock_channel):
        """Test batch processing"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Processed content"
        mock_create.return_value = mock_response
        
        result = await bot.process_batch(
            model="gpt-4o-mini",
            user_prompt="Analyze this",
            batch_content="Test content",
            current_batch=1,
            total_batches=1,
            channel=mock_channel()
        )
        
        # Should call the API
        mock_create.assert_called_once()
        self.assertTrue(result)
        
    @patch('asyncio.create_subprocess_exec')
    async def test_execute_code(self, mock_subprocess):
        """Test code execution"""
        # Configure the mock
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"Hello World", b""))
        mock_subprocess.return_value = mock_proc
        
        result = await bot.execute_code("print('Hello World')", "python")
        self.assertIn("Hello World", result)
        


class TestToolFunctions(unittest.TestCase):
    
    @patch('bot.google_custom_search')
    def test_tool_function_google_search(self, mock_search):
        """Test Google search tool function"""
        mock_search.return_value = {"results": [{"title": "Test", "link": "https://example.com"}]}
        
        result = bot.tool_functions["google_search"]({"query": "test", "num_results": 1})
        mock_search.assert_called_once_with("test", 1)
        self.assertIsInstance(result, dict)
    
    @patch('bot.scrape_web_content')
    def test_tool_function_scrape_webpage(self, mock_scrape):
        """Test web scraping tool function"""
        mock_scrape.return_value = "Scraped content"
        
        result = bot.tool_functions["scrape_webpage"]({"url": "https://example.com"})
        mock_scrape.assert_called_once_with("https://example.com")
        self.assertEqual(result, "Scraped content")

    @patch('bot.requests.get')
    def test_scrape_web_content_detailed(self, mock_requests_get):
        """Test web scraping function with different scenarios"""
        # Setup mock response for successful HTML page with article content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.content = """
        <html><body>
            <article>
                <h1>Test Article</h1>
                <p>This is a test paragraph with meaningful content.</p>
            </article>
        </body></html>
        """
        mock_requests_get.return_value = mock_response
        
        # Test successful scraping with article tag
        content = bot.scrape_web_content("https://example.com")
        self.assertIn("Test Article", content)
        self.assertIn("test paragraph", content)
        
        # Test with HTTP error
        mock_response.status_code = 404
        mock_response.headers = {'content-type': 'text/html'}
        content = bot.scrape_web_content("https://example.com/not-found")
        self.assertIn("Error: Received status code 404", content)
        
        # Test with request exception
        mock_requests_get.side_effect = Exception("Connection error")
        content = bot.scrape_web_content("https://example.com")
        self.assertIn("An error occurred", content)


@pytest.mark.asyncio
class TestErrorHandling:
    
    @pytest.fixture
    def mock_channel(self):
        mock = AsyncMock()
        mock.send = AsyncMock()
        return mock
        
    @patch('bot.client.chat.completions.create')
    async def test_handle_api_error(self, mock_create, mock_channel):
        """Test handling of API errors"""
        mock_create.side_effect = Exception("API Error")
        
        channel = mock_channel()
        # Call function that would use the API
        await bot.send_chatgpt_response(
            prompt="Test prompt",
            channel=channel,
            conversation_history=[],
            user_id="12345"
        )
        
        # Verify error was handled and communicated
        channel.send.assert_called_once()
        args, _ = channel.send.call_args
        self.assertIn("error", args[0].lower())

    @patch('bot.execute_code')
    async def test_code_execution_timeout(self, mock_execute):
        """Test handling of code execution timeout"""
        mock_execute.side_effect = asyncio.TimeoutError()
        
        result = await bot.process_tool_calls(
            {"tool_calls": [{"function": {"name": "code_interpreter", "arguments": '{"code": "print(\\"test\\")", "language": "python"}'}}]},
            []
        )
        
        self.assertIn("execution timed out", result[0]["content"].lower())

class TestCommandHandling(unittest.TestCase):
    
    def setUp(self):
        self.ctx = MagicMock()
        self.ctx.author = MagicMock()
        self.ctx.author.id = "12345"
        self.ctx.send = AsyncMock()
    
    @patch('bot.prepare_messages_for_api')
    @patch('bot.client.chat.completions.create')
    async def test_chat_command(self, mock_create, mock_prepare):
        """Test chat command"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, I'm ChatGPT"
        mock_create.return_value = mock_response
        mock_prepare.return_value = []
        
        # Call chat command
        await bot.chat(self.ctx, "Tell me about Python")
        
        # Verify
        self.ctx.send.assert_called()
        mock_create.assert_called_once()

@patch('bot.HISTORY_DB', {})  # Mocking the database dictionary
class TestDatabaseFunctions(unittest.TestCase):
    
    async def test_history_functions(self):
        """Test user history saving and retrieval"""
        user_id = "12345"
        test_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        # Test saving history
        await bot.save_history(user_id, test_history)
        
        # Test retrieving history
        retrieved = await bot.get_history(user_id)
        self.assertEqual(retrieved, test_history)
        
        # Test history for new user
        new_user = "67890"
        new_history = await bot.get_history(new_user)
        self.assertEqual(len(new_history), 1)  # Should have system message
        self.assertEqual(new_history[0]["role"], "system")
    
    @patch('bot.DEFAULT_MODEL', 'gpt-4o-mini')
    async def test_user_settings(self):
        """Test user settings management"""
        user_id = "12345"
        
        # Test default settings
        model = await bot.get_user_model(user_id)
        self.assertEqual(model, "gpt-4o-mini")
        
        # Test updating settings
        await bot.set_user_model(user_id, "gpt-4o")
        updated_model = await bot.get_user_model(user_id)
        self.assertEqual(updated_model, "gpt-4o")
        
        # Test invalid model handling
        with self.assertRaises(ValueError):
            await bot.set_user_model(user_id, "invalid-model")

if __name__ == '__main__':
    unittest.main()