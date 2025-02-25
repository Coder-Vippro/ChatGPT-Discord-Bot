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
        

if __name__ == '__main__':
    unittest.main()