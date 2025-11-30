"""
Comprehensive test suite for the ChatGPT Discord Bot.

This module contains unit tests and integration tests for all major components.
Uses pytest with pytest-asyncio for async test support.
"""

import asyncio
import pytest
import os
import sys
import json
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def mock_db_handler():
    """Create a mock database handler."""
    mock = MagicMock()
    mock.get_history = AsyncMock(return_value=[])
    mock.save_history = AsyncMock()
    mock.get_user_model = AsyncMock(return_value="openai/gpt-4o")
    mock.save_user_model = AsyncMock()
    mock.is_admin = AsyncMock(return_value=False)
    mock.is_user_whitelisted = AsyncMock(return_value=True)
    mock.is_user_blacklisted = AsyncMock(return_value=False)
    mock.get_user_tool_display = AsyncMock(return_value=False)
    mock.get_user_files = AsyncMock(return_value=[])
    mock.save_token_usage = AsyncMock()
    return mock


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock = MagicMock()
    
    # Mock response structure
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    
    mock.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock


@pytest.fixture
def mock_discord_message():
    """Create a mock Discord message."""
    mock = MagicMock()
    mock.author.id = 123456789
    mock.author.name = "TestUser"
    mock.content = "Hello, bot!"
    mock.channel.send = AsyncMock()
    mock.channel.typing = MagicMock(return_value=AsyncMock().__aenter__())
    mock.attachments = []
    mock.reference = None
    mock.guild = MagicMock()
    return mock


# ============================================================
# Pricing Module Tests
# ============================================================

class TestPricingModule:
    """Tests for the pricing configuration module."""
    
    def test_model_pricing_exists(self):
        """Test that all expected models have pricing defined."""
        from src.config.pricing import MODEL_PRICING
        
        expected_models = [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-4.1",
            "openai/gpt-5",
            "openai/o1",
        ]
        
        for model in expected_models:
            assert model in MODEL_PRICING, f"Missing pricing for {model}"
    
    def test_calculate_cost(self):
        """Test cost calculation for known models."""
        from src.config.pricing import calculate_cost
        
        # GPT-4o: $5.00 input, $20.00 output per 1M tokens
        cost = calculate_cost("openai/gpt-4o", 1_000_000, 1_000_000)
        assert cost == 25.00  # $5 + $20
        
        # Test smaller amounts
        cost = calculate_cost("openai/gpt-4o", 1000, 1000)
        assert cost == pytest.approx(0.025, rel=1e-6)  # $0.005 + $0.020
    
    def test_calculate_cost_unknown_model(self):
        """Test that unknown models return 0 cost."""
        from src.config.pricing import calculate_cost
        
        cost = calculate_cost("unknown/model", 1000, 1000)
        assert cost == 0.0
    
    def test_format_cost(self):
        """Test cost formatting for display."""
        from src.config.pricing import format_cost
        
        assert format_cost(0.000001) == "$0.000001"
        assert format_cost(0.005) == "$0.005000"  # 6 decimal places for small amounts
        assert format_cost(1.50) == "$1.50"
        assert format_cost(100.00) == "$100.00"


# ============================================================
# Validator Module Tests
# ============================================================

class TestValidators:
    """Tests for input validation utilities."""
    
    def test_validate_message_content(self):
        """Test message content validation."""
        from src.utils.validators import validate_message_content
        
        # Valid content
        result = validate_message_content("Hello, world!")
        assert result.is_valid
        assert result.sanitized_value == "Hello, world!"
        
        # Empty content is valid
        result = validate_message_content("")
        assert result.is_valid
        
        # Content with null bytes should be sanitized
        result = validate_message_content("Hello\x00World")
        assert result.is_valid
        assert "\x00" not in result.sanitized_value
    
    def test_validate_message_too_long(self):
        """Test that overly long messages are rejected."""
        from src.utils.validators import validate_message_content, MAX_MESSAGE_LENGTH
        
        long_message = "x" * (MAX_MESSAGE_LENGTH + 1)
        result = validate_message_content(long_message)
        assert not result.is_valid
        assert "too long" in result.error_message.lower()
    
    def test_validate_url(self):
        """Test URL validation."""
        from src.utils.validators import validate_url
        
        # Valid URLs
        assert validate_url("https://example.com").is_valid
        assert validate_url("http://localhost:8080/path").is_valid
        assert validate_url("https://api.example.com/v1/data?q=test").is_valid
        
        # Invalid URLs
        assert not validate_url("").is_valid
        assert not validate_url("not-a-url").is_valid
        assert not validate_url("javascript:alert(1)").is_valid
        assert not validate_url("file:///etc/passwd").is_valid
    
    def test_validate_filename(self):
        """Test filename validation and sanitization."""
        from src.utils.validators import validate_filename
        
        # Valid filename
        result = validate_filename("test_file.txt")
        assert result.is_valid
        assert result.sanitized_value == "test_file.txt"
        
        # Path traversal attempt
        result = validate_filename("../../../etc/passwd")
        assert result.is_valid  # Sanitized, not rejected
        assert ".." not in result.sanitized_value
        assert "/" not in result.sanitized_value
        
        # Empty filename
        result = validate_filename("")
        assert not result.is_valid
    
    def test_sanitize_for_logging(self):
        """Test that secrets are properly redacted for logging."""
        from src.utils.validators import sanitize_for_logging
        
        # Test OpenAI key redaction
        text = "API key is sk-abcdefghijklmnopqrstuvwxyz123456"
        sanitized = sanitize_for_logging(text)
        assert "sk-" not in sanitized
        assert "[OPENAI_KEY]" in sanitized
        
        # Test MongoDB URI redaction
        text = "mongodb+srv://user:password@cluster.mongodb.net/db"
        sanitized = sanitize_for_logging(text)
        assert "password" not in sanitized
        assert "[REDACTED]" in sanitized
        
        # Test truncation
        long_text = "x" * 500
        sanitized = sanitize_for_logging(long_text, max_length=100)
        assert len(sanitized) < 150  # Account for truncation marker


# ============================================================
# Retry Module Tests
# ============================================================

class TestRetryModule:
    """Tests for retry utilities."""
    
    @pytest.mark.asyncio
    async def test_retry_success_first_try(self):
        """Test that successful functions don't retry."""
        from src.utils.retry import async_retry_with_backoff
        
        call_count = 0
        
        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await async_retry_with_backoff(success_func, max_retries=3)
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_eventual_success(self):
        """Test that functions eventually succeed after retries."""
        from src.utils.retry import async_retry_with_backoff
        
        call_count = 0
        
        async def eventual_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await async_retry_with_backoff(
            eventual_success,
            max_retries=5,
            base_delay=0.01,  # Fast for testing
            retryable_exceptions=(ConnectionError,)
        )
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test that RetryError is raised when retries are exhausted."""
        from src.utils.retry import async_retry_with_backoff, RetryError
        
        async def always_fail():
            raise ConnectionError("Always fails")
        
        with pytest.raises(RetryError):
            await async_retry_with_backoff(
                always_fail,
                max_retries=2,
                base_delay=0.01,
                retryable_exceptions=(ConnectionError,)
            )


# ============================================================
# Discord Utils Tests
# ============================================================

class TestDiscordUtils:
    """Tests for Discord utility functions."""
    
    def test_split_message_short(self):
        """Test that short messages aren't split."""
        from src.utils.discord_utils import split_message
        
        short = "This is a short message."
        chunks = split_message(short)
        assert len(chunks) == 1
        assert chunks[0] == short
    
    def test_split_message_long(self):
        """Test that long messages are properly split."""
        from src.utils.discord_utils import split_message
        
        # Create a message longer than 2000 characters
        long = "Hello world. " * 200
        chunks = split_message(long, max_length=2000)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 2000
    
    def test_split_code_block(self):
        """Test code block splitting."""
        from src.utils.discord_utils import split_code_block
        
        code = "\n".join([f"line {i}" for i in range(100)])
        chunks = split_code_block(code, "python", max_length=500)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.startswith("```python\n")
            assert chunk.endswith("\n```")
            assert len(chunk) <= 500
    
    def test_create_error_embed(self):
        """Test error embed creation."""
        from src.utils.discord_utils import create_error_embed
        import discord
        
        embed = create_error_embed("Test Error", "Something went wrong", "ValidationError")
        
        assert isinstance(embed, discord.Embed)
        assert "Test Error" in embed.title
        assert embed.color == discord.Color.red()
    
    def test_create_success_embed(self):
        """Test success embed creation."""
        from src.utils.discord_utils import create_success_embed
        import discord
        
        embed = create_success_embed("Success!", "Operation completed")
        
        assert isinstance(embed, discord.Embed)
        assert "Success!" in embed.title
        assert embed.color == discord.Color.green()


# ============================================================
# Code Interpreter Security Tests
# ============================================================

class TestCodeInterpreterSecurity:
    """Tests for code interpreter security features."""
    
    def test_blocked_imports(self):
        """Test that dangerous imports are blocked."""
        from src.utils.code_interpreter import BLOCKED_PATTERNS
        import re
        
        dangerous_code = [
            "import os",
            "import subprocess",
            "from os import system",
            "import socket",
            "import requests",
            "__import__('os')",
            "eval('print(1)')",
            "exec('import os')",
        ]
        
        for code in dangerous_code:
            blocked = any(
                re.search(pattern, code, re.IGNORECASE)
                for pattern in BLOCKED_PATTERNS
            )
            assert blocked, f"Should block: {code}"
    
    def test_allowed_imports(self):
        """Test that safe imports are allowed."""
        from src.utils.code_interpreter import BLOCKED_PATTERNS
        import re
        
        safe_code = [
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "from sklearn.model_selection import train_test_split",
            "import os.path",  # os.path is allowed
        ]
        
        for code in safe_code:
            blocked = any(
                re.search(pattern, code, re.IGNORECASE)
                for pattern in BLOCKED_PATTERNS
            )
            assert not blocked, f"Should allow: {code}"
    
    def test_file_type_detection(self):
        """Test file type detection for various extensions."""
        from src.utils.code_interpreter import FileManager
        
        fm = FileManager()
        
        assert fm._detect_file_type("data.csv") == "csv"
        assert fm._detect_file_type("data.xlsx") == "excel"
        assert fm._detect_file_type("config.json") == "json"
        assert fm._detect_file_type("image.png") == "image"
        assert fm._detect_file_type("script.py") == "python"
        assert fm._detect_file_type("unknown.xyz") == "binary"


# ============================================================
# OpenAI Utils Tests
# ============================================================

class TestOpenAIUtils:
    """Tests for OpenAI utility functions."""
    
    def test_count_tokens(self):
        """Test token counting function."""
        from src.utils.openai_utils import count_tokens
        
        text = "Hello, world!"
        tokens = count_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_trim_content_to_token_limit(self):
        """Test content trimming."""
        from src.utils.openai_utils import trim_content_to_token_limit
        
        # Short content should not be trimmed
        short = "Hello, world!"
        trimmed = trim_content_to_token_limit(short, max_tokens=100)
        assert trimmed == short
        
        # Long content should be trimmed
        long = "Hello " * 10000
        trimmed = trim_content_to_token_limit(long, max_tokens=100)
        assert len(trimmed) < len(long)
    
    def test_prepare_messages_for_api(self):
        """Test message preparation for API."""
        from src.utils.openai_utils import prepare_messages_for_api
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        
        prepared = prepare_messages_for_api(messages)
        
        assert len(prepared) == 3
        assert all(m.get("role") in ["user", "assistant", "system"] for m in prepared)
    
    def test_prepare_messages_filters_none_content(self):
        """Test that messages with None content are filtered."""
        from src.utils.openai_utils import prepare_messages_for_api
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "World"},
        ]
        
        prepared = prepare_messages_for_api(messages)
        
        assert len(prepared) == 2


# ============================================================
# Database Handler Tests (with mocking)
# ============================================================

class TestDatabaseHandlerMocked:
    """Tests for database handler using mocks."""
    
    def test_filter_expired_images_no_images(self):
        """Test that messages without images pass through unchanged."""
        from src.database.db_handler import DatabaseHandler
        
        with patch('motor.motor_asyncio.AsyncIOMotorClient'):
            handler = DatabaseHandler("mongodb://localhost")
            
            history = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            
            filtered = handler._filter_expired_images(history)
            assert len(filtered) == 2
            assert filtered[0]["content"] == "Hello"
    
    def test_filter_expired_images_recent_image(self):
        """Test that recent images are kept."""
        from src.database.db_handler import DatabaseHandler
        
        with patch('motor.motor_asyncio.AsyncIOMotorClient'):
            handler = DatabaseHandler("mongodb://localhost")
            
            recent_timestamp = datetime.now().isoformat()
            history = [
                {"role": "user", "content": [
                    {"type": "text", "text": "Check this image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}, "timestamp": recent_timestamp}
                ]}
            ]
            
            filtered = handler._filter_expired_images(history)
            assert len(filtered) == 1
            assert len(filtered[0]["content"]) == 2  # Both items kept
    
    def test_filter_expired_images_old_image(self):
        """Test that old images are filtered out."""
        from src.database.db_handler import DatabaseHandler
        
        with patch('motor.motor_asyncio.AsyncIOMotorClient'):
            handler = DatabaseHandler("mongodb://localhost")
            
            old_timestamp = (datetime.now() - timedelta(hours=24)).isoformat()
            history = [
                {"role": "user", "content": [
                    {"type": "text", "text": "Check this image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}, "timestamp": old_timestamp}
                ]}
            ]
            
            filtered = handler._filter_expired_images(history)
            assert len(filtered) == 1
            assert len(filtered[0]["content"]) == 1  # Only text kept


# ============================================================
# ============================================================
# Cache Module Tests
# ============================================================

class TestLRUCache:
    """Tests for the LRU cache implementation."""
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        from src.utils.cache import LRUCache
        
        cache = LRUCache(max_size=100, default_ttl=60.0)
        
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test that cache entries expire after TTL."""
        from src.utils.cache import LRUCache
        
        cache = LRUCache(max_size=100, default_ttl=0.1)  # 100ms TTL
        
        await cache.set("key1", "value1")
        
        # Should exist immediately
        assert await cache.get("key1") == "value1"
        
        # Wait for expiration
        await asyncio.sleep(0.15)
        
        # Should be expired now
        assert await cache.get("key1") is None
    
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test that LRU eviction works correctly."""
        from src.utils.cache import LRUCache
        
        cache = LRUCache(max_size=3, default_ttl=60.0)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        await cache.get("key1")
        
        # Add new key, should evict key2 (least recently used)
        await cache.set("key4", "value4")
        
        assert await cache.get("key1") == "value1"  # Should exist
        assert await cache.get("key2") is None  # Should be evicted
        assert await cache.get("key3") == "value3"  # Should exist
        assert await cache.get("key4") == "value4"  # Should exist
    
    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics tracking."""
        from src.utils.cache import LRUCache
        
        cache = LRUCache(max_size=100, default_ttl=60.0)
        
        await cache.set("key1", "value1")
        await cache.get("key1")  # Hit
        await cache.get("key2")  # Miss
        await cache.get("key1")  # Hit
        
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1
    
    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test cache clearing."""
        from src.utils.cache import LRUCache
        
        cache = LRUCache(max_size=100, default_ttl=60.0)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        cleared = await cache.clear()
        assert cleared == 2
        
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None


# ============================================================
# Monitoring Module Tests
# ============================================================

class TestMonitoring:
    """Tests for the monitoring utilities."""
    
    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        from src.utils.monitoring import PerformanceMetrics
        import time
        
        metrics = PerformanceMetrics(name="test_operation")
        time.sleep(0.01)  # Small delay
        metrics.finish(success=True)
        
        assert metrics.success
        assert metrics.duration_ms > 0
        assert metrics.duration_ms < 1000  # Should be fast
    
    def test_measure_sync_context_manager(self):
        """Test synchronous measurement context manager."""
        from src.utils.monitoring import measure_sync
        import time
        
        with measure_sync("test_op", custom_field="value") as metrics:
            time.sleep(0.01)
        
        assert metrics.duration_ms > 0
        assert metrics.metadata["custom_field"] == "value"
    
    @pytest.mark.asyncio
    async def test_measure_async_context_manager(self):
        """Test async measurement context manager."""
        from src.utils.monitoring import measure_async
        
        async with measure_async("async_op") as metrics:
            await asyncio.sleep(0.01)
        
        assert metrics.duration_ms > 0
        assert metrics.success
    
    @pytest.mark.asyncio
    async def test_track_performance_decorator(self):
        """Test performance tracking decorator."""
        from src.utils.monitoring import track_performance
        
        call_count = 0
        
        @track_performance("tracked_function")
        async def tracked_func():
            nonlocal call_count
            call_count += 1
            return "result"
        
        result = await tracked_func()
        assert result == "result"
        assert call_count == 1
    
    def test_health_status(self):
        """Test health status structure."""
        from src.utils.monitoring import HealthStatus
        
        status = HealthStatus(healthy=True)
        
        status.add_check("database", True, "Connected")
        status.add_check("api", False, "Timeout")
        
        assert not status.healthy  # Should be unhealthy due to API check
        assert status.checks["database"]["healthy"]
        assert not status.checks["api"]["healthy"]


# ============================================================
# Integration Tests (require environment setup)
# ============================================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests that require actual services."""
    
    @pytest.mark.asyncio
    async def test_database_connection(self):
        """Test actual database connection (skip if no MongoDB)."""
        from dotenv import load_dotenv
        load_dotenv()
        
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            pytest.skip("MONGODB_URI not set")
        
        from src.database.db_handler import DatabaseHandler
        handler = DatabaseHandler(mongodb_uri)
        
        connected = await handler.ensure_connected()
        assert connected
        
        await handler.close()


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
