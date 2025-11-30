"""
Input validation utilities for the Discord bot.

This module provides centralized validation for user inputs,
enhancing security and reducing code duplication.
"""

import re
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass


# Maximum allowed lengths for various inputs
MAX_MESSAGE_LENGTH = 4000  # Discord's limit is 2000, but we process longer
MAX_PROMPT_LENGTH = 32000  # Reasonable limit for AI prompts
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_FILENAME_LENGTH = 255
MAX_URL_LENGTH = 2048
MAX_CODE_LENGTH = 100000  # 100KB of code


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    error_message: Optional[str] = None
    sanitized_value: Optional[str] = None


def validate_message_content(content: str) -> ValidationResult:
    """
    Validate and sanitize message content.
    
    Args:
        content: The message content to validate
        
    Returns:
        ValidationResult with validation status and sanitized content
    """
    if not content:
        return ValidationResult(is_valid=True, sanitized_value="")
    
    if len(content) > MAX_MESSAGE_LENGTH:
        return ValidationResult(
            is_valid=False,
            error_message=f"Message too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed."
        )
    
    # Remove null bytes and other control characters (except newlines/tabs)
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)
    
    return ValidationResult(is_valid=True, sanitized_value=sanitized)


def validate_prompt(prompt: str) -> ValidationResult:
    """
    Validate AI prompt content.
    
    Args:
        prompt: The prompt to validate
        
    Returns:
        ValidationResult with validation status
    """
    if not prompt or not prompt.strip():
        return ValidationResult(
            is_valid=False,
            error_message="Prompt cannot be empty."
        )
    
    if len(prompt) > MAX_PROMPT_LENGTH:
        return ValidationResult(
            is_valid=False,
            error_message=f"Prompt too long. Maximum {MAX_PROMPT_LENGTH} characters allowed."
        )
    
    # Remove null bytes
    sanitized = prompt.replace('\x00', '')
    
    return ValidationResult(is_valid=True, sanitized_value=sanitized)


def validate_url(url: str) -> ValidationResult:
    """
    Validate and sanitize a URL.
    
    Args:
        url: The URL to validate
        
    Returns:
        ValidationResult with validation status
    """
    if not url:
        return ValidationResult(
            is_valid=False,
            error_message="URL cannot be empty."
        )
    
    if len(url) > MAX_URL_LENGTH:
        return ValidationResult(
            is_valid=False,
            error_message=f"URL too long. Maximum {MAX_URL_LENGTH} characters allowed."
        )
    
    # Basic URL pattern check
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    
    if not url_pattern.match(url):
        return ValidationResult(
            is_valid=False,
            error_message="Invalid URL format."
        )
    
    # Check for potentially dangerous URL schemes
    dangerous_schemes = ['javascript:', 'data:', 'file:', 'vbscript:']
    url_lower = url.lower()
    for scheme in dangerous_schemes:
        if scheme in url_lower:
            return ValidationResult(
                is_valid=False,
                error_message="URL contains potentially dangerous content."
            )
    
    return ValidationResult(is_valid=True, sanitized_value=url)


def validate_filename(filename: str) -> ValidationResult:
    """
    Validate and sanitize a filename.
    
    Args:
        filename: The filename to validate
        
    Returns:
        ValidationResult with validation status and sanitized filename
    """
    if not filename:
        return ValidationResult(
            is_valid=False,
            error_message="Filename cannot be empty."
        )
    
    if len(filename) > MAX_FILENAME_LENGTH:
        return ValidationResult(
            is_valid=False,
            error_message=f"Filename too long. Maximum {MAX_FILENAME_LENGTH} characters allowed."
        )
    
    # Remove path traversal attempts
    sanitized = filename.replace('..', '').replace('/', '').replace('\\', '')
    
    # Remove dangerous characters
    sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', '', sanitized)
    
    # Ensure it's not empty after sanitization
    if not sanitized:
        return ValidationResult(
            is_valid=False,
            error_message="Filename contains only invalid characters."
        )
    
    return ValidationResult(is_valid=True, sanitized_value=sanitized)


def validate_file_size(size: int) -> ValidationResult:
    """
    Validate file size.
    
    Args:
        size: The file size in bytes
        
    Returns:
        ValidationResult with validation status
    """
    if size <= 0:
        return ValidationResult(
            is_valid=False,
            error_message="File size must be greater than 0."
        )
    
    if size > MAX_FILE_SIZE:
        max_mb = MAX_FILE_SIZE / (1024 * 1024)
        return ValidationResult(
            is_valid=False,
            error_message=f"File too large. Maximum {max_mb:.0f}MB allowed."
        )
    
    return ValidationResult(is_valid=True)


def validate_code(code: str) -> ValidationResult:
    """
    Validate code for execution.
    
    Args:
        code: The code to validate
        
    Returns:
        ValidationResult with validation status
    """
    if not code or not code.strip():
        return ValidationResult(
            is_valid=False,
            error_message="Code cannot be empty."
        )
    
    if len(code) > MAX_CODE_LENGTH:
        return ValidationResult(
            is_valid=False,
            error_message=f"Code too long. Maximum {MAX_CODE_LENGTH} characters allowed."
        )
    
    return ValidationResult(is_valid=True, sanitized_value=code)


def validate_user_id(user_id) -> ValidationResult:
    """
    Validate a Discord user ID.
    
    Args:
        user_id: The user ID to validate
        
    Returns:
        ValidationResult with validation status
    """
    try:
        uid = int(user_id)
        if uid <= 0:
            return ValidationResult(
                is_valid=False,
                error_message="Invalid user ID."
            )
        # Discord IDs are 17-19 digits
        if len(str(uid)) < 17 or len(str(uid)) > 19:
            return ValidationResult(
                is_valid=False,
                error_message="Invalid user ID format."
            )
        return ValidationResult(is_valid=True)
    except (ValueError, TypeError):
        return ValidationResult(
            is_valid=False,
            error_message="User ID must be a valid integer."
        )


def sanitize_for_logging(text: str, max_length: int = 200) -> str:
    """
    Sanitize text for safe logging (remove sensitive data, truncate).
    
    Args:
        text: The text to sanitize
        max_length: Maximum length for logged text
        
    Returns:
        Sanitized text safe for logging
    """
    if not text:
        return ""
    
    # Remove potential secrets/tokens (common patterns)
    patterns = [
        (r'(sk-[a-zA-Z0-9]{20,})', '[OPENAI_KEY]'),
        (r'(xoxb-[a-zA-Z0-9-]+)', '[SLACK_TOKEN]'),
        (r'([A-Za-z0-9_-]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27})', '[DISCORD_TOKEN]'),
        (r'(mongodb\+srv://[^@]+@)', 'mongodb+srv://[REDACTED]@'),
        (r'(Bearer\s+[A-Za-z0-9_-]+)', 'Bearer [TOKEN]'),
        (r'(password["\']?\s*[:=]\s*["\']?)[^"\'\s]+', r'\1[REDACTED]'),
    ]
    
    sanitized = text
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    
    # Truncate if needed
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + '...[truncated]'
    
    return sanitized
