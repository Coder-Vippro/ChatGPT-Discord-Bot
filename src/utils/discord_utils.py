"""
Discord response utilities for sending messages with proper handling.

This module provides utilities for sending messages to Discord with
proper length handling, error recovery, and formatting.
"""

import discord
import asyncio
import logging
import io
from typing import Optional, List, Union
from dataclasses import dataclass


# Discord message limits
MAX_MESSAGE_LENGTH = 2000
MAX_EMBED_DESCRIPTION = 4096
MAX_EMBED_FIELD_VALUE = 1024
MAX_EMBED_FIELDS = 25
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB for non-nitro


@dataclass
class MessageChunk:
    """A chunk of a message that fits within Discord limits."""
    content: str
    is_code_block: bool = False
    language: Optional[str] = None


def split_message(
    content: str,
    max_length: int = MAX_MESSAGE_LENGTH,
    split_on: List[str] = None
) -> List[str]:
    """
    Split a long message into chunks that fit within Discord limits.
    
    Args:
        content: The message content to split
        max_length: Maximum length per chunk
        split_on: Preferred split points (default: newlines, spaces)
        
    Returns:
        List of message chunks
    """
    if len(content) <= max_length:
        return [content]
    
    if split_on is None:
        split_on = ['\n\n', '\n', '. ', ' ']
    
    chunks = []
    remaining = content
    
    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break
        
        # Find the best split point
        split_index = max_length
        
        for delimiter in split_on:
            # Look for delimiter before max_length
            last_index = remaining.rfind(delimiter, 0, max_length)
            if last_index > max_length // 2:  # Don't split too early
                split_index = last_index + len(delimiter)
                break
        
        # If no good split point, hard cut at max_length
        if split_index >= max_length:
            split_index = max_length
        
        chunks.append(remaining[:split_index])
        remaining = remaining[split_index:]
    
    return chunks


def split_code_block(
    code: str,
    language: str = "",
    max_length: int = MAX_MESSAGE_LENGTH
) -> List[str]:
    """
    Split code into properly formatted code block chunks.
    
    Args:
        code: The code content
        language: The language for syntax highlighting
        max_length: Maximum length per chunk
        
    Returns:
        List of formatted code block strings
    """
    # Account for code block markers
    marker_length = len(f"```{language}\n") + len("```")
    effective_max = max_length - marker_length - 20  # Extra buffer
    
    lines = code.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line_length = len(line) + 1  # +1 for newline
        
        if current_length + line_length > effective_max and current_chunk:
            # Finish current chunk
            chunk_code = '\n'.join(current_chunk)
            chunks.append(f"```{language}\n{chunk_code}\n```")
            current_chunk = [line]
            current_length = line_length
        else:
            current_chunk.append(line)
            current_length += line_length
    
    # Add remaining chunk
    if current_chunk:
        chunk_code = '\n'.join(current_chunk)
        chunks.append(f"```{language}\n{chunk_code}\n```")
    
    return chunks


async def send_long_message(
    channel: discord.abc.Messageable,
    content: str,
    max_length: int = MAX_MESSAGE_LENGTH,
    delay: float = 0.5
) -> List[discord.Message]:
    """
    Send a long message split across multiple Discord messages.
    
    Args:
        channel: The channel to send to
        content: The message content
        max_length: Maximum length per message
        delay: Delay between messages to avoid rate limiting
        
    Returns:
        List of sent messages
    """
    chunks = split_message(content, max_length)
    messages = []
    
    for i, chunk in enumerate(chunks):
        try:
            msg = await channel.send(chunk)
            messages.append(msg)
            
            # Add delay between messages (except for the last one)
            if i < len(chunks) - 1:
                await asyncio.sleep(delay)
                
        except discord.HTTPException as e:
            logging.error(f"Failed to send message chunk {i+1}: {e}")
            # Try sending as file if message still too long
            if "too long" in str(e).lower():
                file = discord.File(
                    io.StringIO(chunk),
                    filename=f"message_part_{i+1}.txt"
                )
                msg = await channel.send(file=file)
                messages.append(msg)
    
    return messages


async def send_code_response(
    channel: discord.abc.Messageable,
    code: str,
    language: str = "python",
    title: Optional[str] = None
) -> List[discord.Message]:
    """
    Send code with proper formatting, handling long code.
    
    Args:
        channel: The channel to send to
        code: The code content
        language: Programming language for highlighting
        title: Optional title to display before code
        
    Returns:
        List of sent messages
    """
    messages = []
    
    if title:
        msg = await channel.send(title)
        messages.append(msg)
    
    # If code is too long for code blocks, send as file
    if len(code) > MAX_MESSAGE_LENGTH - 100:
        file = discord.File(
            io.StringIO(code),
            filename=f"code.{language}" if language else "code.txt"
        )
        msg = await channel.send("📎 Code attached as file:", file=file)
        messages.append(msg)
    else:
        chunks = split_code_block(code, language)
        for chunk in chunks:
            msg = await channel.send(chunk)
            messages.append(msg)
            await asyncio.sleep(0.3)
    
    return messages


def create_error_embed(
    title: str,
    description: str,
    error_type: str = "Error"
) -> discord.Embed:
    """
    Create a standardized error embed.
    
    Args:
        title: Error title
        description: Error description
        error_type: Type of error for categorization
        
    Returns:
        Discord Embed object
    """
    embed = discord.Embed(
        title=f"❌ {title}",
        description=description[:MAX_EMBED_DESCRIPTION],
        color=discord.Color.red()
    )
    embed.set_footer(text=f"Error Type: {error_type}")
    return embed


def create_success_embed(
    title: str,
    description: str = ""
) -> discord.Embed:
    """
    Create a standardized success embed.
    
    Args:
        title: Success title
        description: Success description
        
    Returns:
        Discord Embed object
    """
    embed = discord.Embed(
        title=f"✅ {title}",
        description=description[:MAX_EMBED_DESCRIPTION] if description else None,
        color=discord.Color.green()
    )
    return embed


def create_info_embed(
    title: str,
    description: str = "",
    fields: List[tuple] = None
) -> discord.Embed:
    """
    Create a standardized info embed with optional fields.
    
    Args:
        title: Info title
        description: Info description
        fields: List of (name, value, inline) tuples
        
    Returns:
        Discord Embed object
    """
    embed = discord.Embed(
        title=f"ℹ️ {title}",
        description=description[:MAX_EMBED_DESCRIPTION] if description else None,
        color=discord.Color.blue()
    )
    
    if fields:
        for name, value, inline in fields[:MAX_EMBED_FIELDS]:
            embed.add_field(
                name=name[:256],
                value=str(value)[:MAX_EMBED_FIELD_VALUE],
                inline=inline
            )
    
    return embed


def create_progress_embed(
    title: str,
    description: str,
    progress: float = 0.0
) -> discord.Embed:
    """
    Create a progress indicator embed.
    
    Args:
        title: Progress title
        description: Progress description
        progress: Progress value 0.0 to 1.0
        
    Returns:
        Discord Embed object
    """
    # Create progress bar
    bar_length = 20
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)
    percentage = int(progress * 100)
    
    embed = discord.Embed(
        title=f"⏳ {title}",
        description=f"{description}\n\n`{bar}` {percentage}%",
        color=discord.Color.orange()
    )
    return embed


async def edit_or_send(
    message: Optional[discord.Message],
    channel: discord.abc.Messageable,
    content: str = None,
    embed: discord.Embed = None
) -> discord.Message:
    """
    Edit an existing message or send a new one if editing fails.
    
    Args:
        message: Message to edit (or None to send new)
        channel: Channel to send to if message is None
        content: Message content
        embed: Message embed
        
    Returns:
        The edited or new message
    """
    try:
        if message:
            await message.edit(content=content, embed=embed)
            return message
        else:
            return await channel.send(content=content, embed=embed)
    except discord.HTTPException:
        return await channel.send(content=content, embed=embed)


class ProgressMessage:
    """
    A message that can be updated to show progress.
    
    Usage:
        async with ProgressMessage(channel, "Processing") as progress:
            for i in range(100):
                await progress.update(i / 100, f"Step {i}")
    """
    
    def __init__(
        self,
        channel: discord.abc.Messageable,
        title: str,
        description: str = "Starting..."
    ):
        self.channel = channel
        self.title = title
        self.description = description
        self.message: Optional[discord.Message] = None
        self._last_update = 0.0
        self._update_interval = 2.0  # Minimum seconds between updates
    
    async def __aenter__(self):
        embed = create_progress_embed(self.title, self.description, 0.0)
        self.message = await self.channel.send(embed=embed)
        return self
    
    async def __aexit__(self, *args):
        # Clean up or finalize
        pass
    
    async def update(self, progress: float, description: str = None):
        """Update the progress message."""
        import time
        
        now = time.monotonic()
        if now - self._last_update < self._update_interval:
            return
        
        self._last_update = now
        
        if description:
            self.description = description
        
        try:
            embed = create_progress_embed(self.title, self.description, progress)
            await self.message.edit(embed=embed)
        except discord.HTTPException:
            pass  # Ignore edit failures
    
    async def complete(self, message: str = "Complete!"):
        """Mark the progress as complete."""
        try:
            embed = create_success_embed(self.title, message)
            await self.message.edit(embed=embed)
        except discord.HTTPException:
            pass
    
    async def error(self, message: str):
        """Mark the progress as failed."""
        try:
            embed = create_error_embed(self.title, message)
            await self.message.edit(embed=embed)
        except discord.HTTPException:
            pass
