"""
Claude (Anthropic) API utility functions.

This module provides utilities for interacting with Anthropic's Claude models,
including message conversion and API calls compatible with the existing bot structure.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple


def is_claude_model(model: str) -> bool:
    """
    Check if the model is a Claude/Anthropic model.
    
    Args:
        model: Model name (e.g., "anthropic/claude-sonnet-4-20250514")
        
    Returns:
        True if it's a Claude model, False otherwise
    """
    return model.startswith("anthropic/")


def get_claude_model_id(model: str) -> str:
    """
    Extract the Claude model ID from the full model name.
    
    Args:
        model: Full model name (e.g., "anthropic/claude-sonnet-4-20250514")
        
    Returns:
        Claude model ID (e.g., "claude-sonnet-4-20250514")
    """
    if model.startswith("anthropic/"):
        return model[len("anthropic/"):]
    return model


def convert_openai_messages_to_claude(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Convert OpenAI message format to Claude message format.
    
    OpenAI uses:
    - {"role": "system", "content": "..."}
    - {"role": "user", "content": "..."}
    - {"role": "assistant", "content": "..."}
    
    Claude uses:
    - system parameter (separate from messages)
    - {"role": "user", "content": "..."}
    - {"role": "assistant", "content": "..."}
    
    Args:
        messages: List of messages in OpenAI format
        
    Returns:
        Tuple of (system_prompt, claude_messages)
    """
    system_prompt = None
    claude_messages = []
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        
        # Skip messages with None content
        if content is None:
            continue
        
        if role == "system":
            # Claude uses a separate system parameter
            if isinstance(content, str):
                system_prompt = content
            elif isinstance(content, list):
                # Extract text from list content
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                system_prompt = " ".join(text_parts)
        elif role in ["user", "assistant"]:
            # Convert content format
            converted_content = convert_content_to_claude(content)
            if converted_content:
                claude_messages.append({
                    "role": role,
                    "content": converted_content
                })
        elif role == "tool":
            # Claude handles tool results differently - add as user message with tool result
            tool_call_id = msg.get("tool_call_id", "")
            tool_name = msg.get("name", "unknown")
            claude_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": str(content)
                    }
                ]
            })
    
    # Claude requires alternating user/assistant messages
    # Merge consecutive messages of the same role
    merged_messages = merge_consecutive_messages(claude_messages)
    
    return system_prompt, merged_messages


def convert_content_to_claude(content: Any) -> Any:
    """
    Convert content from OpenAI format to Claude format.
    
    Args:
        content: Content in OpenAI format (string or list)
        
    Returns:
        Content in Claude format
    """
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        claude_content = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                
                if item_type == "text":
                    claude_content.append({
                        "type": "text",
                        "text": item.get("text", "")
                    })
                elif item_type == "image_url":
                    # Convert image_url format to Claude format
                    image_url_data = item.get("image_url", {})
                    if isinstance(image_url_data, dict):
                        url = image_url_data.get("url", "")
                    else:
                        url = str(image_url_data)
                    
                    if url:
                        # Claude requires base64 data or URLs
                        if url.startswith("data:"):
                            # Parse base64 data URL
                            try:
                                media_type, base64_data = parse_data_url(url)
                                claude_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_data
                                    }
                                })
                            except Exception as e:
                                logging.warning(f"Failed to parse data URL: {e}")
                        else:
                            # Regular URL - Claude supports URLs directly
                            claude_content.append({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": url
                                }
                            })
                else:
                    # Handle other types as text
                    if "text" in item:
                        claude_content.append({
                            "type": "text",
                            "text": str(item.get("text", ""))
                        })
            elif isinstance(item, str):
                claude_content.append({
                    "type": "text",
                    "text": item
                })
        
        return claude_content if claude_content else None
    
    return str(content) if content else None


def parse_data_url(data_url: str) -> Tuple[str, str]:
    """
    Parse a data URL into media type and base64 data.
    
    Args:
        data_url: Data URL (e.g., "data:image/png;base64,...")
        
    Returns:
        Tuple of (media_type, base64_data)
    """
    if not data_url.startswith("data:"):
        raise ValueError("Not a data URL")
    
    # Remove "data:" prefix
    content = data_url[5:]
    
    # Split by semicolon and comma
    parts = content.split(";base64,")
    if len(parts) != 2:
        raise ValueError("Invalid data URL format")
    
    media_type = parts[0]
    base64_data = parts[1]
    
    return media_type, base64_data


def merge_consecutive_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive messages with the same role.
    Claude requires alternating user/assistant messages.
    
    Args:
        messages: List of messages
        
    Returns:
        List of merged messages
    """
    if not messages:
        return []
    
    merged = []
    current_role = None
    current_content = []
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        
        if role == current_role:
            # Same role, merge content
            if isinstance(content, str):
                if current_content and isinstance(current_content[-1], dict) and current_content[-1].get("type") == "text":
                    current_content[-1]["text"] += "\n" + content
                else:
                    current_content.append({"type": "text", "text": content})
            elif isinstance(content, list):
                current_content.extend(content)
        else:
            # Different role, save previous and start new
            if current_role is not None and current_content:
                merged.append({
                    "role": current_role,
                    "content": simplify_content(current_content)
                })
            
            current_role = role
            if isinstance(content, str):
                current_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                current_content = content.copy()
            else:
                current_content = []
    
    # Don't forget the last message
    if current_role is not None and current_content:
        merged.append({
            "role": current_role,
            "content": simplify_content(current_content)
        })
    
    return merged


def simplify_content(content: List[Dict[str, Any]]) -> Any:
    """
    Simplify content list to string if it only contains text.
    
    Args:
        content: List of content items
        
    Returns:
        Simplified content (string or list)
    """
    if not content:
        return ""
    
    # If only one text item, return as string
    if len(content) == 1 and content[0].get("type") == "text":
        return content[0].get("text", "")
    
    # If all items are text, merge them
    if all(item.get("type") == "text" for item in content):
        texts = [item.get("text", "") for item in content]
        return "\n".join(texts)
    
    return content


def get_claude_tools() -> List[Dict[str, Any]]:
    """
    Get tool definitions for Claude API.
    Claude uses a slightly different tool format than OpenAI.
    
    Returns:
        List of tool definitions in Claude format
    """
    return [
        {
            "name": "google_search",
            "description": "Search the web for current information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "num_results": {"type": "integer", "description": "Number of results (max 10)", "maximum": 10}
                },
                "required": ["query"]
            }
        },
        {
            "name": "scrape_webpage",
            "description": "Extract and read content from a webpage URL",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The webpage URL to scrape"}
                },
                "required": ["url"]
            }
        },
        {
            "name": "execute_python_code",
            "description": "Run Python code. Packages auto-install. Use load_file('file_id') for user files. Output files auto-sent to user.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "maximum": 300}
                },
                "required": ["code"]
            }
        },
        {
            "name": "generate_image",
            "description": "Create/generate images from text. Models: flux (best), flux-dev, sdxl, realistic (photos), anime, dreamshaper.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Detailed description of the image to create"},
                    "model": {"type": "string", "description": "Model to use", "enum": ["flux", "flux-dev", "sdxl", "realistic", "anime", "dreamshaper"]},
                    "num_images": {"type": "integer", "description": "Number of images (1-4)", "maximum": 4},
                    "aspect_ratio": {"type": "string", "description": "Aspect ratio preset", "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"]}
                },
                "required": ["prompt"]
            }
        },
        {
            "name": "set_reminder",
            "description": "Set a reminder",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Reminder content"},
                    "time": {"type": "string", "description": "Reminder time"}
                },
                "required": ["content", "time"]
            }
        },
        {
            "name": "get_reminders",
            "description": "List all reminders",
            "input_schema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "upscale_image",
            "description": "Enlarge/upscale an image to higher resolution. Pass 'latest_image' to use the user's most recently uploaded image.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "Pass 'latest_image' to use the user's most recently uploaded image"},
                    "scale_factor": {"type": "integer", "description": "Scale factor (2 or 4)", "enum": [2, 4]},
                    "model": {"type": "string", "description": "Upscale model", "enum": ["clarity", "ccsr", "sd-latent", "swinir"]}
                },
                "required": ["image_url"]
            }
        },
        {
            "name": "remove_background",
            "description": "Remove background from an image. Pass 'latest_image' to use the user's most recently uploaded image.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "Pass 'latest_image' to use the user's most recently uploaded image"},
                    "model": {"type": "string", "description": "Background removal model", "enum": ["bria", "rembg", "birefnet-base", "birefnet-general", "birefnet-portrait"]}
                },
                "required": ["image_url"]
            }
        },
        {
            "name": "image_to_text",
            "description": "Generate a text description/caption of an image. Pass 'latest_image' to use the user's most recently uploaded image.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "Pass 'latest_image' to use the user's most recently uploaded image"}
                },
                "required": ["image_url"]
            }
        }
    ]


async def call_claude_api(
    anthropic_client,
    messages: List[Dict[str, Any]],
    model: str,
    max_tokens: int = 4096,
    use_tools: bool = True
) -> Dict[str, Any]:
    """
    Call the Claude API with the given messages.
    
    Args:
        anthropic_client: Anthropic client instance
        messages: List of messages in OpenAI format
        model: Model name (e.g., "anthropic/claude-sonnet-4-20250514")
        max_tokens: Maximum tokens in response
        use_tools: Whether to include tools
        
    Returns:
        Dict with response data including:
        - content: Response text
        - input_tokens: Number of input tokens
        - output_tokens: Number of output tokens
        - tool_calls: Any tool calls made
        - stop_reason: Why the response stopped
    """
    try:
        # Convert messages
        system_prompt, claude_messages = convert_openai_messages_to_claude(messages)
        
        # Get Claude model ID
        model_id = get_claude_model_id(model)
        
        # Build API parameters
        api_params = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": claude_messages
        }
        
        if system_prompt:
            api_params["system"] = system_prompt
        
        if use_tools:
            api_params["tools"] = get_claude_tools()
        
        # Make API call
        response = await anthropic_client.messages.create(**api_params)
        
        # Extract response data
        result = {
            "content": "",
            "input_tokens": response.usage.input_tokens if response.usage else 0,
            "output_tokens": response.usage.output_tokens if response.usage else 0,
            "tool_calls": [],
            "stop_reason": response.stop_reason
        }
        
        # Process content blocks
        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
        
        return result
        
    except Exception as e:
        logging.error(f"Error calling Claude API: {e}")
        raise


def convert_claude_tool_calls_to_openai(tool_calls: List[Dict[str, Any]]) -> List[Any]:
    """
    Convert Claude tool calls to OpenAI format for compatibility with existing code.
    
    Args:
        tool_calls: Tool calls from Claude API
        
    Returns:
        Tool calls in OpenAI format
    """
    from dataclasses import dataclass
    
    @dataclass
    class FunctionCall:
        name: str
        arguments: str
    
    @dataclass
    class ToolCall:
        id: str
        type: str
        function: FunctionCall
    
    result = []
    for tc in tool_calls:
        result.append(ToolCall(
            id=tc["id"],
            type=tc["type"],
            function=FunctionCall(
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"]
            )
        ))
    
    return result
