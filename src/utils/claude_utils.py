"""
Claude API utilities for handling Anthropic Claude model interactions.
This module provides similar functionality to openai_utils.py but for Claude models.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional


# Map from internal model names to Anthropic API model names
CLAUDE_MODEL_MAP = {
    "claude/claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude/claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "claude/claude-3-opus": "claude-3-opus-20240229",
}


def get_anthropic_model_name(model: str) -> str:
    """Convert internal model name to Anthropic API model name."""
    return CLAUDE_MODEL_MAP.get(model, model)


def is_claude_model(model: str) -> bool:
    """Check if the model is a Claude model."""
    return model.startswith("claude/")


def convert_messages_for_claude(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Convert OpenAI-style messages to Claude format.
    
    Claude requires:
    - System message as a separate parameter (not in messages array)
    - Messages array without system messages
    - Different image format
    
    Args:
        messages: List of OpenAI-style messages
        
    Returns:
        Tuple of (system_prompt, converted_messages)
    """
    system_prompt = None
    converted_messages = []
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        # Extract system message
        if role == 'system':
            system_prompt = content if isinstance(content, str) else str(content)
            continue
        
        # Skip tool and tool_call messages for now (Claude handles tools differently)
        if role in ['tool', 'function']:
            continue
        
        # Convert content based on type
        if isinstance(content, str):
            converted_messages.append({
                "role": role,
                "content": content
            })
        elif isinstance(content, list):
            # Handle mixed content (text + images)
            claude_content = []
            for item in content:
                item_type = item.get('type', '')
                
                if item_type == 'text':
                    claude_content.append({
                        "type": "text",
                        "text": item.get('text', '')
                    })
                elif item_type == 'image_url':
                    # Convert image_url format to Claude's format
                    image_url_data = item.get('image_url', {})
                    url = image_url_data.get('url') if isinstance(image_url_data, dict) else str(image_url_data)
                    
                    if url:
                        # Claude expects base64 data or URLs in a specific format
                        if url.startswith('data:'):
                            # Handle base64 encoded images
                            # Format: data:image/png;base64,<base64data>
                            try:
                                media_type = url.split(';')[0].split(':')[1]
                                base64_data = url.split(',')[1]
                                claude_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_data
                                    }
                                })
                            except (IndexError, ValueError) as e:
                                logging.warning(f"Failed to parse base64 image: {e}")
                        else:
                            # Claude doesn't support direct URL images via API
                            # Convert to text description mentioning the image
                            claude_content.append({
                                "type": "text",
                                "text": f"[Image URL: {url}]"
                            })
                            logging.info(f"Converted image URL to text reference for Claude: {url[:80]}...")
            
            if claude_content:
                converted_messages.append({
                    "role": role,
                    "content": claude_content
                })
        elif content is not None:
            # Handle any other content types by converting to string
            converted_messages.append({
                "role": role,
                "content": str(content)
            })
    
    return system_prompt, converted_messages


async def call_claude_api(
    client,
    messages: List[Dict[str, Any]],
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Call the Claude API with the given messages.
    
    Args:
        client: Anthropic client instance
        messages: List of messages in OpenAI format
        model: Model name (internal format like "claude/claude-3-5-sonnet")
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        
    Returns:
        Dict containing response content and usage info
    """
    try:
        # Convert model name to Anthropic format
        anthropic_model = get_anthropic_model_name(model)
        
        # Convert messages to Claude format
        system_prompt, claude_messages = convert_messages_for_claude(messages)
        
        # Prepare API parameters
        api_params = {
            "model": anthropic_model,
            "max_tokens": max_tokens,
            "messages": claude_messages,
        }
        
        # Add system prompt if present
        if system_prompt:
            api_params["system"] = system_prompt
        
        # Add temperature (Claude supports 0-1 range)
        api_params["temperature"] = min(max(temperature, 0), 1)
        
        # Make the API call
        response = await client.messages.create(**api_params)
        
        # Extract content from response
        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text
        
        # Extract usage information
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage'):
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)
        
        return {
            "success": True,
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "stop_reason": getattr(response, 'stop_reason', None),
            "model": anthropic_model
        }
        
    except Exception as e:
        logging.error(f"Claude API call failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "content": None,
            "input_tokens": 0,
            "output_tokens": 0
        }


def get_claude_tools() -> List[Dict[str, Any]]:
    """
    Get tool definitions in Claude format.
    Claude uses a different tool format than OpenAI.
    
    Note: Tool support for Claude is simplified for now.
    Full tool support would require more extensive integration.
    """
    # For now, we return an empty list as tool support is complex
    # and would require significant changes to the tool handling logic
    # Users can use Claude models for text-only interactions
    return []
