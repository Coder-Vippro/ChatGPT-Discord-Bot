from typing import List, Dict, Any, Optional, Tuple
import tiktoken
import json
from openai import AsyncOpenAI

def get_tools_for_model():
    """Returns the tools configuration for OpenAI API."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Search Google for up-to-date information on a topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "The number of results to return (default: 3)",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "scrape_webpage",
                "description": "Scrape and extract text content from a webpage URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the webpage to scrape"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "Execute code in Python or C++ and return the output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to execute"
                        },
                        "language": {
                            "type": "string",
                            "description": "The programming language to use (python or cpp)",
                            "enum": ["python", "cpp"]
                        },
                        "input": {
                            "type": "string",
                            "description": "Optional input data for the program (for cin>>, input() functions). All inputs should be on a single line, separated by spaces",
                        }
                    },
                    "required": ["code", "language"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate images from a text prompt using AI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detailed description of the image you want to generate"
                        },
                        "num_images": {
                            "type": "integer",
                            "description": "Number of images to generate (default: 1, max: 4)",
                            "default": 1
                        }
                    },
                    "required": ["prompt"]
                }
            }
        }
    ]
    return tools

async def process_tool_calls(client, model_response, messages_history, tool_functions):
    """Process tool calls returned by the model and add results to message history."""
    if (model_response.choices[0].finish_reason == "tool_calls" and 
        hasattr(model_response.choices[0].message, 'tool_calls')):
        # Add model's message with tool calls to history
        model_message = {"role": "assistant", "content": model_response.choices[0].message.content, "tool_calls": []}
        
        for tool_call in model_response.choices[0].message.tool_calls:
            model_message["tool_calls"].append({
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            })
        
        messages_history.append(model_message)
        
        # Process each tool call
        for tool_call in model_response.choices[0].message.tool_calls:
            if tool_call.type == "function":
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute the function if it exists
                if function_name in tool_functions:
                    try:
                        function_response = await tool_functions[function_name](function_args)
                        
                        # Add function response to messages
                        messages_history.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": str(function_response)
                        })
                    except Exception as e:
                        # Add error message if function execution failed
                        messages_history.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error executing function: {str(e)}"
                        })
        
        return True, messages_history
    
    return False, messages_history

def count_tokens(text: str) -> int:
    """Estimate token count using a simple approximation."""
    # Rough estimate: 1 word ≈ 1.3 tokens
    return int(len(text.split()) * 1.3)

def trim_content_to_token_limit(content: str, max_tokens: int = 8096) -> str:
    """Trim content to stay within token limit while preserving the most recent content."""
    current_tokens = count_tokens(content)
    if (current_tokens <= max_tokens):
        return content
        
    # Split into lines and start removing from the beginning until under limit
    lines = content.split('\n')
    while lines and count_tokens('\n'.join(lines)) > max_tokens:
        lines.pop(0)
    
    if not lines:  # If still too long, take the last part
        text = content
        while count_tokens(text) > max_tokens:
            text = text[text.find('\n', 1000):]
        return text
        
    return '\n'.join(lines)

def prepare_messages_for_api(messages, max_tokens=8096):
    """Prepare messages for API while ensuring token limit and no null content."""
    if not messages:
        from src.config.config import NORMAL_CHAT_PROMPT
        return [{"role": "system", "content": NORMAL_CHAT_PROMPT}]
        
    total_tokens = 0
    prepared_messages = []
    
    # Process messages in reverse order to keep the most recent ones
    for msg in reversed(messages):
        # Ensure message has valid role and content
        if not msg or not isinstance(msg, dict):
            continue
            
        role = msg.get('role')
        content = msg.get('content')
        
        if not role or content is None:
            continue
            
        # Convert complex content to text for token counting
        if isinstance(content, list):
            text_content = ""
            for item in content:
                if not item or not isinstance(item, dict):
                    continue
                    
                item_type = item.get('type')
                if item_type == 'text' and item.get('text'):
                    text_content += item.get('text', "") + "\n"
            
            # Skip if there's no actual text content
            if not text_content:
                continue
                
            msg_tokens = count_tokens(text_content)
            if total_tokens + msg_tokens > max_tokens:
                # Trim the content
                trimmed_text = trim_content_to_token_limit(text_content, max_tokens - total_tokens)
                if trimmed_text:
                    new_content = [{"type": "text", "text": trimmed_text}]
                    # Preserve any image URLs from the original content
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'image_url' and item.get('image_url'):
                            new_content.append(item)
                    prepared_messages.insert(0, {"role": role, "content": new_content})
                break
            else:
                prepared_messages.insert(0, msg)
                total_tokens += msg_tokens
        else:
            # Handle string content
            msg_content_str = str(content) if content is not None else ""
            if not msg_content_str:  # Skip empty content
                continue
                
            msg_tokens = count_tokens(msg_content_str)
            if total_tokens + msg_tokens > max_tokens:
                # Trim the content
                trimmed_text = trim_content_to_token_limit(msg_content_str, max_tokens - total_tokens)
                if trimmed_text:
                    prepared_messages.insert(0, {"role": role, "content": trimmed_text})
                break
            else:
                prepared_messages.insert(0, {"role": role, "content": msg_content_str})
                total_tokens += msg_tokens
    
    # Ensure we have at least one message with valid content
    if not prepared_messages:
        from src.config.config import NORMAL_CHAT_PROMPT
        return [{"role": "system", "content": NORMAL_CHAT_PROMPT}]
                
    return prepared_messages