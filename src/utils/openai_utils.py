import json
import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional, Callable

def get_tools_for_model() -> List[Dict[str, Any]]:
    """
    Returns the list of tools available to the model.
    
    Returns:
        List of tool objects
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Search the web for current information. Use this when you need to answer questions about current events or recent information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "The number of search results to return (1-10)",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 10
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
                "description": "Scrape and extract content from a webpage. Use this to get the content of a specific webpage.",
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
                "description": "Run code in Python or other supported languages. Use this to execute code, perform calculations, generate plots, and analyze data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to execute"
                        },
                        "language": {
                            "type": "string",
                            "description": "The programming language (default: python)",
                            "default": "python",
                            "enum": ["python", "javascript", "bash", "c++"]
                        },
                        "input": {
                            "type": "string",
                            "description": "Optional input data for the code"
                        }
                    },
                    "required": ["code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate images based on text prompts. Use this when the user asks for an image to be created.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt describing the image to generate"
                        },
                        "num_images": {
                            "type": "integer",
                            "description": "The number of images to generate (1-4)",
                            "default": 1,
                            "minimum": 1,
                            "maximum": 4
                        }
                    },
                    "required": ["prompt"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_data",
                "description": "Analyze data files (CSV, Excel) and create visualizations. Use this when users need to analyze data, create charts, or extract insights from their data files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query describing what analysis to perform on the data, including what type of chart to create (e.g. 'Create a histogram of ages', 'Show a pie chart of categories', 'Calculate average by group')"
                        },
                        "visualization_type": {
                            "type": "string",
                            "description": "The type of visualization to create",
                            "enum": ["bar", "line", "pie", "scatter", "histogram", "auto"],
                            "default": "auto"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_reminder",
                "description": "Set a reminder for the user. Use this when a user wants to be reminded about something at a specific time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content of the reminder"
                        },
                        "time": {
                            "type": "string",
                            "description": "The time for the reminder. Can be relative (e.g., '30m', '2h', '1d') or specific times ('tomorrow', '3:00pm', etc.)"
                        }
                    },
                    "required": ["content", "time"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "get_reminders",
                "description": "Get a list of upcoming reminders for the user. Use this when user asks about their reminders.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]

async def process_tool_calls(client, response, messages, tool_functions) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Process and execute tool calls from the OpenAI API response.
    
    Args:
        client: OpenAI client
        response: API response containing tool calls
        messages: The current chat messages
        tool_functions: Dictionary mapping tool names to handler functions
        
    Returns:
        Tuple containing (processed_any_tools, updated_messages)
    """
    processed_any = False
    tool_calls = response.choices[0].message.tool_calls
    
    # Create a copy of the messages to update
    updated_messages = messages.copy()
    
    # Add the assistant message with the tool calls
    updated_messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            } for tc in tool_calls
        ] if tool_calls else None
    })
    
    # Process each tool call
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        if function_name in tool_functions:
            # Parse the JSON arguments
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON in tool call arguments: {tool_call.function.arguments}")
                function_args = {}
                
            # Call the appropriate function
            try:
                function_response = await tool_functions[function_name](function_args)
                
                # Add the tool output back to messages
                updated_messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response)
                })
                
                processed_any = True
                
            except Exception as e:
                error_message = f"Error executing {function_name}: {str(e)}"
                logging.error(error_message)
                
                # Add the error as tool output
                updated_messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": error_message
                })
                
                processed_any = True

    return processed_any, updated_messages

def count_tokens(text: str) -> int:
    """Estimate token count using a simple approximation."""
    # Rough estimate: 1 word ≈ 1.3 tokens
    return int(len(text.split()) * 1.3)

def trim_content_to_token_limit(content: str, max_tokens: int = 8096) -> str:
    """Trim content to stay within token limit while preserving the most recent content."""
    current_tokens = count_tokens(content)
    if current_tokens <= max_tokens:
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

def prepare_messages_for_api(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare message history for the OpenAI API.
    
    Args:
        messages: List of message objects
        
    Returns:
        Prepared messages for API
    """
    prepared_messages = []
    
    # Check if there's a system message already
    has_system_message = any(msg.get('role') == 'system' for msg in messages)
    
    # If no system message exists, add a default one
    if not has_system_message:
        prepared_messages.append({
            "role": "system",
            "content": "You are a helpful AI assistant that can answer questions, provide information, and assist with various tasks."
        })
    
    for msg in messages:
        # Skip messages with None content
        if msg.get('content') is None:
            continue
            
        # Create a copy of the message to avoid modifying the original
        processed_msg = dict(msg)
        
        # Handle image URLs with timestamps in content
        if isinstance(processed_msg.get('content'), list):
            # Filter out images that have a timestamp (they're already handled specially)
            new_content = []
            for item in processed_msg['content']:
                if item.get('type') == 'image_url' and 'timestamp' in item:
                    # Remove timestamp from API calls
                    new_item = dict(item)
                    if 'timestamp' in new_item:
                        del new_item['timestamp']
                    new_content.append(new_item)
                else:
                    new_content.append(item)
            
            processed_msg['content'] = new_content
        
        prepared_messages.append(processed_msg)
        
    return prepared_messages