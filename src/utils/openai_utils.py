import json
import logging
import os
import base64
import hashlib
import re
import threading
import datetime
import time
import traceback
import sys
from typing import List, Dict, Any, Tuple, Optional
import discord
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to sys.path to ensure imports work consistently
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def get_tools_for_model() -> List[Dict[str, Any]]:
    """Returns concise tool definitions optimized for token usage."""
    return [
        {
            "type": "function",
            "function": {
                "name": "analyze_data_file",
                "description": "Analyze CSV/Excel files with templates or custom analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to CSV/Excel file"},
                        "analysis_type": {
                            "type": "string", 
                            "enum": ["summary", "correlation", "distribution", "comprehensive"],
                            "default": "comprehensive"
                        },
                        "custom_analysis": {"type": "string", "description": "Custom analysis request"}
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_image",
                "description": "Edit images (remove background, etc). Returns image URLs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_url": {"type": "string", "description": "Image URL to edit"},
                        "operation": {"type": "string", "enum": ["remove_background"], "default": "remove_background"}
                    },
                    "required": ["image_url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "enhance_prompt",
                "description": "Create enhanced versions of text prompts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Original prompt"},
                        "num_versions": {"type": "integer", "default": 3, "minimum": 1, "maximum": 5}
                    },
                    "required": ["prompt"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "image_to_text",
                "description": "Convert image to text description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_url": {"type": "string", "description": "Image URL to analyze"}
                    },
                    "required": ["image_url"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "upscale_image",
                "description": "Upscale image resolution. Returns image URLs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_url": {"type": "string", "description": "Image URL to upscale"},
                        "scale_factor": {"type": "integer", "enum": [2, 3, 4], "default": 4}
                    },
                    "required": ["image_url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "photo_maker",
                "description": "Generate images based on reference photos. Returns image URLs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Generation prompt"},
                        "input_images": {"type": "array", "items": {"type": "string"}, "description": "Reference image URLs"},
                        "strength": {"type": "integer", "default": 40, "minimum": 1, "maximum": 100},
                        "num_images": {"type": "integer", "default": 1, "minimum": 1, "maximum": 4}
                    },
                    "required": ["prompt", "input_images"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_image_with_refiner",
                "description": "Generate high-quality images. Returns image URLs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Image prompt"},
                        "num_images": {"type": "integer", "default": 1, "minimum": 1, "maximum": 4},
                        "negative_prompt": {"type": "string", "default": "blurry, low quality"}
                    },
                    "required": ["prompt"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Search web for current information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "num_results": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "scrape_webpage",
                "description": "Extract content from webpage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Webpage URL"}
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate images from text prompts. Returns image URLs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Image prompt"},
                        "num_images": {"type": "integer", "default": 1, "minimum": 1, "maximum": 4}
                    },
                    "required": ["prompt"]
                }
            }
        },
        {
            "type": "function",            
            "function": {
                "name": "execute_python_code",
                "description": "Execute Python code with package installation support. If you need specific packages, list them in 'install_packages' parameter. The system will install them before running your code. MUST use print() for output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code with print() statements"},
                        "input_data": {"type": "string", "description": "Optional input data"},
                        "install_packages": {
                            "type": "array", 
                            "items": {"type": "string"}, 
                            "description": "List of pip package names to install before running code (e.g., ['requests', 'beautifulsoup4', 'opencv-python'])"
                        },
                        "enable_visualization": {"type": "boolean", "description": "For charts/graphs"},
                        "timeout": {"type": "integer", "default": 30, "minimum": 1, "maximum": 240}
                    },
                    "required": ["code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_reminder",
                "description": "Set user reminder. Supports relative time (30m, 2h, 1d), specific times (9:00, 15:30, 9:00 pm, 2:30 am), keywords (tomorrow, tonight, noon), and combinations (9:00 pm today, 2:00 pm tomorrow).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Reminder content"},
                        "time": {"type": "string", "description": "Time in formats like: '30m', '2h', '1d', '9:00', '15:30', '9:00 pm', '2:30 am', '9:00 pm today', '2:00 pm tomorrow', 'tomorrow', 'tonight', 'noon'"}
                    },
                    "required": ["content", "time"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "get_reminders",
                "description": "Get user reminders list.",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }
    ]

async def process_tool_calls(client, response, messages, tool_functions) -> Tuple[bool, List[Dict[str, Any]]]:
    """Process and execute tool calls from the OpenAI API response."""
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

async def prepare_file_from_path(file_path: str) -> discord.File:
    """Convert a file path to a Discord File object."""
    return discord.File(file_path)

def prepare_messages_for_api(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare message history for the OpenAI API with image URL handling."""
    prepared_messages = []
    
    # Check if there's a system message already
    has_system_message = any(msg.get('role') == 'system' for msg in messages)
    
    # If no system message exists, add a default one
    if not has_system_message:
        prepared_messages.append({
            "role": "system",
            "content": "You are a helpful AI assistant. For image tools, you'll receive image URLs in responses - reference them instead of sending binary data."
        })
    
    for msg in messages:
        # Skip messages with None content
        if msg.get('content') is None:
            continue
            
        # Create a copy of the message to avoid modifying the original
        processed_msg = dict(msg)
        
        # Handle image URLs differently based on message role
        if isinstance(processed_msg.get('content'), list):
            # For assistant messages, convert image URLs to text descriptions
            if processed_msg.get('role') == 'assistant':
                text_parts = []
                
                # Extract text and reference images in a text format instead
                for item in processed_msg['content']:
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'image_url':
                        # Add a text reference to the image instead of the actual image URL
                        image_desc = "[Image URL provided in response]"
                        text_parts.append(image_desc)
                
                # Join all text parts into a single string
                processed_msg['content'] = ' '.join(text_parts)
                
            # For user messages, keep the image URLs as they are allowed
            elif processed_msg.get('role') == 'user':
                new_content = []
                for item in processed_msg['content']:
                    if item.get('type') == 'image_url':
                        new_item = {
                            'type': 'image_url',
                            'image_url': item.get('image_url', '')
                        }
                        new_content.append(new_item)
                    else:
                        new_content.append(item)
                processed_msg['content'] = new_content
        
        prepared_messages.append(processed_msg)
        
    return prepared_messages

def generate_data_analysis_code(analysis_request: str, file_path: str) -> str:
    """Generate Python code for data analysis based on user request."""
    # Set up imports
    code = """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""

    # Basic data loading
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.xlsx':
        code += f"\n# Read the Excel file\ndf = pd.read_excel('{file_path}')\n"
    else:
        code += f"\n# Read the CSV file\ndf = pd.read_csv('{file_path}')\n"

    # Basic data exploration
    code += """
# Display basic information
print("Dataset Info:")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\\nColumns:", df.columns.tolist())
print("\\nData Types:")
print(df.dtypes)
print("\\nMissing Values:")
print(df.isnull().sum())
"""

    # Generate specific analysis code based on request
    if 'correlation' in analysis_request.lower():
        code += """
# Generate correlation matrix
plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=['number']).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
"""

    if any(word in analysis_request.lower() for word in ['distribution', 'histogram']):
        code += """
# Plot distributions for numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns
for col in numeric_cols[:3]:  # Limit to first 3 columns
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
"""

    return code

# Simplified API function without retries to avoid extra costs
async def call_openai_api(client, messages, model, temperature=0.7, max_tokens=None, tools=None):
    """Call OpenAI API without retry logic to avoid extra costs."""
    try:
        # Prepare API parameters
        api_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": "auto" if tools else None
        }
        
        # Add temperature only for models that support it (exclude GPT-5 family)
        if model not in ["openai/gpt-5", "openai/gpt-5-nano", "openai/gpt-5-mini", "openai/gpt-5-chat"]:
            api_params["temperature"] = temperature
        
        # Single API call without retries
        response = await client.chat.completions.create(**api_params)
        return response
    except Exception as e:
        logging.error(f"OpenAI API call failed: {str(e)}")
        raise e

async def analyze_with_ai(
    messages: List[Dict[str, Any]], 
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    file_path: Optional[str] = None,
    analysis_request: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze with AI using optimized token usage."""
    response = {"success": True}
    
    try:
        # Process messages for API
        prepared_messages = prepare_messages_for_api(messages)
        
        if file_path and analysis_request:
            # Generate data analysis code
            analysis_code = generate_data_analysis_code(analysis_request, file_path)
            response["generated_code"] = analysis_code
            
            # Add analysis context to messages
            prepared_messages.append({
                "role": "system",
                "content": f"Data file analysis requested. Generated code available."
            })
        
        # The actual API call would go here
        # response = await call_openai_api(client, prepared_messages, model, temperature, tools=get_tools_for_model())
        
    except Exception as e:
        logging.error(f"Error in analyze_with_ai: {str(e)}")
        response["success"] = False
        response["error"] = str(e)
        
    return response
