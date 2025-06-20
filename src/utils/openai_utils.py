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
    """
    Returns the list of tools available to the model.
    
    IMPORTANT: CODE EXECUTION HAS BEEN SEPARATED INTO TWO SPECIALIZED TOOLS:
    
    1. execute_python_code: For general programming, calculations, algorithms, custom code
       - Use when: math problems, programming tasks, custom scripts, algorithm implementations
       - Can create visualizations from scratch
       - Has sandboxed environment with package installation
       - Automatically gets context about uploaded files when available
    
    2. analyze_data_file: For structured data analysis from CSV/Excel files  
       - Use when: user explicitly requests data analysis, statistics, or insights from data files
       - Has pre-built analysis templates (summary, correlation, distribution, comprehensive)
       - Automatically handles data loading and creates appropriate visualizations
       - Specialized for data science workflows
       - Best for quick data exploration and standard analysis
    
    AI MODEL GUIDANCE FOR DATA FILE UPLOADS:
    - When user uploads a data file (.csv, .xlsx, .xls) to Discord:
      * File is automatically downloaded and saved
      * User intent is detected (data analysis vs. general programming)
      * If intent is "data analysis" → analyze_data_file is automatically called
      * If intent is "general programming" → file context is added to conversation
    
    - When to use analyze_data_file:
      * User uploads data file and asks for analysis, insights, statistics
      * User wants standard data exploration (correlations, distributions, summaries)
      * User requests "analyze this data" type queries
    
    - When to use execute_python_code:
      * User asks for calculations, math problems, algorithms
      * User wants custom code or programming solutions
      * User uploads data file but wants custom processing (not standard analysis)
      * User needs specific data transformations or custom visualizations
      * File paths will be automatically provided in the execution environment
    
    DISCORD FILE INTEGRATION:
    - Data files uploaded to Discord are automatically downloaded and saved
    - File paths are automatically provided to the appropriate tools
    - Context about uploaded files is added to Python execution environment
    - Visualizations are automatically uploaded to Discord and displayed    
    Returns:
        List of tool objects for the OpenAI API
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "analyze_data_file",
                "description": "**DATA ANALYSIS TOOL** - Use this tool for structured data analysis from CSV/Excel files. This tool specializes in analyzing data with pre-built templates (summary, correlation, distribution, comprehensive) and generates appropriate visualizations automatically. Use this tool when: (1) User explicitly requests data analysis or insights, (2) User asks for statistics, correlations, or data exploration, (3) User wants standard data science analysis. The tool handles file loading, data validation, and creates visualizations that are automatically displayed in Discord.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the data file to analyze (CSV or Excel format required)"
                        },
                        "analysis_type": {
                            "type": "string",
                            "description": "Type of pre-built analysis template to use",
                            "enum": ["summary", "correlation", "distribution", "comprehensive"],
                            "default": "comprehensive"
                        },
                        "custom_analysis": {
                            "type": "string",
                            "description": "Optional custom analysis request in natural language. If provided, this overrides the analysis_type and generates custom Python code for the specific analysis requested."
                        },
                        "install_packages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of Python packages to install before analysis"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_image",
                "description": "Edit an image with operations like background removal. Use this when a user wants to edit an existing image, such as removing the background.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_url": {
                            "type": "string",
                            "description": "URL of the image to edit. This should be an image URL from the conversation."
                        },
                        "operation": {
                            "type": "string", 
                            "description": "Type of edit operation to perform",
                            "enum": ["remove_background"],
                            "default": "remove_background"
                        }
                    },
                    "required": ["image_url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "enhance_prompt",
                "description": "Enhance a text prompt with AI to create more detailed or creative versions. Use this when a user wants help creating better image prompts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The original text prompt to enhance"
                        },
                        "num_versions": {
                            "type": "integer",
                            "description": "Number of enhanced versions to generate (1-5)",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 5
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum length of each enhanced prompt",
                            "default": 100
                        }
                    },
                    "required": ["prompt"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "image_to_text",
                "description": "Convert an image to a text description/caption. Use this when a user wants to understand what's in an image or get a caption for it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_url": {
                            "type": "string",
                            "description": "URL of the image to analyze. This should be an image URL from the conversation."
                        }
                    },
                    "required": ["image_url"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "upscale_image",
                "description": "Upscale an image to a higher resolution. Use this when a user wants to improve the quality or size of an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_url": {
                            "type": "string",
                            "description": "URL of the image to upscale. This should be an image URL from the conversation."
                        },
                        "scale_factor": {
                            "type": "integer",
                            "description": "Factor by which to upscale the image (2, 3, or 4)",
                            "default": 4,
                            "enum": [2, 3, 4]
                        }
                    },
                    "required": ["image_url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "photo_maker",
                "description": "Generate images based on reference photos and a text prompt. Use this when a user wants to create images that maintain the style or characteristics of reference images.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text prompt describing what to generate"
                        },
                        "input_images": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of reference image URLs to use as input (1-4 images)"
                        },
                        "style": {
                            "type": "string",
                            "description": "Style to apply to the generated image",
                            "default": "No style"
                        },
                        "strength": {
                            "type": "integer",
                            "description": "Strength of the input images' influence (1-100)",
                            "default": 40,
                            "minimum": 1,
                            "maximum": 100
                        },
                        "num_images": {
                            "type": "integer",
                            "description": "Number of images to generate (1-4)",
                            "default": 1,
                            "minimum": 1,
                            "maximum": 4
                        }
                    },
                    "required": ["prompt", "input_images"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_image_with_refiner",
                "description": "Generate high-quality images using a refiner model for better details. Use this when a user wants premium quality image generation.",
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
                        },
                        "negative_prompt": {
                            "type": "string",
                            "description": "Things to avoid in the generated image",
                            "default": "blurry, distorted, low quality, disfigured"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        },
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
        },        {
            "type": "function",            "function": {
                "name": "execute_python_code",
                "description": "**GENERAL PYTHON EXECUTION TOOL** - Use this tool for general programming tasks, mathematical calculations, algorithm implementations, and custom Python scripts. Use this tool when: (1) User asks for calculations or math problems, (2) User wants to run custom Python code, (3) User needs algorithm implementations, (4) User requests programming solutions, (5) Creating custom visualizations or data processing, (6) User uploads data files but wants custom code rather than standard analysis. File paths for uploaded data files are automatically made available in the execution environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute. Include all necessary imports and ensure code is complete and runnable."
                        },
                        "input_data": {
                            "type": "string",
                            "description": "Optional input data to be made available to the code as a variable named 'input_data'"
                        },
                        "install_packages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of Python packages to install before execution (e.g., ['numpy', 'matplotlib'])"
                        },
                        "enable_visualization": {
                            "type": "boolean",
                            "description": "Set to true when creating charts, graphs, or any visual output with matplotlib/seaborn"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Maximum execution time in seconds (default: 30, max: 120)",
                            "default": 30,
                            "minimum": 1,
                            "maximum": 120
                        }
                    },
                    "required": ["code"]
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
        
    return '\n.join(lines)'

async def prepare_file_from_path(file_path: str) -> discord.File:
    """Convert a file path to a Discord File object."""
    return discord.File(file_path)

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
            "content": "You are a helpful AI assistant that can answer questions, provide information, and assist with various tasks. When handling data analysis, always describe the visualizations in detail and refer to them by their chart_id. For any data file uploaded by the user, use the analyze_data_file tool or code_interpreter tool to generate analysis and visualizations. When visualizations are created, they will be automatically displayed in the Discord chat. Always mention that the user can see the visualizations directly in Discord."
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
                        image_desc = "[Image previously shared]"
                        text_parts.append(image_desc)
                
                # Join all text parts into a single string
                processed_msg['content'] = ' '.join(text_parts)
                
            # For user messages, keep the image URLs as they are allowed
            elif processed_msg.get('role') == 'user':
                new_content = []
                for item in processed_msg['content']:
                    if item.get('type') == 'image_url':
                        # Remove timestamp and ensure we're using the actual file path
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
    """
    Generate Python code for data analysis based on user request
    """
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

    if 'scatter' in analysis_request.lower():
        code += """
# Generate scatter plots for numeric columns
numeric_cols = df.select_dtypes(include(['number']).columns
if len(numeric_cols) >= 2:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1])
    plt.title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}')
    plt.tight_layout()
"""

    if 'box' in analysis_request.lower() or 'boxplot' in analysis_request.lower():
        code += """
# Generate box plots for numeric columns
numeric_cols = df.select_dtypes(include(['number']).columns
plt.figure(figsize=(12, 6))
df[numeric_cols].boxplot()
plt.xticks(rotation=45)
plt.title('Box Plots of Numeric Variables')
plt.tight_layout()
"""

    return code

async def analyze_with_ai(
    messages: List[Dict[str, Any]], 
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    file_path: Optional[str] = None,
    analysis_request: Optional[str] = None
) -> Dict[str, Any]:
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
                "content": f"The user has provided a data file for analysis. Generated code:\n{analysis_code}"
            })
        
        # Add your existing OpenAI API call logic here
        # ... existing API call code ...
        
    except Exception as e:
        logging.error(f"Error in analyze_with_ai: {str(e)}")
        response["success"] = False
        response["error"] = str(e)
        
    return response
