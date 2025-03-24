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
    
    Returns:
        List of tool objects
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "analyze_data_file",
                "description": "Analyze a data file (CSV or Excel) and generate visualizations. Use this tool when a user uploads a data file and wants insights or visualizations. The visualizations will be automatically displayed in Discord. When describing the results, refer to visualizations by their chart_id and explain what they show. Always inform the user they can see the visualizations directly in the Discord chat.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the data file to analyze"
                        },
                        "analysis_type": {
                            "type": "string",
                            "description": "Type of analysis to perform (e.g., 'summary', 'correlation', 'distribution')",
                            "enum": ["summary", "correlation", "distribution", "comprehensive"]
                        }
                    },
                    "required": ["file_path"]
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
        },
        {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "Execute Python code to solve problems, perform calculations, or create data visualizations. Use this for data analysis, generating charts, and processing data. When analyzing data, ALWAYS include code for visualizations (using matplotlib, seaborn, or plotly) if the user requests charts or graphs. When visualizations are created, tell the user they can view the charts directly in Discord, and reference visualizations by their chart_id in your descriptions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute. For data analysis, include necessary imports (pandas, matplotlib, etc.) and visualization code."
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language (only Python supported)",
                            "enum": ["python", "py"]
                        },
                        "input": {
                            "type": "string",
                            "description": "Optional input data for the code"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Optional path to a data file to analyze (supports CSV and Excel files)"
                        },
                        "analysis_request": {
                            "type": "string",
                            "description": "Natural language description of the analysis to perform. If this includes visualization requests, the generated code must include plotting code using matplotlib, seaborn, or plotly."
                        },
                        "include_visualization": {
                            "type": "boolean",
                            "description": "Whether to include visualizations using matplotlib/seaborn"
                        }
                    }
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
        
        # Handle image URLs and file paths in content
        if isinstance(processed_msg.get('content'), list):
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
