import os
import sys
import io
import re
import logging
import asyncio
import subprocess
import tempfile
import time
import uuid
from logging.handlers import RotatingFileHandler
import traceback
import contextlib
from typing import Dict, Any, Optional, List

# Import the new separated modules
from .python_executor import execute_python_code
from .data_analyzer import analyze_data_file

# Configure logging
log_file = 'logs/code_interpreter.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(formatter)
logger = logging.getLogger('code_interpreter')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

async def execute_code(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for code execution - routes to appropriate handler.
    
    This function maintains backward compatibility while routing requests
    to the appropriate specialized handler.
    
    Args:
        args: Dictionary containing execution parameters
        
    Returns:
        Dict containing execution results
    """
    try:
        # Check if this is a data analysis request
        file_path = args.get("file_path", "")
        analysis_request = args.get("analysis_request", "")
        
        if file_path and (analysis_request or args.get("analysis_type")):
            # Route to data analyzer
            logger.info("Routing to data analyzer")
            return await analyze_data_file(args)
        else:
            # Route to Python executor
            logger.info("Routing to Python executor")
            return await execute_python_code(args)
            
    except Exception as e:
        error_msg = f"Error in code execution router: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "success": False,
            "error": error_msg,
            "output": "",
            "traceback": traceback.format_exc()
        }
    """
    Execute code with support for data analysis and visualization.
    
    Args:
        args: Dictionary containing:
            - code: The code to execute (optional if analysis_request is provided)
            - language: Programming language (default: python)
            - input: Optional input data
            - file_path: Optional path to data file
            - analysis_request: Optional natural language analysis request
            - include_visualization: Whether to include visualizations
            - user_id: Optional user ID for file management
            
    Returns:
        Dict containing execution results
    """
    try:
        code = args.get("code", "")
        language = args.get("language", "python")
        input_data = args.get("input", "")
        file_path = args.get("file_path", "")
        analysis_request = args.get("analysis_request", "")
        include_visualization = args.get("include_visualization", True)
        user_id = args.get("user_id")
        logger.info(f"Executing code for user {user_id}")
        if file_path:
            logger.info(f"Using data file: {file_path}")

        # Clean up old files before execution
        await asyncio.to_thread(clean_old_files)

        # If we have an analysis request but no code, generate the code
        if analysis_request and file_path and not code:
            logger.info(f"Generating analysis code for request: {analysis_request}")
            code = generate_analysis_code(file_path, analysis_request)
            
        # If still no code, return error
        if not code:
            logger.error("No code provided or generated")
            return {
                "success": False,
                "output": "No code provided or generated",
                "code": "",
                "error": "No code to execute"
            }

        # Sanitize the code
        logger.info("Sanitizing code")
        is_safe, sanitized_code = sanitize_code(code, language)
        if not is_safe:
            logger.warning(f"Code sanitization failed: {sanitized_code}")
            return {
                "success": False,
                "output": sanitized_code,  # Contains error message
                "code": code,
                "error": "Code contains unsafe operations"
            }

        # Prepare execution environment
        output_buffer = io.StringIO()
        visualization_paths = []
        error_message = None
        # Reset matplotlib state
        plt.close('all')
        
        # Track file paths created during execution
        file_paths_created = []
        
        # Create namespace for code execution with file_path
        namespace = {
            '__name__': '__main__',
            'print': lambda *args, **kwargs: print(*args, **kwargs, file=output_buffer),
            'input': lambda *args: input_data,
            'file_path': file_path,
            'DATA_FILES_DIR': DATA_FILES_DIR,
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns
        }

        # Execute the code with captured output
        logger.info("Executing code")
        with contextlib.redirect_stdout(output_buffer):
            with contextlib.redirect_stderr(output_buffer):
                try:
                    # Set style before execution
                    plt.style.use('default')
                    sns.set_theme()
                    
                    # Execute the code
                    exec(sanitized_code, namespace)
                    
                    # Process output to remove sandbox paths using format_output_path
                    output = output_buffer.getvalue()
                    output = format_output_path(output)
                    output_buffer = io.StringIO()
                    output_buffer.write(output)
                    
                    # Save any generated plots with proper cleanup
                    if plt.get_fignums() and include_visualization:
                        for i, fig in enumerate(plt.get_fignums()):
                            try:
                                current_fig = plt.figure(fig)
                                # Ensure the figure has content
                                if len(current_fig.get_axes()) > 0:
                                    # Save to bytes buffer instead of file
                                    img_buffer = io.BytesIO()
                                    current_fig.tight_layout()
                                    current_fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
                                    img_buffer.seek(0)
                                    visualization_paths.append(img_buffer.getvalue())
                                    logger.info(f"Generated visualization {i+1}")
                                plt.close(fig)
                            except Exception as e:
                                logger.error(f"Error saving figure {i}: {str(e)}")
                                continue
                                
                except Exception as e:
                    error_message = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_message)
                finally:
                    # Ensure all figures are properly closed
                    plt.close('all')

        # Get the output
        output = output_buffer.getvalue()
        output_buffer.close()
        
        logger.info("Code execution completed")
        if output:
            logger.info(f"Output length: {len(output)} characters")
        
        if visualization_paths:
            logger.info(f"Generated {len(visualization_paths)} visualizations")
            
        # Check if there are image file paths in the output and read those files
        image_paths = re.findall(IMAGE_PATH_PATTERN, output)
        for _, img_path in image_paths:
            if os.path.exists(img_path) and os.path.isfile(img_path):
                try:
                    with open(img_path, 'rb') as img_file:
                        img_data = img_file.read()
                        visualization_paths.append(img_data)
                        logger.info(f"Added image from path: {img_path}")
                except Exception as e:
                    logger.error(f"Error reading image file {img_path}: {str(e)}")
        
        # If image paths were found in output, replace them with placeholders
        if image_paths:
            for i, (prefix, img_path) in enumerate(image_paths):
                output = output.replace(f"{prefix or ''}{img_path}", f"[Image {i+1} will be displayed in Discord]")
            
        # Format the response with binary image data
        response = {
            "success": error_message is None,
            "output": error_message if error_message else output,
            "code": code,
            "has_chart": len(visualization_paths) > 0,
            "binary_images": visualization_paths  # Now contains list of binary image data
        }
        
        if error_message:
            response["error"] = error_message
            
        return response
    except Exception as e:
        error_message = f"Error in code interpreter: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return {
            "success": False,
            "output": error_message,
            "code": code if 'code' in locals() else "",
            "error": str(e)
        }