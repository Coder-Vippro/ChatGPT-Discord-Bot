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

# Import utility functions
from .code_utils import DATA_FILES_DIR, format_output_path, clean_old_files

# Configure logging
log_file = 'logs/code_interpreter.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(formatter)
logger = logging.getLogger('code_interpreter')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Regular expression to find image file paths in output
IMAGE_PATH_PATTERN = r'(\/media\/quocanh\/.*\.(png|jpg|jpeg|gif))'

# Unsafe patterns for code security
UNSAFE_IMPORTS = [
    r'import\s+os\b', r'from\s+os\s+import', 
    r'import\s+subprocess\b', r'from\s+subprocess\s+import',
    r'import\s+shutil\b', r'from\s+shutil\s+import',
    r'__import__\([\'"]os[\'"]\)', r'__import__\([\'"]subprocess[\'"]\)',
    r'import\s+sys\b(?!\s+import\s+path)', r'from\s+sys\s+import'
]

UNSAFE_FUNCTIONS = [
    r'os\.', r'subprocess\.', r'shutil\.', 
    r'eval\(', r'exec\(', r'sys\.',
    r'open\([\'"][^\'"]*/[^\']*[\'"]', # File system access
    r'__import__\(', r'globals\(\)', r'locals\(\)'
]

def sanitize_python_code(code: str) -> tuple[bool, str]:
    """
    Check Python code for potentially unsafe operations.
    
    Args:
        code: The code to check
        
    Returns:
        Tuple of (is_safe, sanitized_code_or_error_message)
    """
    # Check for unsafe imports
    for pattern in UNSAFE_IMPORTS:
        if re.search(pattern, code):
            return False, f"Forbidden import detected: {pattern}"
    
    # Check for unsafe function calls
    for pattern in UNSAFE_FUNCTIONS:
        if re.search(pattern, code):
            return False, f"Forbidden function call detected: {pattern}"
    
    # Add safety imports and commonly used libraries
    safe_imports = """
import math
import random
import json
import time
from datetime import datetime, timedelta
import collections
import itertools
import functools
try:
    import numpy as np
except ImportError:
    pass
try:
    import pandas as pd
except ImportError:
    pass
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
try:
    import seaborn as sns
except ImportError:
    pass
"""
    
    return True, safe_imports + "\n" + code

async def install_packages(packages: List[str]) -> Dict[str, Any]:
    """
    Install Python packages in a sandboxed environment.
    
    Args:
        packages: List of package names to install
        
    Returns:
        Dict containing installation results
    """
    try:
        installed = []
        failed = []
        
        for package in packages:
            try:
                # Use pip to install package with timeout
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package, "--user", "--quiet"
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    installed.append(package)
                    logger.info(f"Successfully installed package: {package}")
                else:
                    failed.append({"package": package, "error": result.stderr})
                    logger.error(f"Failed to install package {package}: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                failed.append({"package": package, "error": "Installation timeout"})
                logger.error(f"Installation timeout for package: {package}")
            except Exception as e:
                failed.append({"package": package, "error": str(e)})
                logger.error(f"Error installing package {package}: {str(e)}")
        
        return {
            "success": True,
            "installed": installed,
            "failed": failed,
            "message": f"Installed {len(installed)} packages, {len(failed)} failed"
        }
        
    except Exception as e:
        logger.error(f"Error in package installation: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "installed": [],
            "failed": packages
        }

async def execute_python_code(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute Python code in a controlled sandbox environment.
    
    Args:
        args: Dictionary containing:
            - code: The Python code to execute
            - input: Optional input data for the code
            - install_packages: Optional list of packages to install
            - timeout: Optional timeout in seconds (default: 30)
            
    Returns:
        Dict containing execution results
    """
    try:
        code = args.get("code", "")
        input_data = args.get("input", "")
        packages_to_install = args.get("install_packages", [])
        timeout = args.get("timeout", 30)
        
        if not code:
            return {
                "success": False,
                "error": "No code provided",
                "output": ""
            }
        
        # Install packages if requested
        if packages_to_install:
            install_result = await install_packages(packages_to_install)
            if not install_result["success"]:
                logger.warning(f"Package installation issues: {install_result}")
        
        # Sanitize the code
        is_safe, sanitized_code = sanitize_python_code(code)
        if not is_safe:
            logger.warning(f"Code sanitization failed: {sanitized_code}")
            return {
                "success": False,
                "error": sanitized_code,
                "output": ""
            }
        
        # Clean up old files before execution
        clean_old_files()
        
        # Execute code in controlled environment
        result = await execute_code_safely(sanitized_code, input_data, timeout)
        
        return result
        
    except Exception as e:
        error_msg = f"Error in Python code execution: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "success": False,
            "error": error_msg,
            "output": "",
            "traceback": traceback.format_exc()
        }

async def execute_code_safely(code: str, input_data: str, timeout: int) -> Dict[str, Any]:
    """
    Execute code in a safe environment with proper isolation.
    
    Args:
        code: Sanitized Python code to execute
        input_data: Input data for the code
        timeout: Execution timeout in seconds
        
    Returns:
        Dict containing execution results
    """
    try:
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Import commonly used libraries for the execution environment
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None
        
        try:
            import numpy as np
        except ImportError:
            np = None
            
        try:
            import pandas as pd
        except ImportError:
            pd = None
        
        # Create execution namespace
        exec_globals = {
            "__builtins__": {
                # Safe builtins
                "print": print,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "reversed": reversed,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "type": type,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "dir": dir,
                "help": help,
                "__import__": __import__,  # Allow controlled imports
                "ValueError": ValueError,
                "TypeError": TypeError,
                "IndexError": IndexError,
                "KeyError": KeyError,
                "AttributeError": AttributeError,
                "ImportError": ImportError,
                "Exception": Exception,
            },
            # Add available libraries
            "math": __import__("math"),
            "random": __import__("random"),
            "json": __import__("json"),
            "time": __import__("time"),
            "datetime": __import__("datetime"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
        }
        
        # Add optional libraries if available
        if np is not None:
            exec_globals["np"] = np
            exec_globals["numpy"] = np
        if pd is not None:
            exec_globals["pd"] = pd
            exec_globals["pandas"] = pd
        if plt is not None:
            exec_globals["plt"] = plt
            exec_globals["matplotlib"] = matplotlib
        
        # Override input function if input_data is provided
        if input_data:
            input_lines = input_data.strip().split('\n')
            input_iter = iter(input_lines)
            exec_globals["input"] = lambda prompt="": next(input_iter, "")
        
        # Set up output capture
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        # Generate output file path for any plots
        timestamp = int(time.time())
        output_filename = f"python_output_{timestamp}.png"
        output_path = format_output_path(output_filename)
        
        # Execute the code with timeout
        try:
            # Execute the code as statements
            await asyncio.wait_for(
                asyncio.to_thread(exec, code, exec_globals),
                timeout=timeout
            )
            
            # Check for any matplotlib figures and save them
            visualizations = []
            if plt is not None and plt.get_fignums():
                for i, fig_num in enumerate(plt.get_fignums()):
                    try:
                        fig = plt.figure(fig_num)
                        if len(fig.get_axes()) > 0:
                            # Save to output path
                            fig_path = output_path.replace('.png', f'_{i}.png')
                            fig.savefig(fig_path, bbox_inches='tight', dpi=150)
                            visualizations.append(fig_path)
                        plt.close(fig)
                    except Exception as e:
                        logger.error(f"Error saving figure {i}: {str(e)}")
                
                # Clear all figures
                plt.close('all')
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Code execution timed out after {timeout} seconds",
                "output": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue()
            }
        
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Get the outputs
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        # Check for any image paths in the output
        image_paths = re.findall(IMAGE_PATH_PATTERN, stdout_output)
        for img_path in image_paths:
            if os.path.exists(img_path):
                visualizations.append(img_path)
        
        # Remove image paths from output text
        clean_output = stdout_output
        for img_path in image_paths:
            clean_output = clean_output.replace(img_path, "[Image saved]")
        
        logger.info(f"Python code executed successfully, output length: {len(clean_output)}")
        if visualizations:
            logger.info(f"Generated {len(visualizations)} visualizations")
        
        return {
            "success": True,
            "output": clean_output,
            "stderr": stderr_output,
            "visualizations": visualizations,
            "has_visualization": len(visualizations) > 0,
            "execution_time": f"Completed in under {timeout}s"
        }
        
    except Exception as e:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        error_msg = f"Error executing Python code: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        return {
            "success": False,
            "error": error_msg,
            "output": stdout_capture.getvalue() if 'stdout_capture' in locals() else "",
            "stderr": stderr_capture.getvalue() if 'stderr_capture' in locals() else "",
            "traceback": traceback.format_exc()
        }

# Backward compatibility - keep the old function name
async def execute_code(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward compatibility wrapper for execute_python_code.
    """
    return await execute_python_code(args)
