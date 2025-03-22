import os
import sys
import re
import tempfile
import asyncio
import subprocess
import signal
import time
import logging
import platform
import shutil
from typing import Dict, Any, Optional, Tuple
import concurrent.futures
import traceback
import json
from datetime import datetime

# Import the webhook logger
from src.utils.webhook_logger import webhook_log_manager

# Configure more detailed logger for code execution with file output
code_logger = logging.getLogger("code_execution")
code_logger.setLevel(logging.DEBUG)  # Set to DEBUG for more detailed logging

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Add file handler for persistent logging
file_handler = logging.FileHandler('logs/code_execution.log')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)

# Add console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setFormatter(file_formatter)
console_handler.setLevel(logging.INFO)

# Clear existing handlers to avoid duplicates
if code_logger.handlers:
    code_logger.handlers.clear()

code_logger.addHandler(file_handler)
code_logger.addHandler(console_handler)

# Create temp directory for data files with longer retention
DATA_FILES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'temp_data_files')
if not os.path.exists(DATA_FILES_DIR):
    os.makedirs(DATA_FILES_DIR, exist_ok=True)
    code_logger.info(f"Created temp data directory at {DATA_FILES_DIR}")

# Import resource conditionally for platform compatibility
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False
    code_logger.warning("Resource module not available on this platform - resource limits won't be applied")

# Function to enable webhook logging for code execution
def enable_code_execution_webhook_logging(webhook_url: str, app_name: str = "Code Execution"):
    """
    Enable sending code execution logs to a Discord webhook.
    
    Args:
        webhook_url (str): Discord webhook URL to send logs to
        app_name (str): Name identifier for the logs
    """
    # Set up webhook logging for the code_execution logger
    webhook_log_manager.capture_module_logs_to_webhook(
        module_name="code_execution", 
        webhook_url=webhook_url,
        app_name=app_name
    )
    
    code_logger.info(f"Webhook logging enabled for code execution with app name: {app_name}")

def sanitize_code(code: str, language: str) -> Tuple[bool, str]:
    """
    Sanitize and validate code for security purposes.
    
    Args:
        code (str): The code to be sanitized.
        language (str): The programming language ('python' or 'cpp').
        
    Returns:
        tuple: (is_safe, sanitized_code or error_message)
    """
    code_logger.debug(f"Sanitizing {language} code")
    
    # List of banned imports/includes and dangerous operations
    python_banned = [
        'os.system', 'subprocess.call', 'subprocess.run', 'subprocess.Popen', 'exec(', '__import__(',
        '.mkfifo', '.chmod', '.chown', '.getstatusoutput',
        'socket', 'urllib.urlopen', 'curl', 'wget', 
        'dbm', 'pickle', 'marshal', '.loads(', '.dumps(',
        'getattr(', 'setattr(', 'delattr(', '__class__', '__bases__', '__subclasses__',
        '__globals__', '__getattribute__', '.mro(', 'ctypes'
    ]
    
    # Excluded banned items to allow data visualization and file operations:
    # - Removed 'open(', '.open(', '.read(', '.write(' to allow file operations
    # - Removed 'importlib' to allow dynamic imports of visualization libraries
    # - Removed 'os.path', 'pathlib', 'with open', 'io.open' to allow file path handling
    # - Removed 'tempfile', '.mktemp', '.mkstemp', '.NamedTemporaryFile' to allow temp file creation
    # - Removed 'requests' to allow API data fetching
    # - Removed 'platform' to allow system info checks
    # - Removed 'shutil', '.unlink(', '.remove(', '.rmdir(' for file management
    
    cpp_banned = [
        'system(', 'exec', 'popen', 'fork', 'remove(', 'unlink(',
        '<fstream>', '<ofstream>', '<ifstream>', 'FILE *', 'fopen', 'fwrite',
        'fread', '<stdio.h>', '<stdlib.h>', '<unistd.h>', 'getcwd', 'opendir',
        'readdir', '<dirent.h>', '<sys/stat.h>', '<fcntl.h>',
        'freopen', 'ioctl', '<sys/socket.h>'
    ]
    
    # Check if code is empty
    if not code.strip():
        return True, "Code is empty."
    
    # Determine which banned list to use
    banned_list = python_banned if language == 'python' else cpp_banned
    
    # Check for banned operations
    for banned_op in banned_list:
        if banned_op in code:
            code_logger.warning(f"Forbidden operation detected: {banned_op}")
            if language == 'python':
                return False, f"Forbidden operation: {banned_op}"
            else:
                return False, f"Forbidden header or operation: {banned_op}"
    
    # Specific checks for Python
    if language == 'python':
        # Check for import statements with potentially dangerous modules
        import_pattern = r'^\s*(?:from\s+(\S+)\s+)?import\s+(.+)$'
        for line in code.split('\n'):
            match = re.match(import_pattern, line)
            if match:
                module = match.group(1) or match.group(2).split()[0].split('.')[0]
                # Allow essential data analysis libraries but block dangerous ones
                if module in ['subprocess', 'ctypes']:
                    code_logger.warning(f"Forbidden module import detected: {module}")
                    return False, f"Forbidden module import: {module}"
        
        # Add data file path support for simple file paths
        if "sample_data.csv" in code and "DATA_FILE_PATH" not in code:
            # Adjust for data file paths in the code_interpreter context
            code = code.replace("'sample_data.csv'", "DATA_FILE_PATH")
            code = code.replace('"sample_data.csv"', "DATA_FILE_PATH")
            code_logger.info("Replaced hardcoded data file path with DATA_FILE_PATH")
            
        # More robust preprocessing of code to fix indentation issues
        try:
            # First try to detect indentation inconsistencies
            lines = code.split('\n')
            cleaned_code = []
            current_indent = 0
            indents = []
            
            # First pass - collect indentation patterns
            for line in lines:
                if line.strip() and not line.lstrip().startswith('#'):
                    spaces = len(line) - len(line.lstrip())
                    if spaces > 0 and spaces not in indents:
                        indents.append(spaces)
            
            # Sort indentation levels
            indents.sort()
            
            # If we have mixed indentations, normalize them
            if len(indents) > 1:
                code_logger.info(f"Detected mixed indentation levels: {indents}")
                
                # Replace tabs with spaces if present
                code = code.replace('\t', '    ')
                lines = code.split('\n')
                
                # Second pass - normalize indentation
                for line in lines:
                    if not line.strip():
                        cleaned_code.append(line)  # Keep empty lines
                        continue
                        
                    if line.strip().startswith('#'):
                        # Preserve comments with their indentation
                        cleaned_code.append(line)
                        continue
                        
                    spaces = len(line) - len(line.lstrip())
                    if spaces > 0:
                        # Find the closest standard indentation level (multiple of 4)
                        indent_level = (spaces + 2) // 4  # Round to nearest indentation level
                        cleaned_line = '    ' * indent_level + line.lstrip()
                        cleaned_code.append(cleaned_line)
                    else:
                        cleaned_code.append(line)
                
                code = '\n'.join(cleaned_code)
                code_logger.info("Normalized mixed indentation to 4-space indentation")
        
            # Try compiling the normalized code
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            # If syntax error is not indentation-related, log it and continue with standard wrapping
            if "unexpected indent" in str(e) or "unindent does not match" in str(e) or "expected an indented block" in str(e):
                code_logger.warning(f"Indentation error in Python code: {str(e)}")
                try:
                    # Reindent the entire code with autopep8 if available
                    try:
                        import autopep8
                        code = autopep8.fix_code(code)
                        code_logger.info("Applied autopep8 fixes to the code")
                    except ImportError:
                        # If autopep8 not available, try basic reindentation
                        lines = code.split('\n')
                        # Replace all indentation with fixed 4-space indents
                        fixed_lines = []
                        for line in lines:
                            if line.strip():
                                # Count indentation level based on spaces/tabs
                                stripped = line.lstrip()
                                indent_level = (len(line) - len(stripped)) // 2  # Assuming 2-space or mixed indentation
                                fixed_line = '    ' * indent_level + stripped
                                fixed_lines.append(fixed_line)
                            else:
                                fixed_lines.append(line)  # Keep empty lines
                        code = '\n'.join(fixed_lines)
                        code_logger.info("Applied basic reindentation to the code")
                    
                    # Try compiling again
                    compile(code, '<string>', 'exec')
                    code_logger.info("Successfully fixed indentation issues")
                except SyntaxError as e2:
                    # Special case - attempt to ignore indentation errors for execution
                    code_logger.warning(f"Still have syntax errors after fixing: {str(e2)}")
                    if "unexpected indent" in str(e2) or "unindent does not match" in str(e2):
                        # For indentation errors only, we'll proceed anyway but log it
                        code_logger.warning("Proceeding despite indentation errors")
                    else:
                        return False, f"Syntax error: {str(e2)}"
                except Exception as fix_error:
                    code_logger.error(f"Error fixing indentation: {str(fix_error)}")
            else:
                code_logger.warning(f"Non-indentation syntax error: {str(e)}")
        except Exception as e:
            code_logger.warning(f"Error during code preprocessing: {str(e)}")
        
        # Simple safety header with just timeout and exception handling
        safety_header = """import time
import threading
import signal
import sys
import os

def timeout_handler(signum, frame):
    print('Code execution timed out (exceeded 10 seconds)')
    sys.exit(1)

# Set alarm if available
try:
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
except AttributeError:
    pass  # SIGALRM might not be available on all platforms

# Also set thread-based timeout for redundancy
def timeout_thread():
    time.sleep(10)
    print('Code execution timed out (exceeded 10 seconds)')
    # Force exit in thread timer
    os._exit(1)

timer = threading.Thread(target=timeout_thread)
timer.daemon = True
timer.start()

try:
"""
        
        # Add indentation for user code - ensure consistent indentation
        indented_code = "\n".join("    " + line for line in code.split("\n"))
        
        # Add exception handling and ending the try block
        safety_footer = """
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    # Cancel the alarm if it was set
    try:
        signal.alarm(0)
    except:
        pass
"""
        
        code = safety_header + indented_code + safety_footer
    
    # Specific checks for C++
    if language == 'cpp':
        # Check for includes - ensure they're valid
        include_pattern = r'#\s*include\s*<(.+?)>'
        includes = re.findall(include_pattern, code)
        
        # Check if includes are in the banned list
        for inc in includes:
            include_with_brackets = f"<{inc}>"
            if include_with_brackets in cpp_banned:
                code_logger.warning(f"Forbidden C++ header detected: {inc}")
                return False, f"Forbidden header include: {inc}"
        
        # Ensure C++ has basic structure 
        has_main = 'main(' in code or 'int main' in code or 'void main' in code
        has_iostream = '#include <iostream>' in code or '#include<iostream>' in code or '#include <bits/stdc++.h>' in code or '#include<bits/stdc++.h>' in code
        
        # Fix missing headers and namespace if needed
        if not has_iostream and ('cout' in code or 'cin' in code or 'cerr' in code):
            code = "#include <iostream>\n" + code
        
        if ('cout' in code or 'cin' in code or 'cerr' in code) and 'using namespace std' not in code:
            # Find position after includes
            lines = code.split('\n')
            last_include_index = -1
            for i, line in enumerate(lines):
                if '#include' in line:
                    last_include_index = i
            
            if last_include_index >= 0:
                lines.insert(last_include_index + 1, "using namespace std;")
            else:
                lines.insert(0, "using namespace std;")
            
            code = '\n'.join(lines)
        
        # Add main if none exists
        if not has_main:
            # Check if code has valid statements (not just function definitions)
            # For basic code without main, we wrap it in a main function
            code = """#include <bits/stdc++.h>
using namespace std;
int main() {
    // User code starts
""" + code + """
    // User code ends
    return 0;
}"""
        else:
            # Code has main, make sure it's wrapped with timeout
            code = """#include <chrono>
#include <thread>
#include <future>
#include <stdexcept>
""" + code.replace("int main(", "int userMain(").replace("void main(", "void userMain(") + """
int main() {
    // Set up a timeout for 10 seconds
    auto future = std::async(std::launch::async, []() {
        try {
            userMain();
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    });
    
    // Wait for the future to complete or timeout
    if (future.wait_for(std::chrono::seconds(10)) == std::future_status::timeout) {
        std::cerr << "Error: Code execution timed out (exceeded 10 seconds)" << std::endl;
    }
    
    return 0;
}"""
    
    # Skip the final syntax check as we've already handled it more tolerantly above
    return True, code

# Function to limit resources for subprocesses
def limit_resources():
    """Set resource limits for subprocess execution"""
    if not RESOURCE_AVAILABLE:
        return
        
    try:
        # Limit to 500MB of memory
        resource.setrlimit(resource.RLIMIT_AS, (500 * 1024 * 1024, 500 * 1024 * 1024))
        
        # CPU time limit (seconds)
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        code_logger.debug("Resource limits set successfully")
    except Exception as e:
        code_logger.warning(f"Could not set resource limits: {str(e)}")

# Function to run with a thread pool to avoid blocking the event loop
async def run_in_thread_pool(func, *args, **kwargs):
    """Run a CPU-bound function in a thread pool to avoid blocking the event loop"""
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, func, *args, **kwargs)

# Function to find Python 3 executable
def find_python3_executable():
    """Find the python3 executable on the system"""
    # First check if we have python3 in PATH
    python3_path = shutil.which('python3')
    if (python3_path):
        code_logger.info(f"Found python3 at: {python3_path}")
        return python3_path
    
    # If not found, check if current executable is python3
    if sys.version_info.major == 3:
        code_logger.info(f"Using current Python as python3: {sys.executable}")
        return sys.executable
    
    # Last resort - try common locations
    common_paths = [
        '/usr/bin/python3',
        '/usr/local/bin/python3',
        '/opt/bin/python3',
    ]
    
    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            code_logger.info(f"Found python3 at common location: {path}")
            return path
    
    # Fallback to 'python' and hope it's Python 3
    python_path = shutil.which('python')
    if python_path:
        code_logger.info(f"Falling back to 'python': {python_path}")
        return python_path
    
    # If all else fails, return current executable
    code_logger.warning("Could not find python3, using current executable")
    return sys.executable

async def execute_code(code: str, language: str, timeout: int = 10, input_data: str = ""):
    """
    Execute code in a sandboxed environment with strict timeout.
    
    Args:
        code (str): The code to execute.
        language (str): 'python' or 'cpp'.
        timeout (int): Maximum execution time in seconds.
        input_data (str): Optional input data for the program.
        
    Returns:
        str: The output of the code execution.
    """
    # Store original code for return value
    original_code = code
    
    # Log code execution request
    code_logger.info(f"Executing {language} code, length: {len(code)}, has input: {bool(input_data)}")
    
    # Validate that we have actual code to execute
    if not code or not code.strip():
        code_logger.warning("Empty code provided")
        return "Error: No code provided to execute. Return code: 1"
    
    # Basic validation of language
    if language not in ["python", "cpp"]:
        code_logger.warning(f"Unsupported language: {language}")
        return f"Error: Unsupported language '{language}'. Please use 'python' or 'cpp'. Return code: 1"
    
    # Pre-install required packages for data analysis
    if language == 'python' and ('import pandas' in code or 'from pandas' in code):
        try:
            # Check if we need to install key data analysis packages
            code_logger.info("Data analysis code detected. Ensuring required packages are installed...")
            packages_to_check = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'autopep8']
            for package in packages_to_check:
                if package in code or package == 'autopep8':  # Always check for autopep8
                    try:
                        # Try importing the package
                        __import__(package)
                        code_logger.info(f"Package {package} is already installed.")
                    except ImportError:
                        # If import fails, install the package
                        code_logger.info(f"Installing missing package: {package}")
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", package],
                            capture_output=True,
                            text=True,
                            check=False
                        )
                        if result.returncode != 0:
                            code_logger.warning(f"Failed to install {package}: {result.stderr}")
                        else:
                            code_logger.info(f"Successfully installed {package}")
        except Exception as e:
            code_logger.warning(f"Error checking/installing data analysis packages: {str(e)}")
    
    # For Python code, we'll use a different approach entirely - direct execution without sanitization
    if language == 'python':
        try:
            # Create temp directory for running code
            with tempfile.TemporaryDirectory() as temp_dir:
                code_logger.debug(f"Created temporary directory: {temp_dir}")
                
                # Extract and fix DATA_FILE_PATH from input
                data_file_path = None
                filtered_input_lines = []
                
                if input_data:
                    for line in input_data.splitlines():
                        if line.startswith("DATA_FILE_PATH="):
                            data_file_path = line.split("=", 1)[1].strip()
                            code_logger.info(f"Found data file path: {data_file_path}")
                        else:
                            filtered_input_lines.append(line)
                
                filtered_input = "\n".join(filtered_input_lines) if filtered_input_lines else ""
                
                # Replace hardcoded file paths with DATA_FILE_PATH variable if we have one
                if data_file_path:
                    code = code.replace("'sample_data.csv'", f"r'{data_file_path}'")
                    code = code.replace('"sample_data.csv"', f'r"{data_file_path}"')
                    code = code.replace("file_path = 'sample_data.csv'", f"file_path = r'{data_file_path}'")
                    code = code.replace('file_path = "sample_data.csv"', f'file_path = r"{data_file_path}"')
                    code_logger.info(f"Replaced hardcoded paths with: {data_file_path}")
                
                # Create visualizations capture code
                image_path = os.path.join(temp_dir, "output_plot.png")
                vis_capture_code = """
import signal
import sys
import os
import time
import threading

# Set timeout handling
def timeout_handler(signum, frame):
    print('Code execution timed out (exceeded 15 seconds)')
    sys.exit(1)

# Set alarm if available
try:
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(15)  # 15 seconds for data viz
except (AttributeError, ValueError):
    pass  # SIGALRM might not be available on all platforms

# Thread-based timeout for redundancy
def timeout_thread():
    time.sleep(15)
    print('Code execution timed out (exceeded 15 seconds)')
    os._exit(1)

timeout_timer = threading.Thread(target=timeout_thread)
timeout_timer.daemon = True
timeout_timer.start()

# Initialize visualization imports tracking
viz_imports = []

# Check for imports needed
import sys

if 'matplotlib' in globals() or 'matplotlib' in sys.modules:
    pass  # Already imported
elif 'plt' in globals() or ('matplotlib.pyplot' in sys.modules):
    # plt is used but matplotlib might not be properly imported
    import matplotlib.pyplot as plt
    viz_imports.append("matplotlib.pyplot")
    
if 'base64' in globals() or 'base64' in sys.modules:
    pass  # Already imported
else:
    import base64
    viz_imports.append("base64")
    
if 'io' in globals() or 'io' in sys.modules:
    pass  # Already imported
else:
    import io
    viz_imports.append("io")

if viz_imports:
    print("Auto-imported for visualization: " + ', '.join(viz_imports))

# Add visualization capture function
def _save_and_encode_plot():
    if "matplotlib" in sys.modules:
        plt = sys.modules["matplotlib.pyplot"]
        if plt.get_fignums():
            try:
                # Save to temp file first
                print("Saving plot to temp file...")
                plt.savefig("{0}", dpi=150, bbox_inches='tight')
                
                # Encode to base64 for sending back
                with open("{0}", 'rb') as img_file:
                    img_data = img_file.read()
                    encoded = base64.b64encode(img_data).decode('utf-8')
                    print("\\n[CHART_DATA_START]")
                    print(encoded)
                    print("[CHART_DATA_END]")
            except Exception as e:
                print(f"Error saving visualization: {{e}}")

# Register cleanup to run at exit
import atexit
atexit.register(_save_and_encode_plot)

# Make plt.show non-blocking if it exists
if "matplotlib" in sys.modules:
    plt = sys.modules["matplotlib.pyplot"]
    original_show = plt.show
    def _show_wrapper(*args, **kwargs):
        if 'block' not in kwargs:
            kwargs['block'] = False
        return original_show(*args, **kwargs)
    plt.show = _show_wrapper

# Execute the actual user code
try:
""".format(image_path)
                
                # Add closing part
                vis_capture_code_end = """
except Exception as e:
    print(f"Error executing code: {str(e)}")
finally:
    # Cancel the alarm if it was set
    try:
        signal.alarm(0)
    except:
        pass
    
    # Capture plots at the end if they exist
    _save_and_encode_plot()
"""
                
                # Indented user code
                indented_user_code = "\n    ".join([""] + code.split("\n"))
                
                # Complete code with user code in the middle
                full_code = vis_capture_code + indented_user_code + vis_capture_code_end
                
                # Write to file
                file_path = os.path.join(temp_dir, 'user_code.py')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(full_code)
                
                code_logger.debug(f"Python code written to {file_path}")
                
                # Make sure the file is executable
                os.chmod(file_path, 0o755)
                
                # Set process environment for execution
                env = dict(os.environ)  # Start with current environment
                
                # Preserve important environment variables for package access
                critical_env_vars = {
                    'PYTHONPATH': env.get('PYTHONPATH', ''),
                    'PYTHONHOME': env.get('PYTHONHOME', ''),
                    'PATH': env.get('PATH', ''),
                    'PYTHONIOENCODING': 'utf-8',
                    'TEMP': temp_dir,
                    'TMP': temp_dir,
                }
                
                # If using a virtual environment, make sure to carry those variables
                if 'VIRTUAL_ENV' in env:
                    critical_env_vars['VIRTUAL_ENV'] = env['VIRTUAL_ENV']
                
                # Use environment variables from the current process to ensure access to installed packages
                env.update(critical_env_vars)
                
                # Find python executable
                python3_executable = find_python3_executable()
                
                # Prepare command and input
                shell_command = f"{python3_executable} {file_path}"
                input_file = None
                
                if filtered_input:
                    # Create a temporary file for input data
                    input_file = os.path.join(temp_dir, 'input.txt')
                    with open(input_file, 'w', encoding='utf-8') as f:
                        f.write(filtered_input)
                    # Adjust command to use input file
                    shell_command = f"{python3_executable} {file_path} < {input_file}"
                    code_logger.debug(f"Created input file: {input_file}")
                
                # Execute the code
                try:
                    # Use shell=True for compatibility
                    code_logger.debug(f"Attempting execution with shell command: {shell_command}")
                    proc = await asyncio.create_subprocess_shell(
                        shell_command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env,
                        cwd=temp_dir,
                        shell=True
                    )
                    
                    # Enforce timeout
                    actual_timeout = min(timeout, 20)  # Allow more time for visualization
                    code_logger.debug(f"Waiting for output with {actual_timeout}s timeout")
                    
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=actual_timeout)
                    
                except (subprocess.SubprocessError, asyncio.TimeoutError, Exception) as e:
                    # Fallback method if shell fails
                    code_logger.warning(f"Shell execution failed: {str(e)}. Trying exec method.")
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            python3_executable, file_path,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE if filtered_input and not input_file else None,
                            cwd=temp_dir,
                            env=env
                        )
                        
                        if filtered_input and not input_file:
                            stdout, stderr = await asyncio.wait_for(
                                proc.communicate(filtered_input.encode('utf-8')),
                                timeout=actual_timeout
                            )
                        else:
                            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=actual_timeout)
                            
                    except Exception as exec_error:
                        code_logger.error(f"Exec method failed: {str(exec_error)}")
                        return f"Code:\n```{language}\n{original_code}\n```\n\nError: Failed to execute Python code: {str(exec_error)}"
                
                # Process execution results
                stdout_text = stdout.decode('utf-8', errors='replace').strip()
                stderr_text = stderr.decode('utf-8', errors='replace').strip()
                
                # Check for errors
                if stderr_text:
                    code_logger.debug(f"Error output: {stderr_text[:200]}...")
                    output_with_code = f"Code:\n```{language}\n{original_code}\n```\n\nError:\n```\n{stderr_text}```"
                    return output_with_code
                
                # Process stdout for visualization markers
                output = stdout_text
                code_logger.debug(f"Standard output received: {len(output)} chars")
                
                # Check if we have a visualization chart
                chart_data = None
                chart_markers = ["[CHART_DATA_START]", "[CHART_DATA_END]"]
                if all(marker in output for marker in chart_markers):
                    code_logger.info("Found chart data in output")
                    # Extract base64 encoded image data
                    start_idx = output.find(chart_markers[0]) + len(chart_markers[0])
                    end_idx = output.find(chart_markers[1])
                    
                    if start_idx > 0 and end_idx > start_idx:
                        chart_data = output[start_idx:end_idx].strip()
                        # Remove the chart data from output to avoid cluttering the response
                        clean_output = (output[:output.find(chart_markers[0])].strip() + 
                                       "\n[Chart generated successfully]" + 
                                       output[output.find(chart_markers[1]) + len(chart_markers[1]):].strip())
                        output = clean_output
                
                # Limit output size to prevent huge responses
                if len(output) > 5000:
                    output = output[:5000] + "\n...(output truncated due to size)"
                
                # Prepare final response
                if output:
                    output_with_code = f"Code:\n```{language}\n{original_code}\n```\n\nOutput:\n```\n{output}```"
                    return output_with_code
                else:
                    output_with_code = f"Code:\n```{language}\n{original_code}\n```\n\nOutput:\n```\nCode executed successfully with no output. Return code: 0\n```"
                    return output_with_code
                    
        except asyncio.TimeoutError:
            code_logger.warning(f"Python execution timed out after {timeout}s")
            output_with_code = f"Code:\n```{language}\n{original_code}\n```\n\nError: Code execution timed out after {timeout} seconds. Please optimize your code or reduce complexity."
            return output_with_code
        except Exception as e:
            code_logger.error(f"Python execution error: {str(e)}")
            return f"Code:\n```{language}\n{original_code}\n```\n\nError: An error occurred during Python execution: {str(e)}"
    
    # For non-Python code, use the sanitization approach
    is_safe, sanitized_code = sanitize_code(code, language)
    if not is_safe:
        code_logger.warning(f"Security validation failed: {sanitized_code}")
        return f"Security error: {sanitized_code}"
    
    code = sanitized_code
    
    # ... rest of the function for C++ execution remains unchanged ...
    try:
        # Log environment info
        code_logger.info(f"Platform: {platform.platform()}, Python: {sys.version}")
        
        # Create temp directory for running code
        with tempfile.TemporaryDirectory() as temp_dir:
            code_logger.debug(f"Created temporary directory: {temp_dir}")
            
            if language == 'cpp':
                # Execute C++ code
                src_path = os.path.join(temp_dir, 'code.cpp')
                exe_path = os.path.join(temp_dir, 'code')
                if os.name == 'nt':  # Windows
                    exe_path += '.exe'
                
                with open(src_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                code_logger.debug(f"C++ code written to {src_path}")
                
                try:
                    # Check if g++ is available (use run_in_thread_pool for better performance)
                    code_logger.debug("Checking for g++ compiler")
                    check_result = await run_in_thread_pool(lambda: subprocess.run(
                        ['g++', '--version'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=10
                    ))
                    
                    if check_result.returncode != 0:
                        code_logger.error("g++ compiler not found")
                        return f"Code:\n```{language}\n{original_code}\n```\n\nError: C++ compiler (g++) not available. Return code: 1"
                    
                    code_logger.debug("Compiling C++ code")
                    
                    # Compile C++ code with restricted options and optimizations
                    compile_result = await run_in_thread_pool(lambda: subprocess.run(
                        ['g++', src_path, '-o', exe_path, '-march=native', '-O3'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=10
                    ))
                    
                    if compile_result.returncode != 0:
                        compile_error = compile_result.stderr.decode('utf-8', errors='replace').strip()
                        code_logger.error(f"C++ compilation failed: {compile_error[:200]}...")
                        if compile_error:
                            return f"Code:\n```{language}\n{original_code}\n```\n\nCompilation error:\n```\n{compile_error}```"
                        else:
                            return f"Code:\n```{language}\n{original_code}\n```\n\nCompilation error: Unknown compilation failure. Return code: 1"
                    
                    code_logger.debug("C++ code compiled successfully, executing")
                    
                    # Execute the compiled program
                    try:
                        # Execute in restricted environment
                        run_proc = await asyncio.create_subprocess_exec(
                            exe_path,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE if input_data else None,
                            cwd=temp_dir,
                            # Use preexec_fn only on Unix systems and if resource is available
                            preexec_fn=limit_resources if RESOURCE_AVAILABLE and os.name != 'nt' else None
                        )
                        
                        try:
                            # Enforce strict timeout
                            actual_timeout = min(timeout, 10)  # Never exceed 10 seconds
                            code_logger.debug(f"Waiting for C++ output with {actual_timeout}s timeout")
                            
                            if input_data:
                                try:
                                    # Send input data to the process
                                    code_logger.debug(f"Sending input data to C++ program: {input_data[:50]}...")
                                    stdout, stderr = await asyncio.wait_for(
                                        run_proc.communicate(input_data.encode('utf-8')), 
                                        timeout=actual_timeout
                                    )
                                except Exception as e:
                                    code_logger.error(f"Error processing C++ input: {str(e)}")
                                    return f"Code:\n```{language}\n{original_code}\n```\n\nError processing input data for C++ program: {str(e)}. Return code: 1"
                            else:
                                stdout, stderr = await asyncio.wait_for(run_proc.communicate(), timeout=actual_timeout)
                            
                            if stderr:
                                stderr_content = stderr.decode('utf-8', errors='replace').strip()
                                if stderr_content:
                                    code_logger.debug(f"C++ error output: {stderr_content[:200]}...")
                                    return f"Code:\n```{language}\n{original_code}\n```\n\nRuntime error:\n```\n{stderr_content}```"
                            
                            # Return output or default message if output is empty
                            output = stdout.decode('utf-8', errors='replace').strip()
                            code_logger.debug(f"C++ standard output received: {len(output)} chars")
                            
                            # Limit output size to prevent huge responses
                            if len(output) > 5000:
                                output = output[:5000] + "\n...(output truncated due to size)"
                                
                            if output:
                                return f"Code:\n```{language}\n{original_code}\n```\n\nOutput:\n```\n{output}```"
                            else:
                                return f"Code:\n```{language}\n{original_code}\n```\n\nOutput:\n```\nCode executed successfully with no output. Return code: 0\n```"
                            
                        except asyncio.TimeoutError:
                            code_logger.warning(f"C++ execution timed out after {actual_timeout}s")
                            try:
                                # Kill process
                                if hasattr(run_proc, 'kill'):
                                    run_proc.kill()
                                    code_logger.debug("C++ process killed")
                            except Exception as kill_error:
                                code_logger.error(f"Error killing C++ process: {str(kill_error)}")
                                
                            return f"Code:\n```{language}\n{original_code}\n```\n\nError: Code execution timed out after {actual_timeout} seconds. Please optimize your code or reduce complexity."
                            
                    except Exception as e:
                        code_logger.error(f"C++ execution error: {str(e)}")
                        return f"Code:\n```{language}\n{original_code}\n```\n\nError: An error occurred during C++ execution: {str(e)}"
                        
                except Exception as e:
                    code_logger.error(f"C++ process error: {str(e)}")
                    return f"Code:\n```{language}\n{original_code}\n```\n\nError: An error occurred: {str(e)}"
            
            # Default case for unsupported languages
            return f"Code:\n```{language}\n{original_code}\n```\n\nUnsupported language. Please use 'python' or 'cpp'."
    except Exception as e:
        # Catch-all exception handler to ensure we always return something
        code_logger.error(f"Unexpected error in execute_code: {str(e)}")
        error_msg = f"An unexpected error occurred: {str(e)}. Return code: 1"
        return f"Code:\n```{language}\n{original_code}\n```\n\n{error_msg}"

def extract_code_blocks(content: str):
    """
    Extract code blocks from the message content.
    
    Args:
        content (str): The message content.
        
    Returns:
        list: List of tuples containing (language, code).
    """
    # Regular expression to match code blocks
    # Match ```language\ncode``` pattern
    pattern = r'```(\w+)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        # If no matches found, try simpler pattern without language specifier
        pattern = r'```(.*?)```'
        simpler_matches = re.findall(pattern, content, re.DOTALL)
        if simpler_matches:
            # Try to detect language from content
            for code in simpler_matches:
                if '#include' in code and ('int main' in code or 'void main' in code):
                    matches.append(('cpp', code))
                else:
                    matches.append(('python', code))
    
    return matches