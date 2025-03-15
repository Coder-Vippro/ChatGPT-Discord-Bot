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

# Import the webhook logger
from src.utils.webhook_logger import webhook_log_manager

# Configure logger for code execution
code_logger = logging.getLogger("code_execution")
code_logger.setLevel(logging.INFO)
if not code_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    code_logger.addHandler(handler)

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
        'os.system', 'subprocess', 'open(', '.open(', 'eval(', 'exec(', '__import__(',
        'importlib', '.read(', '.write(', 'shutil', '.unlink(', '.remove(', '.rmdir(',
        'socket', 'requests', 'urllib', 'curl', 'wget', '.chmod', '.chown',
        'os.path', 'pathlib', '__file__', '__builtins__._', 'file(', 'with open',
        'io.open', 'fileinput', 'tempfile', '.mktemp', '.mkstemp', '.NamedTemporaryFile',
        'shelve', 'dbm', 'sqlite3', 'pickle', 'marshal', '.loads(', '.dumps(',
        'getattr(', 'setattr(', 'delattr(', '__class__', '__bases__', '__subclasses__',
        '__globals__', '__getattribute__', '.mro(', 'ctypes', 'platform'
    ]
    
    cpp_banned = [
        'system(', 'exec', 'popen', 'fork', 'remove(', 'unlink(',
        '<fstream>', '<ofstream>', '<ifstream>', 'FILE *', 'fopen', 'fwrite',
        'fread', '<stdio.h>', '<stdlib.h>', '<unistd.h>', 'getcwd', 'opendir',
        'readdir', '<dirent.h>', '<sys/stat.h>', '<fcntl.h>',
        'freopen', 'ioctl', '<sys/socket.h>'
    ]
    
    # Allowed C++ headers
    cpp_allowed_headers = [
        '<iostream>', '<vector>', '<string>', '<algorithm>', '<cmath>', '<map>', '<unordered_map>', 
        '<set>', '<unordered_set>', '<queue>', '<stack>', '<deque>', '<list>', '<array>', 
        '<numeric>', '<utility>', '<tuple>', '<functional>', '<chrono>', '<thread>', '<future>', 
        '<mutex>', '<atomic>', '<memory>', '<limits>', '<exception>', '<stdexcept>', '<type_traits>', 
        '<random>', '<regex>', '<bitset>', '<complex>', '<initializer_list>', '<iomanip>',
        '<bits/stdc++.h>'  # Added support for bits/stdc++.h
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
                if module in ['os', 'subprocess', 'sys', 'shutil', 'socket', 'requests', 'io', 
                             'pathlib', 'glob', 'fnmatch', 'fileinput', 'linecache', 
                             'pickle', 'dbm', 'sqlite3', 'ctypes', 'platform']:
                    code_logger.warning(f"Forbidden module import detected: {module}")
                    return False, f"Forbidden module import: {module}"
        
        # Simple safety header with just timeout and exception handling
        # Avoid complex sandboxing that might fail in Alpine
        safety_header = """
# Simple timeout mechanism
import time
import threading
import signal
import sys

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
        
        # Add indentation for user code
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
    
    # Perform syntax check for languages
    if language == 'python':
        try:
            compile(code, '<string>', 'exec')
            return True, code
        except SyntaxError as e:
            code_logger.warning(f"Python syntax error: {str(e)}")
            return False, f"Syntax error: {str(e)}"
    
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
    if python3_path:
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
    
    # Sanitize the code first
    is_safe, sanitized_code = sanitize_code(code, language)
    if not is_safe:
        code_logger.warning(f"Security validation failed: {sanitized_code}")
        return f"Security error: {sanitized_code}"
    
    code = sanitized_code
    
    # Validate and prepare input data
    if input_data and not isinstance(input_data, str):
        try:
            input_data = str(input_data)
        except Exception as e:
            code_logger.error(f"Input data conversion failed: {str(e)}")
            return f"Error: Invalid input data - {str(e)}. Return code: 1"
    
    # Ensure input_data ends with newline
    if input_data and not input_data.endswith('\n'):
        input_data += '\n'
    
    try:
        # Log environment info
        code_logger.info(f"Platform: {platform.platform()}, Python: {sys.version}")
        
        # Create temp directory for running code
        with tempfile.TemporaryDirectory() as temp_dir:
            code_logger.debug(f"Created temporary directory: {temp_dir}")
            
            if language == 'python':
                # Find python3 executable
                python3_executable = find_python3_executable()
                code_logger.info(f"Using Python executable: {python3_executable}")
                
                # Execute Python code
                file_path = os.path.join(temp_dir, 'code.py')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                code_logger.debug(f"Python code written to {file_path}")
                
                # Make sure the file is executable
                os.chmod(file_path, 0o755)
                
                try:
                    # Set process environment to restrict access to the system but keep necessary paths
                    env = dict(os.environ)  # Start with current environment
                    
                    # Override specific variables for sandboxing
                    env.update({
                        'PYTHONIOENCODING': 'utf-8',  # Ensure proper encoding
                        'TEMP': temp_dir,  # Set temp directory to our controlled directory
                        'TMP': temp_dir,
                    })
                    
                    code_logger.debug(f"Starting Python subprocess with executable: {python3_executable}")
                    
                    # Try using direct shell command first for better compatibility in Alpine
                    shell_command = f"{python3_executable} {file_path}"
                    
                    if input_data:
                        # Create a temporary file for input data
                        input_file = os.path.join(temp_dir, 'input.txt')
                        with open(input_file, 'w', encoding='utf-8') as f:
                            f.write(input_data)
                        # Adjust command to use input file
                        shell_command = f"{python3_executable} {file_path} < {input_file}"
                        code_logger.debug(f"Created input file: {input_file}")
                    
                    # Try different methods if one fails
                    try:
                        # Method 1: Use shell=True for Alpine compatibility
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
                        actual_timeout = min(timeout, 10)
                        code_logger.debug(f"Waiting for output with {actual_timeout}s timeout (shell method)")
                        
                        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=actual_timeout)
                        
                    except (subprocess.SubprocessError, asyncio.TimeoutError, Exception) as e:
                        # Method 2: Use create_subprocess_exec as fallback
                        code_logger.warning(f"Shell execution failed with {type(e).__name__}: {str(e)}. Trying exec method.")
                        try:
                            # Run the code with explicit executable path
                            proc = await asyncio.create_subprocess_exec(
                                python3_executable, file_path,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                stdin=subprocess.PIPE if input_data else None,
                                cwd=temp_dir,
                                env=env
                            )
                            
                            # Enforce timeout
                            actual_timeout = min(timeout, 10)
                            code_logger.debug(f"Waiting for output with {actual_timeout}s timeout (exec method)")
                            
                            if input_data:
                                stdout, stderr = await asyncio.wait_for(
                                    proc.communicate(input_data.encode('utf-8')),
                                    timeout=actual_timeout
                                )
                            else:
                                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=actual_timeout)
                                
                        except Exception as exec_error:
                            code_logger.error(f"Exec method failed: {str(exec_error)}")
                            return f"Code:\n```{language}\n{original_code}\n```\n\nError: Failed to execute Python code: {str(exec_error)}"
                    
                    # Process execution results
                    if stderr:
                        stderr_content = stderr.decode('utf-8', errors='replace').strip()
                        if stderr_content:
                            code_logger.debug(f"Error output: {stderr_content[:200]}...")
                            output_with_code = f"Code:\n```{language}\n{original_code}\n```\n\nError:\n```\n{stderr_content}```"
                            return output_with_code
                    
                    # Return output or default message if output is empty
                    output = stdout.decode('utf-8', errors='replace').strip()
                    code_logger.debug(f"Standard output received: {len(output)} chars")
                    
                    # Limit output size to prevent huge responses
                    if len(output) > 5000:
                        output = output[:5000] + "\n...(output truncated due to size)"
                    
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
                    
            elif language == 'cpp':
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