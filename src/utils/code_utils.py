import os
import sys
import re
import tempfile
import asyncio
import subprocess
from typing import Dict, Any, Optional

def sanitize_code(code: str, language: str):
    """
    Sanitize and validate code for security purposes.
    
    Args:
        code (str): The code to be sanitized.
        language (str): The programming language ('python' or 'cpp').
        
    Returns:
        tuple: (is_safe, sanitized_code or error_message)
    """
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
            if language == 'python':
                return False, f"Forbidden module import: {banned_op}"
            else:
                return False, f"Forbidden header include: {banned_op}"
    
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
                    return False, f"Forbidden module import: {module}"
        
        # Add safety header for Python
        safety_header = """
import signal
import time

def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out (exceeded 10 seconds)")

# Set a timeout of 10 seconds
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)

# Restrict __builtins__ to safe functions only
safe_builtins = {}
for k in ['abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes', 
          'chr', 'complex', 'dict', 'dir', 'divmod', 'enumerate', 'filter', 
          'float', 'format', 'frozenset', 'hash', 'hex', 'int', 'iter', 'len',
          'list', 'map', 'max', 'min', 'next', 'oct', 'ord', 'pow', 'print', 
          'range', 'repr', 'reversed', 'round', 'set', 'slice', 'sorted', 'str', 
          'sum', 'tuple', 'type', 'zip']:
    if k in __builtins__:
        safe_builtins[k] = __builtins__[k]

__builtins__ = safe_builtins

try:
"""
        # Add indentation for user code
        indented_code = "\n".join("    " + line for line in code.split("\n"))
        
        # Add exception handling and ending the try block
        safety_footer = """
except TimeoutError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    # Cancel the alarm
    signal.alarm(0)
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
            return False, f"Syntax error: {str(e)}"
    
    return True, code

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
    # Validate that we have actual code to execute
    if not code or not code.strip():
        return "Error: No code provided to execute. Return code: 1"
    
    # Basic validation of language
    if language not in ["python", "cpp"]:
        return f"Error: Unsupported language '{language}'. Please use 'python' or 'cpp'. Return code: 1"
    
    # Sanitize the code first
    is_safe, sanitized_code = sanitize_code(code, language)
    if not is_safe:
        return f"Security error: {sanitized_code}"
    
    code = sanitized_code
    
    # Validate and prepare input data
    if input_data and not isinstance(input_data, str):
        try:
            input_data = str(input_data)
        except Exception as e:
            return f"Error: Invalid input data - {str(e)}. Return code: 1"
    
    # Ensure input data ends with newline
    if input_data and not input_data.endswith('\n'):
        input_data += '\n'
    
    try:
        # Create temp directory for running code
        with tempfile.TemporaryDirectory() as temp_dir:
            if language == 'python':
                # Execute Python code
                file_path = os.path.join(temp_dir, 'code.py')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                try:
                    # Set process environment to restrict access to the system
                    env = {
                        'PYTHONPATH': '',  # Prevent access to installed Python modules
                        'PATH': '',  # Restrict access to system commands
                        'TEMP': temp_dir,  # Set temp directory to our controlled directory
                        'TMP': temp_dir,
                    }
                    
                    # Run the code in a subprocess with timeout
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, file_path,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.PIPE if input_data else None,
                        cwd=temp_dir,
                        env=env,
                        # Use preexec_fn only on Unix systems
                        preexec_fn=os.setpgrp if os.name != 'nt' else None
                    )
                    
                    try:
                        # Additional safety - use a shorter timeout than specified in the code
                        # to ensure our code terminates first
                        if input_data:
                            try:
                                # Send input data to the process
                                stdout, stderr = await asyncio.wait_for(
                                    proc.communicate(input_data.encode('utf-8')), 
                                    timeout=timeout
                                )
                            except Exception as e:
                                return f"Error processing input data: {str(e)}. Return code: 1"
                        else:
                            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                        
                        # Check for errors
                        if stderr:
                            stderr_content = stderr.decode('utf-8', errors='replace').strip()
                            if stderr_content:
                                return f"Error:\n```\n{stderr_content}```"
                        
                        # Return output or default message if output is empty
                        output = stdout.decode('utf-8', errors='replace').strip()
                        if output:
                            return f"Output:\n```\n{output}```"
                        else:
                            return "Output:\n```\nCode executed successfully with no output. Return code: 0\n```"
                        
                    except asyncio.TimeoutError:
                        try:
                            # Kill process differently depending on the OS
                            if os.name != 'nt':  # Unix-like systems
                                os.killpg(os.getpgid(proc.pid), 9)
                            else:  # Windows
                                proc.kill()
                        except:
                            pass
                        return "Code execution timed out after 10 seconds. Please optimize your code or reduce complexity."
                        
                except Exception as e:
                    return f"An error occurred during Python execution: {str(e)}"
                    
            elif language == 'cpp':
                # Execute C++ code
                src_path = os.path.join(temp_dir, 'code.cpp')
                exe_path = os.path.join(temp_dir, 'code')
                if os.name == 'nt':  # Windows
                    exe_path += '.exe'
                
                with open(src_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                try:
                    # Check if g++ is available
                    try:
                        check_proc = await asyncio.create_subprocess_exec(
                            'g++', '--version',
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        await check_proc.communicate()
                        if check_proc.returncode != 0:
                            return "Error: C++ compiler (g++) not available. Return code: 1"
                    except Exception:
                        return "Error: C++ compiler (g++) not available. Return code: 1"
                        
                    # Compile C++ code with restricted options
                    compile_proc = await asyncio.create_subprocess_exec(
                        'g++', src_path, '-o', exe_path, '-std=c++17',
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    compile_stdout, compile_stderr = await compile_proc.communicate()
                    
                    if compile_proc.returncode != 0:
                        compile_error = compile_stderr.decode('utf-8', errors='replace').strip()
                        if compile_error:
                            return f"Compilation error:\n```\n{compile_error}```"
                        else:
                            return "Compilation error: Unknown compilation failure. Return code: 1"
                    
                    # Execute the compiled program
                    try:
                        # Execute in restricted environment
                        run_proc = await asyncio.create_subprocess_exec(
                            exe_path,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE if input_data else None,
                            cwd=temp_dir,
                            # Use preexec_fn only on Unix systems
                            preexec_fn=os.setpgrp if os.name != 'nt' else None
                        )
                        
                        try:
                            # Enforce strict timeout
                            if input_data:
                                try:
                                    # Send input data to the process
                                    stdout, stderr = await asyncio.wait_for(
                                        run_proc.communicate(input_data.encode('utf-8')), 
                                        timeout=timeout
                                    )
                                except Exception as e:
                                    return f"Error processing input data for C++ program: {str(e)}. Return code: 1"
                            else:
                                stdout, stderr = await asyncio.wait_for(run_proc.communicate(), timeout=timeout)
                            
                            if stderr:
                                stderr_content = stderr.decode('utf-8', errors='replace').strip()
                                if stderr_content:
                                    return f"Runtime error:\n```\n{stderr_content}```"
                            
                            # Return output or default message if output is empty
                            output = stdout.decode('utf-8', errors='replace').strip()
                            if output:
                                return f"Output:\n```\n{output}```"
                            else:
                                return "Output:\n```\nCode executed successfully with no output. Return code: 0\n```"
                            
                        except asyncio.TimeoutError:
                            try:
                                # Kill process differently depending on the OS
                                if os.name != 'nt':  # Unix-like systems
                                    os.killpg(os.getpgid(run_proc.pid), 9)
                                else:  # Windows
                                    run_proc.kill()
                            except:
                                pass
                            return "Code execution timed out after 10 seconds. Please optimize your code or reduce complexity."
                            
                    except Exception as e:
                        return f"An error occurred during C++ execution: {str(e)}"
                        
                except Exception as e:
                    return f"An error occurred: {str(e)}"
            
            # Default case for unsupported languages
            return "Unsupported language. Please use 'python' or 'cpp'."
    except Exception as e:
        # Catch-all exception handler to ensure we always return something
        error_msg = f"An unexpected error occurred: {str(e)}. Return code: 1"
        return error_msg

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