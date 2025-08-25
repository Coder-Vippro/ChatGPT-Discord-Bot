"""
Secure Python code execution with complete isolation and package management.
This module provides a completely secure isolated execution environment.
"""

import os
import sys
import subprocess
import asyncio
import tempfile
import venv
import shutil
import time
import re
import logging
import traceback
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Configure logging - console only
logger = logging.getLogger('python_executor')
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

# Security and execution constants
EXECUTION_TIMEOUT = 30  # Default timeout in seconds
MAX_OUTPUT_SIZE = 50000  # Maximum output size in characters

class SecureExecutor:
    """
    Completely isolated Python executor with fresh virtual environments.
    Each execution gets a completely clean environment.
    """
    
    def __init__(self):
        self.temp_dir = None
        self.venv_path = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary directories and virtual environments."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {self.temp_dir}: {e}")
    
    def validate_code_security(self, code: str) -> Tuple[bool, str]:
        """
        Validate code for security threats.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_safe, message)
        """
        # Blocked imports (security-sensitive modules)
        unsafe_imports = [
            r'import\s+os\b', r'from\s+os\s+import', 
            r'import\s+subprocess\b', r'from\s+subprocess\s+import',
            r'import\s+sys\b', r'from\s+sys\s+import',
            r'import\s+shutil\b', r'from\s+shutil\s+import',
            r'import\s+socket\b', r'from\s+socket\s+import',
            r'import\s+urllib\b', r'from\s+urllib\s+import',
            r'import\s+requests\b', r'from\s+requests\s+import',
            r'import\s+pathlib\b', r'from\s+pathlib\s+import',
            r'__import__\s*\(', r'eval\s*\(', r'exec\s*\(',
            r'compile\s*\(', r'open\s*\('
        ]
        
        # Check for unsafe imports
        for pattern in unsafe_imports:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Blocked unsafe import/function: {pattern}"
        
        # Check for file system operations
        file_operations = [
            r'\.write\s*\(', r'\.read\s*\(', r'\.remove\s*\(',
            r'\.mkdir\s*\(', r'\.rmdir\s*\(', r'\.delete\s*\('
        ]
        
        for pattern in file_operations:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Blocked file operation: {pattern}"
        
        # Check for network operations
        network_patterns = [
            r'socket\s*\(', r'connect\s*\(', r'bind\s*\(',
            r'listen\s*\(', r'accept\s*\(', r'send\s*\(',
            r'recv\s*\(', r'http\w*\s*\(', r'ftp\w*\s*\('
        ]
        
        for pattern in network_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Blocked network operation: {pattern}"
        
        return True, "Code passed security validation"
    
    def create_clean_environment(self) -> Tuple[str, str, str]:
        """
        Create a completely clean virtual environment.
        
        Returns:
            Tuple of (venv_path, python_executable, pip_executable)
        """
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="secure_python_")
        self.venv_path = os.path.join(self.temp_dir, "venv")
        
        logger.info(f"Creating clean virtual environment at: {self.venv_path}")
        
        # Create virtual environment
        venv.create(self.venv_path, with_pip=True, clear=True)
        
        # Get paths to executables
        if os.name == 'nt':  # Windows
            python_path = os.path.join(self.venv_path, "Scripts", "python.exe")
            pip_path = os.path.join(self.venv_path, "Scripts", "pip.exe")
        else:  # Unix/Linux
            python_path = os.path.join(self.venv_path, "bin", "python")
            pip_path = os.path.join(self.venv_path, "bin", "pip")
        
        # Verify executables exist
        if not os.path.exists(python_path):
            raise RuntimeError(f"Python executable not found: {python_path}")
        if not os.path.exists(pip_path):
            raise RuntimeError(f"Pip executable not found: {pip_path}")
        
        logger.debug(f"Clean environment created - Python: {python_path}, Pip: {pip_path}")
        return self.venv_path, python_path, pip_path
    
    def validate_package_safety(self, package: str) -> Tuple[bool, str]:
        """
        Validate if a package is safe to install.
        
        Args:
            package: Package name to validate
            
        Returns:
            Tuple of (is_safe, reason)
        """
        package_lower = package.lower().strip()
        
        # Completely blocked packages
        blocked_packages = {
            'os', 'subprocess', 'sys', 'shutil', 'socket', 'urllib', 'requests',
            'paramiko', 'fabric', 'invoke', 'pexpect', 'ptyprocess',
            'cryptography', 'pycrypto', 'pyopenssl', 'psutil',
            'django', 'flask', 'tornado', 'twisted', 'aiohttp', 'fastapi',
            'sqlalchemy', 'psycopg2', 'mysql-connector', 'pymongo',
            'selenium', 'scrapy', 'beautifulsoup4', 'lxml', 'mechanize'
        }
        
        if package_lower in blocked_packages:
            return False, f"Package '{package}' is blocked for security reasons"
        
        # Check for suspicious patterns
        suspicious_patterns = ['exec', 'eval', 'compile', 'system', 'shell', 'cmd', 'hack', 'exploit']
        for pattern in suspicious_patterns:
            if pattern in package_lower:
                return False, f"Package name contains suspicious keyword: {pattern}"
        
        # Allowed safe packages for data science
        safe_packages = {
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'plotly', 'bokeh',
            'scipy', 'scikit-learn', 'sklearn', 'statsmodels',
            'pillow', 'opencv-python', 'imageio', 'skimage',
            'pytz', 'dateutil', 'arrow', 'pendulum',
            'pyyaml', 'toml', 'configparser', 'jsonschema',
            'tqdm', 'progressbar2', 'click', 'typer',
            'openpyxl', 'xlrd', 'xlwt', 'xlsxwriter',
            'sympy', 'networkx', 'igraph'
        }
        
        if package_lower in safe_packages:
            return True, f"Package '{package}' is pre-approved as safe"
        
        # For unknown packages, be restrictive
        return False, f"Package '{package}' is not in the approved safe list"
    
    async def install_packages_clean(self, packages: List[str], pip_path: str) -> Tuple[List[str], List[str]]:
        """
        Install packages in the clean virtual environment (async to prevent blocking).
        
        Args:
            packages: List of package names to install
            pip_path: Path to pip executable in the clean environment
            
        Returns:
            Tuple of (installed_packages, failed_packages)
        """
        installed = []
        failed = []
        
        for package in packages:
            # Validate package safety
            is_safe, reason = self.validate_package_safety(package)
            if not is_safe:
                failed.append(package)
                continue
            
            try:
                # Install package in the clean virtual environment using async subprocess
                process = await asyncio.create_subprocess_exec(
                    pip_path, "install", package,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.temp_dir
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
                    return_code = process.returncode
                    
                    if return_code == 0:
                        installed.append(package)
                    else:
                        failed.append(package)
                        
                except asyncio.TimeoutError:
                    # Kill the process if it times out
                    try:
                        process.kill()
                        await process.wait()
                    except:
                        pass
                    failed.append(package)
                    
            except Exception as e:
                failed.append(package)
        
        return installed, failed
    
    async def execute_code_secure(self, code: str, python_path: str, timeout: int) -> Dict[str, Any]:
        """
        Execute Python code in the completely isolated environment (async to prevent blocking).
        
        Args:
            code: Python code to execute
            python_path: Path to Python executable in clean environment
            timeout: Execution timeout in seconds
            
        Returns:
            Dict containing execution results
        """
        start_time = time.time()
        
        # Create code file in the isolated environment
        code_file = os.path.join(self.temp_dir, "code_to_execute.py")
        
        try:
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Execute code in completely isolated environment using async subprocess
            process = await asyncio.create_subprocess_exec(
                python_path, code_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.temp_dir,
                env={  # Minimal environment variables
                    'PATH': os.path.dirname(python_path),
                    'PYTHONPATH': '',
                    'PYTHONHOME': '',
                }
            )
            
            try:
                # Wait for process completion with timeout
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                return_code = process.returncode
                
                execution_time = time.time() - start_time
                
                # Process results
                output = stdout.decode('utf-8') if stdout else ""
                error_output = stderr.decode('utf-8') if stderr else ""
                
                # Truncate output if too large
                if len(output) > MAX_OUTPUT_SIZE:
                    output = output[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"
                
                if return_code == 0:
                    return {
                        "success": True,
                        "output": output,
                        "error": error_output if error_output else "",
                        "execution_time": execution_time,
                        "return_code": return_code
                    }
                else:
                    return {
                        "success": False,
                        "output": output,
                        "error": error_output,
                        "execution_time": execution_time,
                        "return_code": return_code
                    }
                    
            except asyncio.TimeoutError:
                # Kill the process if it times out
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass
                    
                return {
                    "success": False,
                    "output": "",
                    "error": f"Code execution timed out after {timeout} seconds",
                    "execution_time": timeout,
                    "return_code": -1
                }
                
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Execution error: {str(e)}"
            
            return {
                "success": False,
                "output": "",
                "error": error_msg,
                "execution_time": execution_time,
                "traceback": traceback.format_exc()
            }
        finally:
            # Clean up code file
            try:
                if os.path.exists(code_file):
                    os.remove(code_file)
            except Exception as e:
                pass  # Silent cleanup failure


async def execute_python_code(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute Python code in a completely clean, isolated environment.
    
    Args:
        args: Dictionary containing:
            - code: The Python code to execute
            - input_data: Optional input data for the code
            - install_packages: List of packages to install (will be validated for security)
            - timeout: Optional timeout in seconds (default: 30)
            
    Returns:
        Dict containing execution results
    """
    try:
        code = args.get("code", "")
        input_data = args.get("input_data", "")
        packages_to_install = args.get("install_packages", [])
        timeout = args.get("timeout", EXECUTION_TIMEOUT)
        
        if not code:
            return {
                "success": False,
                "error": "No code provided",
                "output": ""
            }
        
        with SecureExecutor() as executor:
            # Validate code security
            is_safe, safety_message = executor.validate_code_security(code)
            if not is_safe:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Security violation: {safety_message}",
                    "execution_time": 0
                }
            
            # Create completely clean environment
            venv_path, python_path, pip_path = executor.create_clean_environment()
            
            # Install only requested packages (if any)
            installed_packages = []
            failed_packages = []
            if packages_to_install:
                installed_packages, failed_packages = await executor.install_packages_clean(packages_to_install, pip_path)
            
            # Prepare code with input data if provided
            if input_data:
                # Add input data as a variable in the code
                code_with_input = f"input_data = '''{input_data}'''\n\n{code}"
            else:
                code_with_input = code
            
            # Execute code in clean environment
            result = await executor.execute_code_secure(code_with_input, python_path, timeout)
            
            # Add package installation info
            if installed_packages:
                result["installed_packages"] = installed_packages
                # Prepend package installation info to output
                if result.get("success"):
                    package_info = f"[Installed packages: {', '.join(installed_packages)}]\n\n"
                    result["output"] = package_info + result.get("output", "")
            
            if failed_packages:
                result["failed_packages"] = failed_packages
            
            return result
        
    except Exception as e:
        error_msg = f"Error in Python code execution: {str(e)}"
        return {
            "success": False,
            "error": error_msg,
            "output": "",
            "traceback": traceback.format_exc()
        }


# Deprecated - keeping for backward compatibility
async def install_packages(packages: List[str]) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    Note: In the new secure system, packages are installed per execution.
    """
    return {
        "success": False,
        "installed": [],
        "failed": packages,
        "message": "Use install_packages parameter in execute_python_code instead"
    }
