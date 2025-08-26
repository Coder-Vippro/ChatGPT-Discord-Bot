"""
Secure Python code execution with persistent virtual environment and package management.
This module provides secure execution with persistent package storage but clean code execution.
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
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta

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

# Persistent environment configuration  
PACKAGE_CLEANUP_DAYS = 3  # Cleanup packages every 3 days
PERSISTENT_VENV_DIR = Path("/tmp/bot_code_executor")
PACKAGE_CACHE_FILE = PERSISTENT_VENV_DIR / "package_cache.json"

class PersistentPackageManager:
    """
    Manages a persistent virtual environment for packages while keeping code execution clean.
    Packages persist for 3 days, code files are cleaned up after each execution.
    """
    
    def __init__(self):
        self.venv_dir = PERSISTENT_VENV_DIR
        self.cache_file = PACKAGE_CACHE_FILE
        self.python_path = None
        self.pip_path = None
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup Python and pip executable paths."""
        if os.name == 'nt':  # Windows
            self.python_path = self.venv_dir / "Scripts" / "python.exe"
            self.pip_path = self.venv_dir / "Scripts" / "pip.exe"
        else:  # Unix/Linux
            self.python_path = self.venv_dir / "bin" / "python"
            self.pip_path = self.venv_dir / "bin" / "pip"
    
    def _load_package_cache(self) -> Dict[str, Any]:
        """Load package installation cache."""
        if not self.cache_file.exists():
            return {"packages": {}, "last_cleanup": None}
        
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load package cache: {e}")
            return {"packages": {}, "last_cleanup": None}
    
    def _save_package_cache(self, cache_data: Dict[str, Any]):
        """Save package installation cache."""
        try:
            self.venv_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save package cache: {e}")
    
    def _needs_cleanup(self) -> bool:
        """Check if package cleanup is needed (every 3 days)."""
        cache = self._load_package_cache()
        last_cleanup = cache.get("last_cleanup")
        
        if not last_cleanup:
            return True
            
        try:
            last_cleanup_date = datetime.fromisoformat(last_cleanup)
            return datetime.now() - last_cleanup_date > timedelta(days=PACKAGE_CLEANUP_DAYS)
        except Exception:
            return True
    
    async def ensure_venv_ready(self) -> bool:
        """Ensure the persistent virtual environment is ready."""
        try:
            # Check if cleanup is needed
            if self._needs_cleanup():
                logger.info("Performing periodic package cleanup...")
                await self._cleanup_packages()
                return True
            
            # Check if venv exists and is functional
            if not self.venv_dir.exists() or not self.python_path.exists():
                logger.info("Creating persistent virtual environment for packages...")
                await self._create_venv()
                return True
            
            # Test if venv is functional
            try:
                process = await asyncio.create_subprocess_exec(
                    str(self.python_path), "-c", "import sys; print('OK')",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0 or b'OK' not in stdout:
                    logger.info("Persistent venv is corrupted, recreating...")
                    await self._cleanup_packages()
                    return True
                    
            except Exception:
                logger.info("Persistent venv test failed, recreating...")
                await self._cleanup_packages()
                return True
            
            logger.debug("Using existing persistent virtual environment")
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring venv ready: {e}")
            return False
    
    async def _create_venv(self):
        """Create a fresh virtual environment."""
        try:
            # Remove existing venv if it exists
            if self.venv_dir.exists():
                shutil.rmtree(self.venv_dir)
            
            # Create new venv
            self.venv_dir.mkdir(parents=True, exist_ok=True)
            venv.create(str(self.venv_dir), with_pip=True, clear=True)
            
            # Initialize cache
            cache_data = {
                "packages": {},
                "last_cleanup": datetime.now().isoformat()
            }
            self._save_package_cache(cache_data)
            
            logger.info(f"Created fresh persistent venv at {self.venv_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create persistent venv: {e}")
            raise
    
    async def _cleanup_packages(self):
        """Cleanup and recreate the virtual environment."""
        try:
            logger.info("Cleaning up persistent virtual environment...")
            
            # Remove the entire venv directory
            if self.venv_dir.exists():
                shutil.rmtree(self.venv_dir)
            
            # Create fresh venv
            await self._create_venv()
            
            logger.info("Persistent virtual environment cleaned and recreated")
            
        except Exception as e:
            logger.error(f"Failed to cleanup packages: {e}")
            raise
    
    def is_package_installed(self, package: str) -> bool:
        """Check if a package is already installed in cache."""
        cache = self._load_package_cache()
        return package.lower() in cache.get("packages", {})
    
    def mark_package_installed(self, package: str):
        """Mark a package as installed in cache."""
        cache = self._load_package_cache()
        cache["packages"][package.lower()] = {
            "installed_at": datetime.now().isoformat(),
            "name": package
        }
        self._save_package_cache(cache)

# Global persistent package manager
package_manager = PersistentPackageManager()
class SecureExecutor:
    """
    Secure Python executor that uses persistent packages but cleans up code files.
    Each execution gets a clean temporary directory but reuses installed packages.
    """
    
    def __init__(self):
        self.temp_dir = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary directories (code files only)."""
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
    
    async def install_packages_persistent(self, packages: List[str]) -> Tuple[List[str], List[str]]:
        """
        Install packages in the persistent virtual environment.
        
        Args:
            packages: List of package names to install
            
        Returns:
            Tuple of (installed_packages, failed_packages)
        """
        installed = []
        failed = []
        
        # Ensure persistent venv is ready
        if not await package_manager.ensure_venv_ready():
            return [], packages
        
        for package in packages:
            # Validate package safety
            is_safe, reason = self.validate_package_safety(package)
            if not is_safe:
                logger.warning(f"Package '{package}' blocked: {reason}")
                failed.append(package)
                continue
            
            # Check if already installed
            if package_manager.is_package_installed(package):
                logger.debug(f"Package '{package}' already installed")
                installed.append(package)
                continue
            
            try:
                # Install package in the persistent virtual environment
                process = await asyncio.create_subprocess_exec(
                    str(package_manager.pip_path), "install", "--no-cache-dir", package,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
                    return_code = process.returncode
                    
                    if return_code == 0:
                        installed.append(package)
                        package_manager.mark_package_installed(package)
                        logger.info(f"Successfully installed package: {package}")
                    else:
                        failed.append(package)
                        logger.warning(f"Failed to install {package}: {stderr.decode()}")
                        
                except asyncio.TimeoutError:
                    # Kill the process if it times out
                    try:
                        process.kill()
                        await process.wait()
                    except:
                        pass
                    failed.append(package)
                    logger.warning(f"Installation timeout for package: {package}")
                    
            except Exception as e:
                failed.append(package)
                logger.warning(f"Error installing {package}: {e}")
        
        return installed, failed
    
    async def execute_code_secure(self, code: str, timeout: int) -> Dict[str, Any]:
        """
        Execute Python code using persistent packages but clean temporary directory.
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            
        Returns:
            Dict containing execution results
        """
        start_time = time.time()
        
        # Create temporary directory for code execution
        self.temp_dir = tempfile.mkdtemp(prefix="code_exec_")
        code_file = os.path.join(self.temp_dir, "code_to_execute.py")
        
        try:
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Execute code using persistent Python environment
            process = await asyncio.create_subprocess_exec(
                str(package_manager.python_path), code_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.temp_dir
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
            # Clean up code file (but keep packages in persistent venv)
            try:
                if os.path.exists(code_file):
                    os.remove(code_file)
            except Exception:
                pass  # Silent cleanup failure


async def execute_python_code(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute Python code using persistent packages but clean code execution.
    Packages persist for 3 days, code files are cleaned up after each execution.
    
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
            
            # Install packages in persistent environment (if any)
            installed_packages = []
            failed_packages = []
            if packages_to_install:
                installed_packages, failed_packages = await executor.install_packages_persistent(packages_to_install)
            
            # Prepare code with input data if provided
            if input_data:
                # Add input data as a variable in the code
                code_with_input = f"input_data = '''{input_data}'''\n\n{code}"
            else:
                code_with_input = code
            
            # Execute code using persistent packages
            result = await executor.execute_code_secure(code_with_input, timeout)
            
            # Add package installation info
            if installed_packages:
                result["installed_packages"] = installed_packages
                # Prepend package installation info to output
                if result.get("success"):
                    package_info = f"[Using packages: {', '.join(installed_packages)}]\n\n"
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


# Utility functions for package management
async def force_cleanup_packages():
    """Force cleanup of the persistent package environment."""
    logger.info("Forcing cleanup of persistent packages...")
    await package_manager._cleanup_packages()
    logger.info("Forced package cleanup completed")

def get_package_status() -> Dict[str, Any]:
    """Get status information about the persistent package environment."""
    cache = package_manager._load_package_cache()
    
    status = {
        "persistent_venv_exists": package_manager.venv_dir.exists(),
        "python_executable": str(package_manager.python_path),
        "pip_executable": str(package_manager.pip_path),
        "installed_packages": cache.get("packages", {}),
        "last_cleanup": cache.get("last_cleanup"),
        "needs_cleanup": package_manager._needs_cleanup(),
        "cleanup_interval_days": PACKAGE_CLEANUP_DAYS
    }
    
    return status


# Deprecated - keeping for backward compatibility
async def install_packages(packages: List[str]) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    Note: In the persistent system, packages are managed automatically.
    """
    return {
        "success": False,
        "installed": [],
        "failed": packages,
        "message": "Use install_packages parameter in execute_python_code instead"
    }
