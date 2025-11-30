"""
Unified Code Interpreter

Provides secure Python code execution with file management (48-hour expiration).
Similar functionality to ChatGPT or Claude code interpreter.
"""

import os
import sys
import io
import re
import json
import logging
import asyncio
import subprocess
import tempfile
import time
import uuid
import shutil
import traceback
import contextlib
import venv
import ast
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger('code_interpreter')
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

# Constants
# Get CODE_EXECUTION_TIMEOUT from environment (default: 300 seconds = 5 minutes)
# This timeout applies ONLY to actual code execution, not env setup or package installation
CODE_EXECUTION_TIMEOUT = int(os.getenv('CODE_EXECUTION_TIMEOUT', '300'))
EXECUTION_TIMEOUT = CODE_EXECUTION_TIMEOUT  # Backward compatibility
MAX_OUTPUT_SIZE = 100000
# Get file expiration from environment (-1 means never expire)
FILE_EXPIRATION_HOURS = int(os.getenv('FILE_EXPIRATION_HOURS', '48'))
PACKAGE_CLEANUP_DAYS = 7
MAX_FILE_SIZE = 50 * 1024 * 1024

# Directory structure
BASE_DIR = Path("/tmp/bot_code_interpreter")
PERSISTENT_VENV_DIR = BASE_DIR / "venv"
USER_FILES_DIR = BASE_DIR / "user_files"
OUTPUT_DIR = BASE_DIR / "outputs"
PACKAGE_CACHE_FILE = BASE_DIR / "package_cache.json"

# Ensure directories exist
for dir_path in [BASE_DIR, USER_FILES_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Approved safe packages
APPROVED_PACKAGES = {
    'numpy', 'pandas', 'scipy', 'scikit-learn', 'sklearn', 'statsmodels',
    'matplotlib', 'seaborn', 'plotly', 'bokeh', 'altair', 'holoviews',
    'pillow', 'pil', 'imageio', 'scikit-image', 'skimage',
    'pytz', 'python-dateutil', 'arrow', 'pendulum',
    'pyyaml', 'toml', 'jsonschema', 'openpyxl', 'xlrd', 'xlwt', 'xlsxwriter',
    'tqdm', 'progressbar2', 'rich',
    'sympy', 'networkx', 'igraph', 'numba',
    'tensorflow', 'keras', 'torch', 'pytorch', 'xgboost', 'lightgbm', 'catboost',
    'nltk', 'spacy', 'gensim', 'wordcloud',
    'more-itertools', 'toolz', 'cytoolz', 'funcy'
}

# Blocked patterns - Comprehensive security checks
# Note: We allow open() for writing to enable saving plots and outputs
# The sandboxed environment restricts file access to safe directories
BLOCKED_PATTERNS = [
    # ==================== DANGEROUS SYSTEM MODULES ====================
    # OS module (except path)
    r'import\s+os\b(?!\s*\.path)',
    r'from\s+os\s+import\s+(?!path)',
    
    # File system modules
    r'import\s+shutil\b',
    r'from\s+shutil\s+import',
    r'import\s+pathlib\b(?!\s*\.)',  # Allow pathlib usage but monitor
    
    # Subprocess and execution modules
    r'import\s+subprocess\b',
    r'from\s+subprocess\s+import',
    r'import\s+multiprocessing\b',
    r'from\s+multiprocessing\s+import',
    r'import\s+threading\b',
    r'from\s+threading\s+import',
    r'import\s+concurrent\b',
    r'from\s+concurrent\s+import',
    
    # System access modules
    r'import\s+sys\b(?!\s*\.(?:path|version|platform|stdout|stderr))',
    r'from\s+sys\s+import\s+(?!path|version|platform|stdout|stderr)',
    r'import\s+platform\b',
    r'from\s+platform\s+import',
    r'import\s+ctypes\b',
    r'from\s+ctypes\s+import',
    r'import\s+_[a-z]+',  # Block private C modules
    
    # ==================== NETWORK MODULES ====================
    r'import\s+socket\b',
    r'from\s+socket\s+import',
    r'import\s+urllib\b',
    r'from\s+urllib\s+import',
    r'import\s+requests\b',
    r'from\s+requests\s+import',
    r'import\s+aiohttp\b',
    r'from\s+aiohttp\s+import',
    r'import\s+httpx\b',
    r'from\s+httpx\s+import',
    r'import\s+http\.client\b',
    r'from\s+http\.client\s+import',
    r'import\s+ftplib\b',
    r'from\s+ftplib\s+import',
    r'import\s+smtplib\b',
    r'from\s+smtplib\s+import',
    r'import\s+telnetlib\b',
    r'from\s+telnetlib\s+import',
    r'import\s+ssl\b',
    r'from\s+ssl\s+import',
    r'import\s+paramiko\b',
    r'from\s+paramiko\s+import',
    
    # ==================== DANGEROUS CODE EXECUTION ====================
    r'__import__\s*\(',
    r'\beval\s*\(',
    r'\bexec\s*\(',
    r'\bcompile\s*\(',
    r'\bglobals\s*\(',
    r'\blocals\s*\(',
    r'\bgetattr\s*\([^,]+,\s*[\'"]__',  # Block getattr for dunder methods
    r'\bsetattr\s*\([^,]+,\s*[\'"]__',  # Block setattr for dunder methods
    r'\bdelattr\s*\([^,]+,\s*[\'"]__',  # Block delattr for dunder methods
    r'\.\_\_\w+\_\_',  # Block dunder method access
    
    # ==================== FILE SYSTEM OPERATIONS ====================
    r'\.unlink\s*\(',
    r'\.rmdir\s*\(',
    r'\.remove\s*\(',
    r'\.chmod\s*\(',
    r'\.chown\s*\(',
    r'\.rmtree\s*\(',
    r'\.rename\s*\(',
    r'\.replace\s*\(',
    r'\.makedirs\s*\(',  # Allow mkdir but block makedirs outside sandbox
    r'Path\s*\(\s*[\'"]\/(?!tmp)',  # Block absolute paths outside /tmp
    r'open\s*\(\s*[\'"]\/(?!tmp)',  # Block file access outside /tmp
    
    # ==================== PICKLE AND SERIALIZATION ====================
    r'pickle\.loads?\s*\(',
    r'cPickle\.loads?\s*\(',
    r'marshal\.loads?\s*\(',
    r'shelve\.open\s*\(',
    
    # ==================== PROCESS MANIPULATION ====================
    r'os\.system\s*\(',
    r'os\.popen\s*\(',
    r'os\.spawn',
    r'os\.exec',
    r'os\.fork\s*\(',
    r'os\.kill\s*\(',
    r'os\.killpg\s*\(',
    
    # ==================== ENVIRONMENT ACCESS ====================
    r'os\.environ',
    r'os\.getenv\s*\(',
    r'os\.putenv\s*\(',
    
    # ==================== DANGEROUS BUILTINS ====================
    r'__builtins__',
    r'__loader__',
    r'__spec__',
    
    # ==================== CODE OBJECT MANIPULATION ====================
    r'\.f_code',
    r'\.f_globals',
    r'\.f_locals',
    r'\.gi_frame',
    r'\.co_code',
    r'types\.CodeType',
    r'types\.FunctionType',
    
    # ==================== IMPORT SYSTEM MANIPULATION ====================
    r'import\s+importlib\b',
    r'from\s+importlib\s+import',
    r'sys\.modules',
    r'sys\.path\.(?:append|insert|extend)',
    
    # ==================== MEMORY OPERATIONS ====================
    r'gc\.',
    r'sys\.getsizeof',
    r'sys\.getrefcount',
    r'id\s*\(',  # Block id() which can leak memory addresses
]

# Additional patterns that log warnings but don't block
WARNING_PATTERNS = [
    (r'while\s+True', "Infinite loop detected - ensure break condition exists"),
    (r'for\s+\w+\s+in\s+range\s*\(\s*\d{6,}', "Very large loop detected"),
    (r'recursion', "Recursion detected - ensure base case exists"),
]


class FileManager:
    """Manages user files with 48-hour expiration."""
    
    def __init__(self, db_handler=None):
        self.db = db_handler
        self.user_files_dir = USER_FILES_DIR
    
    async def save_file(self, user_id: int, file_data: bytes, filename: str, 
                       file_type: str = None) -> Dict[str, Any]:
        """Save a user-uploaded file with metadata."""
        try:
            file_id = f"{user_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            _, ext = os.path.splitext(filename)
            if not ext:
                ext = '.txt'
            
            user_dir = self.user_files_dir / str(user_id)
            user_dir.mkdir(parents=True, exist_ok=True)
            file_path = user_dir / f"{file_id}{ext}"
            
            if len(file_data) > MAX_FILE_SIZE:
                return {
                    "success": False,
                    "error": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
                }
            
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            # Calculate expiration time (-1 means never expire)
            if FILE_EXPIRATION_HOURS == -1:
                expires_at = None  # Never expire
            else:
                expires_at = (datetime.now() + timedelta(hours=FILE_EXPIRATION_HOURS)).isoformat()
            
            metadata = {
                "file_id": file_id,
                "user_id": user_id,
                "filename": filename,
                "file_path": str(file_path),
                "file_size": len(file_data),
                "file_type": file_type or self._detect_file_type(filename),
                "uploaded_at": datetime.now().isoformat(),
                "expires_at": expires_at
            }
            
            if self.db:
                # Use update_one with upsert to avoid duplicate key errors
                await self.db.db.user_files.update_one(
                    {"file_id": file_id},
                    {"$set": metadata},
                    upsert=True
                )
                logger.info(f"[DEBUG] Saved file metadata to database: {file_id}")
            
            expiration_msg = "never expires" if FILE_EXPIRATION_HOURS == -1 else f"expires in {FILE_EXPIRATION_HOURS}h"
            logger.info(f"Saved file {filename} for user {user_id}: {file_id} ({expiration_msg})")
            
            return {
                "success": True,
                "file_id": file_id,
                "file_path": str(file_path),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return {"success": False, "error": str(e)}
    
    def _detect_file_type(self, filename: str) -> str:
        """Detect file type from extension."""
        ext = os.path.splitext(filename)[1].lower()
        type_map = {
            # Data formats
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.xlsm': 'excel',
            '.xlsb': 'excel',
            '.ods': 'spreadsheet',
            '.json': 'json',
            '.jsonl': 'jsonl',
            '.ndjson': 'jsonl',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.cfg': 'config',
            '.conf': 'config',
            
            # Database formats
            '.db': 'database',
            '.sqlite': 'database',
            '.sqlite3': 'database',
            '.sql': 'sql',
            
            # Text and documents
            '.txt': 'text',
            '.md': 'markdown',
            '.markdown': 'markdown',
            '.rst': 'restructuredtext',
            '.rtf': 'richtext',
            '.doc': 'word',
            '.docx': 'word',
            '.odt': 'document',
            '.pdf': 'pdf',
            
            # Data science formats
            '.parquet': 'parquet',
            '.feather': 'feather',
            '.hdf': 'hdf5',
            '.hdf5': 'hdf5',
            '.h5': 'hdf5',
            '.pickle': 'pickle',
            '.pkl': 'pickle',
            '.joblib': 'joblib',
            '.npy': 'numpy',
            '.npz': 'numpy',
            '.mat': 'matlab',
            '.sav': 'spss',
            '.dta': 'stata',
            '.sas7bdat': 'sas',
            
            # Images
            '.png': 'image',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.gif': 'image',
            '.bmp': 'image',
            '.tiff': 'image',
            '.tif': 'image',
            '.webp': 'image',
            '.svg': 'image',
            '.ico': 'image',
            
            # Programming and code
            '.py': 'python',
            '.pyw': 'python',
            '.ipynb': 'jupyter',
            '.r': 'r',
            '.R': 'r',
            '.rmd': 'rmarkdown',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'header',
            '.hpp': 'header',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.m': 'matlab',
            
            # Web formats
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.less': 'less',
            
            # Compressed files
            '.zip': 'archive',
            '.tar': 'archive',
            '.gz': 'archive',
            '.bz2': 'archive',
            '.xz': 'archive',
            '.7z': 'archive',
            '.rar': 'archive',
            
            # Log and system files
            '.log': 'log',
            '.out': 'output',
            '.err': 'error',
            
            # Geospatial formats
            '.geojson': 'geojson',
            '.shp': 'shapefile',
            '.shx': 'shapefile',
            '.dbf': 'shapefile',
            '.kml': 'kml',
            '.kmz': 'kml',
            '.gpx': 'gpx',
            '.gml': 'gml',
            
            # Audio formats (for metadata/waveform analysis)
            '.mp3': 'audio',
            '.wav': 'audio',
            '.flac': 'audio',
            '.ogg': 'audio',
            '.aac': 'audio',
            '.m4a': 'audio',
            '.wma': 'audio',
            '.opus': 'audio',
            '.aiff': 'audio',
            
            # Video formats (for metadata/frame analysis)
            '.mp4': 'video',
            '.avi': 'video',
            '.mkv': 'video',
            '.mov': 'video',
            '.wmv': 'video',
            '.flv': 'video',
            '.webm': 'video',
            '.m4v': 'video',
            '.mpg': 'video',
            '.mpeg': 'video',
            
            # Additional scientific formats
            '.fits': 'fits',
            '.fts': 'fits',
            '.dicom': 'dicom',
            '.dcm': 'dicom',
            '.nii': 'nifti',
            '.vtk': 'vtk',
            '.stl': 'stl',
            '.obj': '3dmodel',
            '.ply': '3dmodel',
            
            # Additional data formats
            '.avro': 'avro',
            '.orc': 'orc',
            '.protobuf': 'protobuf',
            '.pb': 'protobuf',
            '.msgpack': 'msgpack',
            '.bson': 'bson',
            '.arrow': 'arrow',
            '.rda': 'rdata',
            '.rds': 'rdata',
            '.xpt': 'sas',
            
            # Additional compressed formats
            '.tgz': 'archive',
            '.tbz': 'archive',
            '.lz': 'archive',
            '.lzma': 'archive',
            '.zst': 'archive',
            
            # Additional programming languages
            '.sh': 'shell',
            '.bash': 'shell',
            '.zsh': 'shell',
            '.ps1': 'powershell',
            '.lua': 'lua',
            '.jl': 'julia',
            '.nim': 'nim',
            '.asm': 'assembly',
            '.s': 'assembly',
            
            # Additional document formats
            '.epub': 'ebook',
            '.mobi': 'ebook',
            '.tex': 'latex',
            '.adoc': 'asciidoc',
            '.org': 'org',
            
            # Additional web formats
            '.vue': 'vue',
            '.svelte': 'svelte',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            
            # Binary/Unknown - will be handled as binary by Python
            '.bin': 'binary',
            '.dat': 'binary',
        }
        return type_map.get(ext, 'binary')  # Default to 'binary' for unknown types
    
    async def get_user_files(self, user_id: int, include_expired: bool = False) -> List[Dict[str, Any]]:
        """Get all files for a user."""
        if not self.db:
            return []
        try:
            query = {"user_id": user_id}
            if not include_expired and FILE_EXPIRATION_HOURS != -1:
                # Only check expiration if files expire
                query["expires_at"] = {"$gt": datetime.now().isoformat()}
            files = await self.db.db.user_files.find(query).to_list(length=100)
            return files
        except Exception as e:
            logger.error(f"Error getting user files: {e}")
            return []
    
    async def get_file(self, file_id: str, user_id: int = None) -> Optional[Dict[str, Any]]:
        """Get file metadata by ID."""
        if not self.db:
            return None
        try:
            query = {"file_id": file_id}
            if user_id:
                query["user_id"] = user_id
            file_meta = await self.db.db.user_files.find_one(query)
            if file_meta:
                # Check expiration only if files expire (FILE_EXPIRATION_HOURS != -1)
                expires_at = file_meta.get('expires_at')
                if expires_at and FILE_EXPIRATION_HOURS != -1:
                    expires_dt = datetime.fromisoformat(expires_at)
                    if datetime.now() > expires_dt:
                        logger.info(f"File {file_id} has expired")
                        return None
            return file_meta
        except Exception as e:
            logger.error(f"Error getting file: {e}")
            return None
    
    async def delete_file(self, file_id: str, user_id: int = None) -> bool:
        """Delete a file and its metadata."""
        try:
            file_meta = await self.get_file(file_id, user_id)
            if not file_meta:
                return False
            file_path = Path(file_meta['file_path'])
            if file_path.exists():
                file_path.unlink()
            if self.db:
                await self.db.db.user_files.delete_one({"file_id": file_id})
            logger.info(f"Deleted file {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    async def cleanup_expired_files(self) -> int:
        """Clean up files that have expired. Skip if FILE_EXPIRATION_HOURS = -1 (permanent storage)."""
        try:
            # Skip cleanup if files never expire
            if FILE_EXPIRATION_HOURS == -1:
                logger.debug("Skipping file cleanup: FILE_EXPIRATION_HOURS = -1 (permanent storage)")
                return 0
            
            current_time = datetime.now().isoformat()
            if self.db:
                expired_files = await self.db.db.user_files.find({
                    "expires_at": {"$lt": current_time}
                }).to_list(length=1000)
            else:
                expired_files = []
            
            deleted_count = 0
            for file_meta in expired_files:
                file_path = Path(file_meta['file_path'])
                if file_path.exists():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
                if self.db:
                    await self.db.db.user_files.delete_one({"file_id": file_meta['file_id']})
            
            # Clean up empty user directories
            for user_dir in self.user_files_dir.iterdir():
                if user_dir.is_dir():
                    try:
                        if not any(user_dir.iterdir()):
                            user_dir.rmdir()
                    except Exception:
                        pass
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired files")
            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up expired files: {e}")
            return 0


class PackageManager:
    """Manages persistent virtual environment and packages."""
    
    def __init__(self):
        self.venv_dir = PERSISTENT_VENV_DIR
        self.cache_file = PACKAGE_CACHE_FILE
        self.python_path = None
        self.pip_path = None
        self.is_docker = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup Python and pip executable paths."""
        # In Docker, use system Python directly (no venv needed)
        if self.is_docker:
            self.python_path = Path(sys.executable)
            self.pip_path = Path(sys.executable).parent / "pip"
            logger.info(f"Docker detected - using system Python: {self.python_path}")
        elif os.name == 'nt':
            self.python_path = self.venv_dir / "Scripts" / "python.exe"
            self.pip_path = self.venv_dir / "Scripts" / "pip.exe"
        else:
            self.python_path = self.venv_dir / "bin" / "python"
            self.pip_path = self.venv_dir / "bin" / "pip"
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load package installation cache."""
        if not self.cache_file.exists():
            return {"packages": {}, "last_cleanup": None}
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {"packages": {}, "last_cleanup": None}
    
    def _save_cache(self, cache_data: Dict[str, Any]):
        """Save package installation cache."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _needs_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        cache = self._load_cache()
        last_cleanup = cache.get("last_cleanup")
        if not last_cleanup:
            return True
        try:
            last_date = datetime.fromisoformat(last_cleanup)
            return datetime.now() - last_date > timedelta(days=PACKAGE_CLEANUP_DAYS)
        except Exception:
            return True
    
    async def _cleanup_old_packages(self):
        """Clean up packages not used in PACKAGE_CLEANUP_DAYS days."""
        try:
            cache = self._load_cache()
            packages = cache.get("packages", {})
            current_time = datetime.now()
            packages_to_remove = []
            
            # Find packages older than PACKAGE_CLEANUP_DAYS
            for package_lower, info in packages.items():
                last_used = info.get("last_used", info.get("installed_at"))
                if last_used:
                    try:
                        last_used_date = datetime.fromisoformat(last_used)
                        days_unused = (current_time - last_used_date).days
                        if days_unused > PACKAGE_CLEANUP_DAYS:
                            packages_to_remove.append(info.get("name", package_lower))
                    except Exception as e:
                        logger.warning(f"Error parsing date for {package_lower}: {e}")
            
            if packages_to_remove:
                logger.info(f"Found {len(packages_to_remove)} packages unused for >{PACKAGE_CLEANUP_DAYS} days: {packages_to_remove}")
                
                # In Docker: Uninstall individual packages from system
                if self.is_docker:
                    successfully_removed = []
                    for package in packages_to_remove:
                        try:
                            logger.info(f"Uninstalling unused package from system: {package}")
                            process = await asyncio.create_subprocess_exec(
                                str(self.pip_path), "uninstall", "-y", package,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE
                            )
                            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
                            
                            if process.returncode == 0:
                                logger.info(f"✓ Successfully uninstalled: {package}")
                                successfully_removed.append(package.lower())
                            else:
                                error_msg = stderr.decode()
                                logger.warning(f"Failed to uninstall {package}: {error_msg}")
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout uninstalling {package}")
                        except Exception as e:
                            logger.error(f"Error uninstalling {package}: {e}")
                    
                    # Remove successfully uninstalled packages from cache
                    for package_lower in successfully_removed:
                        cache["packages"].pop(package_lower, None)
                    
                    cache["last_cleanup"] = current_time.isoformat()
                    self._save_cache(cache)
                    logger.info(f"Docker cleanup completed. Removed {len(successfully_removed)}/{len(packages_to_remove)} packages.")
                else:
                    # In non-Docker: Recreate entire venv (existing behavior)
                    logger.info("Non-Docker environment: recreating venv for cleanup")
                    await self._recreate_venv()
            else:
                logger.info("No old packages to clean up")
                # Update cleanup timestamp even if nothing to clean
                cache["last_cleanup"] = current_time.isoformat()
                self._save_cache(cache)
                
        except Exception as e:
            logger.error(f"Error during package cleanup: {e}")
    
    async def ensure_venv_ready(self) -> bool:
        """Ensure virtual environment is ready."""
        try:
            # Check if cleanup is needed (both Docker and non-Docker)
            if self._needs_cleanup():
                logger.info("Performing periodic package cleanup...")
                await self._cleanup_old_packages()
            
            # In Docker, we use system Python directly (no venv needed)
            if self.is_docker:
                logger.info("Docker environment detected - using system Python, skipping venv checks")
                return True
            
            # Non-Docker: full validation
            if not self.venv_dir.exists() or not self.python_path.exists():
                logger.info("Creating virtual environment...")
                await self._recreate_venv()
                return True
            try:
                process = await asyncio.create_subprocess_exec(
                    str(self.python_path), "-c", "import sys; print('OK')",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5)
                if process.returncode != 0 or b'OK' not in stdout:
                    logger.info("Venv corrupted, recreating...")
                    await self._recreate_venv()
                    return True
            except Exception:
                await self._recreate_venv()
                return True
            return True
        except Exception as e:
            logger.error(f"Error ensuring venv ready: {e}")
            # In Docker, continue even if there's an error
            is_docker = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
            if is_docker:
                logger.warning("Docker environment detected - continuing despite venv check error")
                return True
            return False
    
    async def _recreate_venv(self):
        """Recreate virtual environment."""
        try:
            # In Docker, we don't use venv at all - skip entirely
            if self.is_docker:
                logger.info("Docker environment detected - skipping venv recreation, using system Python")
                # Initialize cache for package tracking
                if not self.cache_file.exists():
                    cache_data = {
                        "packages": {},
                        "last_cleanup": datetime.now().isoformat()
                    }
                    self._save_cache(cache_data)
                return
            
            # Non-Docker: safe to recreate venv
            if self.venv_dir.exists():
                shutil.rmtree(self.venv_dir)
            self.venv_dir.mkdir(parents=True, exist_ok=True)
            venv.create(str(self.venv_dir), with_pip=True, clear=True)
            process = await asyncio.create_subprocess_exec(
                str(self.pip_path), "install", "--upgrade", "pip",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            cache_data = {
                "packages": {},
                "last_cleanup": datetime.now().isoformat()
            }
            self._save_cache(cache_data)
            logger.info(f"Created fresh venv at {self.venv_dir}")
        except Exception as e:
            logger.error(f"Failed to recreate venv: {e}")
            # Don't raise in Docker - continue with system Python
            if not self.is_docker:
                raise
    
    def is_package_installed(self, package: str) -> bool:
        """Check if package is installed."""
        cache = self._load_cache()
        return package.lower() in cache.get("packages", {})
    
    def mark_package_installed(self, package: str):
        """Mark package as installed."""
        cache = self._load_cache()
        now = datetime.now().isoformat()
        cache["packages"][package.lower()] = {
            "installed_at": now,
            "last_used": now,
            "name": package
        }
        self._save_cache(cache)
    
    def update_package_usage(self, package: str):
        """Update last used timestamp for a package."""
        cache = self._load_cache()
        package_lower = package.lower()
        if package_lower in cache.get("packages", {}):
            cache["packages"][package_lower]["last_used"] = datetime.now().isoformat()
            self._save_cache(cache)
    
    def is_package_approved(self, package: str) -> Tuple[bool, str]:
        """Check if package is approved for installation."""
        package_lower = package.lower().strip()
        if package_lower in APPROVED_PACKAGES:
            return True, "Package is approved"
        suspicious = ['exec', 'eval', 'compile', 'system', 'shell', 'cmd', 'hack', 'exploit']
        for pattern in suspicious:
            if pattern in package_lower:
                return False, f"Package name contains suspicious keyword: {pattern}"
        return False, f"Package '{package}' is not in the approved list"
    
    async def install_package(self, package: str) -> Tuple[bool, str]:
        """Install a package."""
        is_approved, reason = self.is_package_approved(package)
        if not is_approved:
            return False, reason
        if self.is_package_installed(package):
            return True, f"Package '{package}' already installed"
        try:
            process = await asyncio.create_subprocess_exec(
                str(self.pip_path), "install", "--no-cache-dir", package,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
            if process.returncode == 0:
                self.mark_package_installed(package)
                logger.info(f"Successfully installed: {package}")
                return True, f"Package '{package}' installed successfully"
            else:
                error_msg = stderr.decode()
                logger.warning(f"Failed to install {package}: {error_msg}")
                return False, f"Installation failed: {error_msg[:200]}"
        except asyncio.TimeoutError:
            return False, f"Installation timeout for '{package}'"
        except Exception as e:
            return False, f"Installation error: {str(e)}"


class CodeExecutor:
    """Secure code execution with file access."""
    
    def __init__(self, package_manager: PackageManager, file_manager: FileManager):
        self.package_manager = package_manager
        self.file_manager = file_manager
        self.temp_dir = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary execution directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
    
    def validate_code_security(self, code: str) -> Tuple[bool, str]:
        """
        Validate code for security threats.
        
        Performs comprehensive security checks including:
        - Blocked patterns (dangerous imports, code execution, file ops)
        - Warning patterns (potential issues that are logged)
        - Code structure validation
        
        Args:
            code: The Python code to validate
            
        Returns:
            Tuple of (is_safe, message)
        """
        # Check for blocked patterns
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                logger.warning(f"Blocked code pattern detected: {pattern[:50]}...")
                return False, f"Security violation: Unsafe operation detected"
        
        # Check for warning patterns (log but don't block)
        for pattern, warning_msg in WARNING_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                logger.warning(f"Code warning: {warning_msg}")
        
        # Additional structural checks
        try:
            # Parse the AST to check for suspicious constructs
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Check for suspicious attribute access
                if isinstance(node, ast.Attribute):
                    if node.attr.startswith('_') and node.attr.startswith('__'):
                        logger.warning(f"Dunder attribute access detected: {node.attr}")
                        return False, "Security violation: Private attribute access not allowed"
                
                # Check for suspicious function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                            return False, f"Security violation: {node.func.id}() is not allowed"
                    
        except SyntaxError:
            # Syntax errors will be caught during execution
            pass
        except Exception as e:
            logger.warning(f"Error during AST validation: {e}")
        
        return True, "Code passed security validation"
    
    def _extract_imports_from_code(self, code: str) -> List[str]:
        """
        Extract imported module names from code using AST parsing.
        
        Args:
            code: Python code to analyze
        
        Returns:
            List of top-level module names (e.g., ['numpy', 'pandas', 'matplotlib'])
        """
        modules = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get top-level module (e.g., 'numpy' from 'numpy.random')
                        module_name = alias.name.split('.')[0]
                        modules.add(module_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Get top-level module (e.g., 'sklearn' from 'sklearn.metrics')
                        module_name = node.module.split('.')[0]
                        modules.add(module_name)
        except SyntaxError:
            # If code has syntax errors, we'll catch them during execution
            logger.debug("Syntax error while parsing imports, will be caught during execution")
        except Exception as e:
            logger.warning(f"Error extracting imports: {e}")
        
        return list(modules)
    
    def _extract_missing_modules(self, error_output: str) -> List[str]:
        """
        Extract missing module names from error output.
        
        Args:
            error_output: Stderr output from code execution
        
        Returns:
            List of missing module names
        """
        missing_modules = []
        
        # Pattern 1: ModuleNotFoundError: No module named 'xxx'
        pattern1 = r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]"
        matches1 = re.findall(pattern1, error_output)
        missing_modules.extend(matches1)
        
        # Pattern 2: ImportError: No module named xxx
        pattern2 = r"ImportError: No module named ['\"]?([^'\"\\s]+)['\"]?"
        matches2 = re.findall(pattern2, error_output)
        missing_modules.extend(matches2)
        
        # Pattern 3: cannot import name 'xxx' from 'yyy' (might be package issue)
        # For this case, we extract the parent module
        pattern3 = r"cannot import name .+ from ['\"]([^'\"]+)['\"]"
        matches3 = re.findall(pattern3, error_output)
        missing_modules.extend(matches3)
        
        # Remove duplicates and return
        return list(set(missing_modules))
    
    async def execute_code(
        self,
        code: str,
        user_id: int,
        timeout: int = EXECUTION_TIMEOUT,
        user_files: List[str] = None,
        install_packages: List[str] = None
    ) -> Dict[str, Any]:
        """Execute Python code with file access."""
        start_time = time.time()
        
        is_safe, safety_msg = self.validate_code_security(code)
        if not is_safe:
            return {
                "success": False,
                "output": "",
                "error": f"Security validation failed: {safety_msg}",
                "execution_time": 0
            }
        
        if not await self.package_manager.ensure_venv_ready():
            return {
                "success": False,
                "output": "",
                "error": "Failed to initialize execution environment",
                "execution_time": 0
            }
        
        installed_packages = []
        failed_packages = []
        if install_packages:
            for package in install_packages:
                success, msg = await self.package_manager.install_package(package)
                if success:
                    installed_packages.append(package)
                else:
                    failed_packages.append((package, msg))
        
        self.temp_dir = tempfile.mkdtemp(prefix="code_exec_")
        
        try:
            file_paths_map = {}
            if user_files:
                logger.info(f"Processing {len(user_files)} file(s) for code execution")
                for file_id in user_files:
                    file_meta = await self.file_manager.get_file(file_id, user_id)
                    if file_meta:
                        file_paths_map[file_id] = file_meta['file_path']
                        logger.info(f"Added file to execution context: {file_id} -> {file_meta['file_path']}")
                    else:
                        logger.warning(f"File {file_id} not found or expired for user {user_id}")
                
                if file_paths_map:
                    logger.info(f"Total files accessible in execution: {len(file_paths_map)}")
                else:
                    logger.warning(f"No files found for user {user_id} despite {len(user_files)} file_ids provided")
            else:
                logger.debug("No user files provided for code execution")
            
            env_setup = f"""
import sys
import os

FILES = {json.dumps(file_paths_map)}

def load_file(file_id):
    '''
    Load a file automatically based on its extension.
    Supports 200+ file types with smart auto-detection.
    
    Args:
        file_id: The file ID provided when the file was uploaded
    
    Returns:
        Loaded file data (varies by file type):
        - CSV/TSV: pandas DataFrame
        - Excel (.xlsx, .xls): pandas ExcelFile object
        - JSON: pandas DataFrame or dict
        - Parquet/Feather: pandas DataFrame
        - Text files: string content
        - Images: PIL Image object
        - And 200+ more formats...
    
    Excel file usage examples:
        excel_file = load_file('file_id')
        sheet_names = excel_file.sheet_names
        df = excel_file.parse('Sheet1')
        df2 = pd.read_excel(excel_file, sheet_name='Sheet1')
    
    Available files: {{', '.join(FILES.keys()) if FILES else 'None'}}
    '''
    if file_id not in FILES:
        available_files = list(FILES.keys())
        error_msg = f"File '{{file_id}}' not found or not accessible.\\n"
        if available_files:
            error_msg += f"Available file IDs: {{', '.join(available_files)}}"
        else:
            error_msg += "No files are currently accessible. Make sure to upload a file first."
        raise ValueError(error_msg)
    file_path = FILES[file_id]
    
    # Import common libraries (they'll auto-install if needed)
    import pandas as pd
    import json
    
    # Get file extension
    ext = file_path.lower().split('.')[-1]
    
    # Tabular data formats
    if ext == 'csv':
        return pd.read_csv(file_path)
    elif ext in ['xlsx', 'xls', 'xlsm', 'xlsb']:
        # Return ExcelFile object for multi-sheet access
        # Users can: excel_file.sheet_names, excel_file.parse('Sheet1'), or pd.read_excel(excel_file, sheet_name='Sheet1')
        return pd.ExcelFile(file_path)
    elif ext == 'ods':
        # Return ExcelFile object for ODS multi-sheet access
        return pd.ExcelFile(file_path, engine='odf')
    elif ext == 'tsv' or ext == 'tab':
        return pd.read_csv(file_path, sep='\\t')
    
    # JSON formats
    elif ext == 'json':
        try:
            return pd.read_json(file_path)
        except:
            with open(file_path, 'r') as f:
                return json.load(f)
    elif ext in ['jsonl', 'ndjson']:
        return pd.read_json(file_path, lines=True)
    elif ext == 'geojson':
        with open(file_path, 'r') as f:
            return json.load(f)
    
    # Binary data formats
    elif ext == 'parquet':
        return pd.read_parquet(file_path)
    elif ext == 'feather':
        return pd.read_feather(file_path)
    elif ext in ['hdf', 'hdf5', 'h5']:
        return pd.read_hdf(file_path)
    elif ext in ['pickle', 'pkl']:
        return pd.read_pickle(file_path)
    elif ext in ['npy', 'npz']:
        import numpy as np
        return np.load(file_path)
    
    # Database formats
    elif ext in ['db', 'sqlite', 'sqlite3']:
        import sqlite3
        return sqlite3.connect(file_path)
    elif ext == 'sql':
        with open(file_path, 'r') as f:
            return f.read()
    
    # Statistical software formats
    elif ext == 'dta':
        return pd.read_stata(file_path)
    elif ext == 'sas7bdat':
        return pd.read_sas(file_path)
    elif ext == 'sav':
        return pd.read_spss(file_path)
    
    # Markup formats
    elif ext in ['yaml', 'yml']:
        import yaml
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    elif ext == 'toml':
        import toml
        with open(file_path, 'r') as f:
            return toml.load(f)
    elif ext == 'xml':
        import xml.etree.ElementTree as ET
        return ET.parse(file_path)
    
    # Text formats
    elif ext in ['txt', 'text', 'log', 'md', 'markdown', 'rst', 'tex']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Code files (return as text)
    elif ext in ['py', 'pyw', 'r', 'js', 'ts', 'java', 'c', 'cpp', 'h', 'hpp', 
                 'cs', 'go', 'rs', 'rb', 'php', 'swift', 'kt', 'scala', 'sh', 
                 'bash', 'ps1', 'lua', 'html', 'htm', 'css', 'scss', 'jsx', 'tsx']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Binary formats (return as bytes)
    elif ext in ['bin', 'dat']:
        with open(file_path, 'rb') as f:
            return f.read()
    
    # Images (return file path - use PIL/OpenCV to load)
    elif ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'webp', 'svg']:
        # Return path - user can use PIL.Image.open(path) or cv2.imread(path)
        return file_path
    
    # Audio/Video (return file path - use librosa/moviepy to process)
    elif ext in ['mp3', 'wav', 'flac', 'ogg', 'aac', 'm4a', 'mp4', 'avi', 'mkv', 'mov']:
        return file_path
    
    # Archives (return file path - use zipfile/tarfile to extract)
    elif ext in ['zip', 'tar', 'gz', 'bz2', 'xz', '7z', 'rar', 'tgz']:
        return file_path
    
    # Default: try to open as text, fallback to binary
    else:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            with open(file_path, 'rb') as f:
                return f.read()

"""
            
            full_code = env_setup + "\n" + code
            code_file = os.path.join(self.temp_dir, "exec_code.py")
            
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(full_code)
            
            process = await asyncio.create_subprocess_exec(
                str(self.package_manager.python_path), code_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.temp_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                return_code = process.returncode
                execution_time = time.time() - start_time
                output = stdout.decode('utf-8', errors='replace') if stdout else ""
                error_output = stderr.decode('utf-8', errors='replace') if stderr else ""
                
                # Check for missing module errors and auto-install if possible
                if return_code != 0 and error_output:
                    missing_modules = self._extract_missing_modules(error_output)
                    if missing_modules:
                        logger.info(f"Detected missing modules: {missing_modules}")
                        auto_installed = []
                        
                        for module in missing_modules:
                            # Try to install the missing module
                            success, msg = await self.package_manager.install_package(module)
                            if success:
                                auto_installed.append(module)
                                logger.info(f"Auto-installed missing module: {module}")
                        
                        # If we successfully installed packages, retry execution
                        if auto_installed:
                            logger.info(f"Retrying execution after installing: {auto_installed}")
                            
                            # Retry the execution
                            process = await asyncio.create_subprocess_exec(
                                str(self.package_manager.python_path), code_file,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                                cwd=self.temp_dir
                            )
                            
                            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                            return_code = process.returncode
                            execution_time = time.time() - start_time
                            output = stdout.decode('utf-8', errors='replace') if stdout else ""
                            error_output = stderr.decode('utf-8', errors='replace') if stderr else ""
                            
                            # Add auto-installed packages to result
                            if not installed_packages:
                                installed_packages = []
                            installed_packages.extend(auto_installed)
                
                if len(output) > MAX_OUTPUT_SIZE:
                    output = output[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"
                
                # Capture ALL generated files (not just images)
                generated_files = []
                generated_file_ids = []  # Store file IDs for 48-hour persistence
                
                for file_name in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, file_name)
                    
                    # Skip the code file itself and non-files
                    if file_name == "exec_code.py" or not os.path.isfile(file_path):
                        continue
                    
                    try:
                        # Determine file type from extension
                        ext = os.path.splitext(file_name)[1].lower()
                        
                        # Categorize file types
                        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.bmp', '.webp', '.tiff']:
                            file_type = "image"
                        elif ext in ['.csv', '.xlsx', '.xls', '.tsv']:
                            file_type = "data"
                        elif ext in ['.txt', '.md', '.log', '.out', '.err']:
                            file_type = "text"
                        elif ext in ['.json', '.jsonl', '.xml', '.yaml', '.yml', '.toml']:
                            file_type = "structured"
                        elif ext in ['.html', '.htm']:
                            file_type = "html"
                        elif ext in ['.pdf']:
                            file_type = "pdf"
                        elif ext in ['.py', '.js', '.java', '.cpp', '.c', '.r', '.sql']:
                            file_type = "code"
                        elif ext in ['.zip', '.tar', '.gz', '.7z']:
                            file_type = "archive"
                        else:
                            file_type = "file"
                        
                        # Check file size (max 50MB per file)
                        file_size = os.path.getsize(file_path)
                        if file_size > MAX_FILE_SIZE:
                            logger.warning(f"Generated file {file_name} is too large ({file_size} bytes), skipping")
                            continue
                        
                        # Read file content
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                        
                        # Save to file manager with 48-hour expiration
                        save_result = await self.file_manager.save_file(
                            user_id=user_id,
                            file_data=file_data,
                            filename=file_name,
                            file_type=file_type
                        )
                        
                        if save_result['success']:
                            generated_file_ids.append(save_result['file_id'])
                            logger.info(f"Saved generated file: {file_name} ({file_type}, {file_size} bytes) -> {save_result['file_id']}")
                        
                        # Also add to immediate return (for Discord upload)
                        generated_files.append({
                            "filename": file_name,
                            "data": file_data,
                            "type": file_type,
                            "size": file_size,
                            "file_id": save_result.get('file_id')  # Include file_id for later access
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to process generated file {file_name}: {e}")
                
                result = {
                    "success": return_code == 0,
                    "output": output,
                    "error": error_output if error_output else "",
                    "execution_time": execution_time,
                    "return_code": return_code,
                    "generated_files": generated_files,
                    "generated_file_ids": generated_file_ids  # File IDs for 48-hour access
                }
                
                if installed_packages:
                    result["installed_packages"] = installed_packages
                if failed_packages:
                    result["failed_packages"] = failed_packages
                
                # Track package usage if execution was successful
                if return_code == 0:
                    try:
                        imported_modules = self._extract_imports_from_code(code)
                        for module in imported_modules:
                            # Update usage timestamp for installed packages
                            if self.package_manager.is_package_installed(module):
                                self.package_manager.update_package_usage(module)
                                logger.debug(f"Updated usage timestamp for package: {module}")
                    except Exception as e:
                        logger.warning(f"Failed to track package usage: {e}")
                
                return result
                
            except asyncio.TimeoutError:
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass
                return {
                    "success": False,
                    "output": "",
                    "error": f"Execution timeout after {timeout} seconds",
                    "execution_time": timeout,
                    "return_code": -1
                }
        
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "output": "",
                "error": f"Execution error: {str(e)}",
                "execution_time": execution_time,
                "traceback": traceback.format_exc()
            }


_package_manager = None
_file_manager = None


def get_package_manager() -> PackageManager:
    """Get or create global package manager."""
    global _package_manager
    if _package_manager is None:
        _package_manager = PackageManager()
    return _package_manager


def get_file_manager(db_handler=None) -> FileManager:
    """Get or create global file manager."""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager(db_handler)
    elif db_handler and not _file_manager.db:
        _file_manager.db = db_handler
    return _file_manager


async def execute_code(
    code: str,
    user_id: int,
    user_files: List[str] = None,
    install_packages: List[str] = None,
    timeout: int = EXECUTION_TIMEOUT,
    db_handler=None
) -> Dict[str, Any]:
    """Execute Python code with optional file access and package installation."""
    try:
        package_mgr = get_package_manager()
        file_mgr = get_file_manager(db_handler)
        with CodeExecutor(package_mgr, file_mgr) as executor:
            result = await executor.execute_code(
                code=code,
                user_id=user_id,
                timeout=timeout,
                user_files=user_files,
                install_packages=install_packages
            )
            return result
    except Exception as e:
        logger.error(f"Error in execute_code: {e}\n{traceback.format_exc()}")
        return {
            "success": False,
            "output": "",
            "error": f"Code interpreter error: {str(e)}",
            "execution_time": 0
        }


async def upload_file(
    user_id: int,
    file_data: bytes,
    filename: str,
    file_type: str = None,
    db_handler=None
) -> Dict[str, Any]:
    """Upload a file for use in code execution."""
    try:
        file_mgr = get_file_manager(db_handler)
        result = await file_mgr.save_file(user_id, file_data, filename, file_type)
        return result
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return {"success": False, "error": str(e)}


async def upload_discord_attachment(
    attachment,
    user_id: int,
    db_handler=None
) -> Dict[str, Any]:
    """
    Upload a Discord attachment to the code interpreter.
    
    Args:
        attachment: Discord attachment object
        user_id: Discord user ID
        db_handler: Database handler
    
    Returns:
        Dict with success status, file_id, and metadata
    """
    try:
        # Read attachment data
        file_data = await attachment.read()
        filename = attachment.filename
        
        # Detect file type from extension (will use comprehensive type map)
        file_mgr = get_file_manager(db_handler)
        file_type = file_mgr._detect_file_type(filename)
        
        # Upload the file
        result = await upload_file(
            user_id=user_id,
            file_data=file_data,
            filename=filename,
            file_type=file_type,
            db_handler=db_handler
        )
        
        if result['success']:
            logger.info(f"Uploaded Discord attachment {filename} for user {user_id}: {result['file_id']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error uploading Discord attachment: {e}")
        return {"success": False, "error": str(e)}


async def load_file(file_id: str, user_id: int, db_handler=None) -> Dict[str, Any]:
    """
    Load a file by its ID (uploaded or generated).
    Returns file metadata and content if file exists and hasn't expired.
    
    Args:
        file_id: File ID to load
        user_id: User ID (for ownership verification)
        db_handler: Database handler
    
    Returns:
        Dict with success status, file metadata, and file data
    """
    try:
        file_mgr = get_file_manager(db_handler)
        file_meta = await file_mgr.get_file(file_id, user_id)
        
        if not file_meta:
            return {"success": False, "error": "File not found or expired"}
        
        # Read file content
        file_path = Path(file_meta['file_path'])
        if not file_path.exists():
            return {"success": False, "error": "File not found on disk"}
        
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file_meta['filename'],
            "file_type": file_meta['file_type'],
            "file_size": file_meta['file_size'],
            "data": file_data,
            "uploaded_at": file_meta['uploaded_at'],
            "expires_at": file_meta['expires_at']
        }
        
    except Exception as e:
        logger.error(f"Error loading file {file_id}: {e}")
        return {"success": False, "error": str(e)}


async def list_user_files(user_id: int, db_handler=None) -> List[Dict[str, Any]]:
    """List all files for a user (non-expired)."""
    try:
        file_mgr = get_file_manager(db_handler)
        files = await file_mgr.get_user_files(user_id, include_expired=False)
        return files
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return []


async def delete_user_file(file_id: str, user_id: int, db_handler=None) -> bool:
    """Delete a user file."""
    try:
        file_mgr = get_file_manager(db_handler)
        return await file_mgr.delete_file(file_id, user_id)
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return False


async def cleanup_expired_files(db_handler=None) -> int:
    """Clean up expired files (>48 hours old)."""
    try:
        file_mgr = get_file_manager(db_handler)
        return await file_mgr.cleanup_expired_files()
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")
        return 0


async def get_interpreter_status(db_handler=None) -> Dict[str, Any]:
    """Get status of the code interpreter system."""
    try:
        package_mgr = get_package_manager()
        file_mgr = get_file_manager(db_handler)
        cache = package_mgr._load_cache()
        total_files = 0
        total_size = 0
        if db_handler:
            all_files = await db_handler.db.user_files.find({}).to_list(length=10000)
            total_files = len(all_files)
            total_size = sum(f.get('file_size', 0) for f in all_files)
        status = {
            "venv_exists": package_mgr.venv_dir.exists(),
            "python_path": str(package_mgr.python_path),
            "installed_packages": list(cache.get("packages", {}).keys()),
            "package_count": len(cache.get("packages", {})),
            "last_cleanup": cache.get("last_cleanup"),
            "total_user_files": total_files,
            "total_file_size_mb": round(total_size / (1024 * 1024), 2),
            "file_expiration_hours": FILE_EXPIRATION_HOURS,
            "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024)
        }
        return status
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {"error": str(e)}


# Cleanup scheduler
class CleanupScheduler:
    """
    Scheduler for automatic cleanup of expired files and venv maintenance.
    Can be used with discord.ext.tasks or asyncio scheduling.
    """
    
    def __init__(self, db_handler=None):
        self.db = db_handler
        self.is_running = False
    
    async def run_cleanup(self) -> Dict[str, Any]:
        """
        Run a full cleanup cycle.
        
        Returns:
            Dict with cleanup statistics
        """
        try:
            logger.info("Starting scheduled cleanup...")
            
            # Clean up expired files
            deleted_files = await cleanup_expired_files(db_handler=self.db)
            
            # Check venv status
            package_mgr = get_package_manager()
            venv_recreated = False
            
            if package_mgr._needs_cleanup():
                logger.info("Venv cleanup needed, recreating...")
                await package_mgr._recreate_venv()
                venv_recreated = True
            
            result = {
                "success": True,
                "deleted_files": deleted_files,
                "venv_recreated": venv_recreated,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Cleanup completed: {deleted_files} files deleted, venv_recreated={venv_recreated}")
            return result
            
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def start_periodic_cleanup(self, interval_hours: int = 1):
        """
        Start periodic cleanup (for use without discord.ext.tasks).
        
        Args:
            interval_hours: Hours between cleanup runs
        """
        self.is_running = True
        logger.info(f"Starting periodic cleanup every {interval_hours} hour(s)")
        
        while self.is_running:
            try:
                await self.run_cleanup()
                await asyncio.sleep(interval_hours * 3600)
            except asyncio.CancelledError:
                logger.info("Cleanup scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    def stop(self):
        """Stop the periodic cleanup."""
        self.is_running = False
        logger.info("Cleanup scheduler stopped")


def create_discord_cleanup_task(bot, db_handler):
    """
    Create a discord.ext.tasks cleanup task for the bot.
    
    Usage in bot.py:
        from src.utils.code_interpreter import create_discord_cleanup_task
        cleanup_task = create_discord_cleanup_task(bot, db_handler)
        cleanup_task.start()
    
    Args:
        bot: Discord bot instance
        db_handler: Database handler
    
    Returns:
        discord.ext.tasks.Loop instance
    """
    try:
        from discord.ext import tasks
        
        @tasks.loop(hours=1)
        async def cleanup_task():
            """Run cleanup every hour."""
            try:
                scheduler = CleanupScheduler(db_handler)
                result = await scheduler.run_cleanup()
                
                if result.get('deleted_files', 0) > 0:
                    logger.info(f"[Cleanup] Removed {result['deleted_files']} expired files")
                
                if result.get('venv_recreated'):
                    logger.info("[Cleanup] Recreated virtual environment")
                    
            except Exception as e:
                logger.error(f"[Cleanup] Error: {e}")
        
        return cleanup_task
        
    except ImportError:
        logger.warning("discord.ext.tasks not available, use CleanupScheduler directly")
        return None


async def list_user_files(user_id: int, db_handler=None) -> list:
    """
    List all files uploaded by a user.
    
    Args:
        user_id: User ID
        db_handler: Database handler
    
    Returns:
        List of file metadata dictionaries
    """
    if not db_handler:
        return []
    
    try:
        db = db_handler.db if hasattr(db_handler, 'db') else db_handler.get_database()
        
        # Query all files for this user
        cursor = db.user_files.find({"user_id": user_id})
        files = await cursor.to_list(length=None)
        
        # Remove MongoDB _id for cleaner output
        for file in files:
            file.pop('_id', None)
        
        logger.info(f"Listed {len(files)} files for user {user_id}")
        return files
        
    except Exception as e:
        logger.error(f"Error listing user files: {e}")
        return []


async def get_file_metadata(file_id: str, user_id: int, db_handler=None):
    """
    Get metadata for a specific file.
    
    Args:
        file_id: The file ID
        user_id: User ID (for permission check)
        db_handler: Database handler
    
    Returns:
        Dict with file metadata or None if not found
    """
    if not db_handler:
        return None
    
    try:
        db = db_handler.db if hasattr(db_handler, 'db') else db_handler.get_database()
        file_meta = await db.user_files.find_one({
            "file_id": file_id,
            "user_id": user_id
        })
        
        if file_meta:
            # Remove MongoDB _id for cleaner output
            file_meta.pop('_id', None)
            return file_meta
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting file metadata: {e}")
        return None


async def delete_file(file_id: str, user_id: int, db_handler=None) -> dict:
    """
    Delete a specific user file.
    
    Args:
        file_id: The file ID to delete
        user_id: User ID (for permission check)
        db_handler: Database handler
    
    Returns:
        Dict with success status
    """
    try:
        # Get file metadata first
        file_meta = await get_file_metadata(file_id, user_id, db_handler)
        
        if not file_meta:
            return {
                "success": False,
                "error": "File not found or you don't have permission to delete it"
            }
        
        # Delete physical file
        file_path = Path(file_meta['file_path'])
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted physical file: {file_path}")
        else:
            logger.warning(f"Physical file not found: {file_path}")
        
        # Delete from database
        if db_handler:
            db = db_handler.db if hasattr(db_handler, 'db') else db_handler.get_database()
            result = await db.user_files.delete_one({
                "file_id": file_id,
                "user_id": user_id
            })
            
            if result.deleted_count > 0:
                logger.info(f"Deleted file from database: {file_id}")
                return {"success": True, "message": f"Deleted file: {file_meta.get('filename', file_id)}"}
            else:
                return {"success": False, "error": "File not found in database"}
        
        return {"success": True, "message": "File deleted from disk"}
        
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def delete_all_user_files(user_id: int, db_handler=None) -> dict:
    """
    Delete all files for a specific user.
    Used when resetting user data or cleaning up.
    
    Args:
        user_id: User ID whose files should be deleted
        db_handler: Database handler
    
    Returns:
        Dict with success status and count of deleted files
    """
    try:
        # Get all user files first
        files = await list_user_files(user_id, db_handler)
        
        if not files:
            return {
                "success": True,
                "deleted_count": 0,
                "message": "No files to delete"
            }
        
        deleted_count = 0
        failed_count = 0
        
        # Delete each file
        for file_meta in files:
            # Delete physical file
            file_path = Path(file_meta['file_path'])
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted physical file: {file_path}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting physical file {file_path}: {e}")
                failed_count += 1
        
        # Delete all from database in one operation
        if db_handler:
            db = db_handler.db if hasattr(db_handler, 'db') else db_handler.get_database()
            result = await db.user_files.delete_many({"user_id": user_id})
            logger.info(f"Deleted {result.deleted_count} file records from database for user {user_id}")
        
        # Try to remove user directory if empty
        try:
            user_dir = Path(USER_FILES_DIR) / str(user_id)
            if user_dir.exists() and not any(user_dir.iterdir()):
                user_dir.rmdir()
                logger.info(f"Removed empty user directory: {user_dir}")
        except Exception as e:
            logger.warning(f"Could not remove user directory: {e}")
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "failed_count": failed_count,
            "message": f"Deleted {deleted_count} file(s)" + (f" ({failed_count} failed)" if failed_count > 0 else "")
        }
        
    except Exception as e:
        logger.error(f"Error deleting all user files: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "deleted_count": 0}
