# Docker Virtual Environment Fix

## Problem
When deploying the bot in a Docker container, the system would attempt to create and manage a virtual environment inside the container, resulting in "Resource busy" errors:
```
Failed to recreate venv: [Errno 16] Resource busy: PosixPath('/tmp/bot_code_interpreter/venv')
```

## Root Cause
- Docker containers already provide complete isolation (similar to virtual environments)
- Creating a venv inside Docker is redundant and causes file locking issues
- The Dockerfile installs all required packages from `requirements.txt` into the system Python during the build phase
- Attempting to create/manage a venv while the container is running conflicts with the running Python process

## Solution
Modified the `PackageManager` class in `src/utils/code_interpreter.py` to detect Docker environments and use system Python directly instead of creating a virtual environment.

### Changes Made

#### 1. Updated `__init__` method (Lines ~485-495)
- Added `self.is_docker` attribute to detect Docker environment once during initialization
- Detection checks for `/.dockerenv` or `/run/.containerenv` files

```python
def __init__(self):
    self.venv_dir = PERSISTENT_VENV_DIR
    self.cache_file = PACKAGE_CACHE_FILE
    self.python_path = None
    self.pip_path = None
    self.is_docker = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
    self._setup_paths()
```

#### 2. Updated `_setup_paths` method (Lines ~497-507)
- In Docker: Uses system Python executable (`sys.executable`)
- In non-Docker: Uses venv paths as before

```python
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
```

#### 3. Updated `ensure_venv_ready` method (Lines ~541-580)
- In Docker: Returns immediately without any venv checks
- In non-Docker: Performs full venv validation and creation as before

```python
async def ensure_venv_ready(self) -> bool:
    """Ensure virtual environment is ready."""
    try:
        # In Docker, we use system Python directly (no venv needed)
        if self.is_docker:
            logger.info("Docker environment detected - using system Python, skipping venv checks")
            return True
        
        # Non-Docker: full validation
        # ... existing venv checks ...
```

#### 4. Updated `_recreate_venv` method (Lines ~583-616)
- In Docker: Skips all venv creation, only initializes package cache
- In non-Docker: Recreates venv normally

```python
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
        # ... existing venv creation logic ...
```

## How It Works

### Docker Environment
1. **Detection**: On initialization, checks for Docker indicator files
2. **Path Setup**: Uses system Python at `/usr/local/bin/python3` (or wherever `sys.executable` points)
3. **Package Management**: 
   - System packages are pre-installed during Docker build
   - `pip install` commands use system pip to install to system site-packages
   - No venv directory is created or managed
4. **Isolation**: Docker container itself provides process and filesystem isolation

### Non-Docker Environment (Local Development)
1. **Venv Creation**: Creates persistent venv at `/tmp/bot_code_interpreter/venv`
2. **Package Management**: Installs packages in isolated venv
3. **Cleanup**: Periodic cleanup to prevent corruption
4. **Validation**: Checks venv health on every code execution

## Benefits

✅ **Eliminates "Resource busy" errors** in Docker deployments
✅ **Faster startup** in Docker (no venv creation overhead)
✅ **Simpler architecture** - leverages Docker's built-in isolation
✅ **Still supports venv** for local development outside Docker
✅ **Consistent behavior** - all packages available from Docker build

## Testing

### To test in Docker:
```bash
docker-compose up --build
# Bot should start without any venv-related errors
# Code execution should work normally
```

### To test locally (non-Docker):
```bash
python bot.py
# Should create venv in /tmp/bot_code_interpreter/venv
# Should work as before
```

## Related Files
- `src/utils/code_interpreter.py` - Main changes
- `Dockerfile` - Installs system packages
- `requirements.txt` - Packages installed in Docker

## Future Considerations
- Package installation in Docker adds to system site-packages permanently
- Consider adding package cleanup mechanism for Docker if needed
- Could add volume mount for persistent package storage in Docker if desired
