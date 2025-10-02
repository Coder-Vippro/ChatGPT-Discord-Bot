# Code Interpreter Guide

## Overview

The unified code interpreter provides ChatGPT/Claude-style code execution capabilities:

- **Secure Python execution** in isolated virtual environments
- **File management** with automatic 48-hour expiration
- **Data analysis** with pandas, numpy, matplotlib, seaborn, plotly
- **Package installation** with security validation
- **Visualization generation** with automatic image handling

## Features

### 1. Code Execution

Execute arbitrary Python code securely:

```python
from src.utils.code_interpreter import execute_code

result = await execute_code(
    code="print('Hello, world!')",
    user_id=123456789
)

# Result:
# {
#     "success": True,
#     "output": "Hello, world!\n",
#     "error": "",
#     "execution_time": 0.05,
#     "return_code": 0
# }
```

### 2. File Upload & Management

Upload files for code to access:

```python
from src.utils.code_interpreter import upload_file, list_user_files

# Upload a CSV file
with open('data.csv', 'rb') as f:
    result = await upload_file(
        user_id=123456789,
        file_data=f.read(),
        filename='data.csv',
        file_type='csv',
        db_handler=db
    )

file_id = result['file_id']

# List user's files
files = await list_user_files(user_id=123456789, db_handler=db)
```

### 3. Code with File Access

Access uploaded files in code:

```python
# Upload a CSV file first
result = await upload_file(user_id=123, file_data=csv_bytes, filename='sales.csv')
file_id = result['file_id']

# Execute code that uses the file
code = """
# load_file() is automatically available
df = load_file('""" + file_id + """')
print(df.head())
print(f"Total rows: {len(df)}")
"""

result = await execute_code(
    code=code,
    user_id=123,
    user_files=[file_id],
    db_handler=db
)
```

### 4. Package Installation

Install approved packages on-demand:

```python
result = await execute_code(
    code="""
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip')
plt.savefig('plot.png')
print('Plot saved!')
""",
    user_id=123,
    install_packages=['seaborn', 'matplotlib']
)
```

### 5. Data Analysis

Automatic data loading and analysis:

```python
# The load_file() helper automatically detects file types
code = """
# Load CSV
df = load_file('file_id_here')

# Basic analysis
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.describe())

# Correlation analysis
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.savefig('correlation.png')
"""

result = await execute_code(code=code, user_id=123, user_files=['file_id_here'])

# Visualizations are returned in result['generated_files']
for file in result.get('generated_files', []):
    print(f"Generated: {file['filename']}")
    # file['data'] contains the image bytes
```

## File Expiration

### Automatic Cleanup (48 Hours)

Files automatically expire after 48 hours:

```python
from src.utils.code_interpreter import cleanup_expired_files

# Run cleanup (should be scheduled periodically)
deleted_count = await cleanup_expired_files(db_handler=db)
print(f"Cleaned up {deleted_count} expired files")
```

### Manual File Deletion

Delete files manually:

```python
from src.utils.code_interpreter import delete_user_file

success = await delete_user_file(
    file_id='user_123_1234567890_abc123',
    user_id=123,
    db_handler=db
)
```

## Security Features

### Approved Packages

Only approved packages can be installed:

- **Data Science**: numpy, pandas, scipy, scikit-learn, statsmodels
- **Visualization**: matplotlib, seaborn, plotly, bokeh, altair
- **Image Processing**: pillow, imageio, scikit-image
- **Machine Learning**: tensorflow, keras, torch, xgboost, lightgbm
- **NLP**: nltk, spacy, gensim, wordcloud
- **Math/Science**: sympy, networkx, numba

### Blocked Operations

Code is validated against dangerous operations:

- ❌ File system writes (outside execution dir)
- ❌ Network operations (socket, requests, urllib)
- ❌ Process spawning (subprocess)
- ❌ System access (os.system, eval, exec)
- ❌ Dangerous functions (__import__, globals, locals)

### Execution Limits

- **Timeout**: 60 seconds (configurable)
- **Output Size**: 100KB max (truncated if larger)
- **File Size**: 50MB max per file

## Environment Management

### Persistent Virtual Environment

The code interpreter uses a persistent venv:

- **Location**: `/tmp/bot_code_interpreter/venv`
- **Cleanup**: Automatically recreated every 7 days
- **Packages**: Cached and reused across executions

### Status Check

Get interpreter status:

```python
from src.utils.code_interpreter import get_interpreter_status

status = await get_interpreter_status(db_handler=db)

# Returns:
# {
#     "venv_exists": True,
#     "python_path": "/tmp/bot_code_interpreter/venv/bin/python",
#     "installed_packages": ["numpy", "pandas", "matplotlib", ...],
#     "package_count": 15,
#     "last_cleanup": "2024-01-15T10:30:00",
#     "total_user_files": 42,
#     "total_file_size_mb": 125.5,
#     "file_expiration_hours": 48,
#     "max_file_size_mb": 50
# }
```

## Database Schema

### user_files Collection

```javascript
{
  "file_id": "user_123_1234567890_abc123",
  "user_id": 123456789,
  "filename": "sales_data.csv",
  "file_path": "/tmp/bot_code_interpreter/user_files/123456789/user_123_1234567890_abc123.csv",
  "file_size": 1024000,
  "file_type": "csv",
  "uploaded_at": "2024-01-15T10:30:00",
  "expires_at": "2024-01-17T10:30:00"  // 48 hours later
}
```

### Indexes

Automatically created for performance:

```python
# Compound index for user queries
await db.user_files.create_index([("user_id", 1), ("expires_at", -1)])

# Unique index for file lookups
await db.user_files.create_index("file_id", unique=True)

# Index for cleanup queries
await db.user_files.create_index("expires_at")
```

## Integration Example

Complete example integrating code interpreter:

```python
from src.utils.code_interpreter import (
    execute_code,
    upload_file,
    list_user_files,
    cleanup_expired_files
)

async def handle_user_request(user_id: int, code: str, files: list, db):
    """Handle a code execution request from a user."""
    
    # Upload any files the user provided
    uploaded_files = []
    for file_data, filename in files:
        result = await upload_file(
            user_id=user_id,
            file_data=file_data,
            filename=filename,
            db_handler=db
        )
        if result['success']:
            uploaded_files.append(result['file_id'])
    
    # Execute the code with file access
    result = await execute_code(
        code=code,
        user_id=user_id,
        user_files=uploaded_files,
        install_packages=['pandas', 'matplotlib'],
        timeout=60,
        db_handler=db
    )
    
    # Check for errors
    if not result['success']:
        return f"❌ Error: {result['error']}"
    
    # Format output
    response = f"✅ Execution completed in {result['execution_time']:.2f}s\n\n"
    
    if result['output']:
        response += f"**Output:**\n```\n{result['output']}\n```\n"
    
    # Handle generated images
    for file in result.get('generated_files', []):
        if file['type'] == 'image':
            response += f"\n📊 Generated: {file['filename']}\n"
            # file['data'] contains image bytes - save or send to Discord
    
    return response

# Periodic cleanup (run every hour)
async def scheduled_cleanup(db):
    """Clean up expired files."""
    deleted = await cleanup_expired_files(db_handler=db)
    if deleted > 0:
        logging.info(f"Cleaned up {deleted} expired files")
```

## Error Handling

### Common Errors

**Security Validation Failed**
```python
result = {
    "success": False,
    "error": "Security validation failed: Blocked unsafe operation: import\s+subprocess"
}
```

**Timeout**
```python
result = {
    "success": False,
    "error": "Execution timeout after 60 seconds",
    "execution_time": 60,
    "return_code": -1
}
```

**Package Not Approved**
```python
result = {
    "success": False,
    "error": "Package 'requests' is not in the approved list"
}
```

**File Too Large**
```python
result = {
    "success": False,
    "error": "File too large. Maximum size is 50MB"
}
```

## Best Practices

1. **Always provide db_handler** for file management
2. **Set reasonable timeouts** for long-running code
3. **Handle generated_files** in results (images, etc.)
4. **Run cleanup_expired_files()** periodically (hourly recommended)
5. **Validate user input** before passing to execute_code()
6. **Check result['success']** before using output
7. **Display execution_time** to users for transparency

## Architecture

### Components

1. **FileManager**: Handles file upload/download, expiration, cleanup
2. **PackageManager**: Manages venv, installs packages, caches installations
3. **CodeExecutor**: Executes code securely, provides file access helpers

### Execution Flow

```
User Code Request
    ↓
Security Validation (blocked patterns)
    ↓
Ensure venv Ready (create if needed)
    ↓
Install Packages (if requested)
    ↓
Create Temp Execution Dir
    ↓
Inject File Access Helpers (load_file, FILES dict)
    ↓
Execute Code (isolated subprocess)
    ↓
Collect Output + Generated Files
    ↓
Cleanup Temp Dir
    ↓
Return Results
```

## Comparison to Old System

### Old System (3 separate files)
- `code_interpreter.py` - Router/dispatcher
- `python_executor.py` - Execution logic
- `data_analyzer.py` - Data analysis templates

### New System (1 unified file)
- ✅ All functionality in `code_interpreter.py`
- ✅ 48-hour file expiration (like images)
- ✅ Persistent venv with package caching
- ✅ Better security validation
- ✅ Automatic data loading helpers
- ✅ Unified API with async/await
- ✅ MongoDB integration for file tracking
- ✅ Automatic cleanup scheduling

## Troubleshooting

### Venv Creation Fails

Check disk space and permissions:
```bash
df -h /tmp
ls -la /tmp/bot_code_interpreter
```

### Packages Won't Install

Check if package is approved:
```python
from src.utils.code_interpreter import get_package_manager

pm = get_package_manager()
is_approved, reason = pm.is_package_approved('package_name')
print(f"Approved: {is_approved}, Reason: {reason}")
```

### Files Not Found

Check expiration:
```python
from src.utils.code_interpreter import get_file_manager

fm = get_file_manager(db_handler=db)
file_meta = await fm.get_file(file_id, user_id)

if not file_meta:
    print("File expired or doesn't exist")
else:
    print(f"Expires at: {file_meta['expires_at']}")
```

### Performance Issues

Check status and cleanup:
```python
status = await get_interpreter_status(db_handler=db)
print(f"Total files: {status['total_user_files']}")
print(f"Total size: {status['total_file_size_mb']} MB")

# Force cleanup
deleted = await cleanup_expired_files(db_handler=db)
print(f"Cleaned up: {deleted} files")
```

## Migration from Old System

If migrating from the old 3-file system:

1. **Replace imports**:
   ```python
   # Old
   from src.utils.python_executor import execute_python_code
   from src.utils.data_analyzer import analyze_data_file
   
   # New
   from src.utils.code_interpreter import execute_code
   ```

2. **Update function calls**:
   ```python
   # Old
   result = await execute_python_code({
       "code": code,
       "user_id": user_id
   })
   
   # New
   result = await execute_code(
       code=code,
       user_id=user_id,
       db_handler=db
   )
   ```

3. **Handle file uploads**:
   ```python
   # New file handling
   result = await upload_file(
       user_id=user_id,
       file_data=bytes,
       filename=name,
       db_handler=db
   )
   ```

4. **Schedule cleanup**:
   ```python
   # Add to bot startup
   @tasks.loop(hours=1)
   async def cleanup_task():
       await cleanup_expired_files(db_handler=db)
   ```

## Summary

The unified code interpreter provides:

- 🔒 **Security**: Validated patterns, approved packages only
- ⏱️ **Expiration**: Automatic 48-hour file cleanup
- 📦 **Packages**: Persistent venv with caching
- 📊 **Analysis**: Built-in data loading helpers
- 🎨 **Visualizations**: Automatic image generation handling
- 🔄 **Integration**: Clean async API with MongoDB
- 📈 **Status**: Real-time monitoring and metrics

All in one file: `src/utils/code_interpreter.py`
