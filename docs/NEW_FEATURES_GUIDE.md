# Code Interpreter - New Features Guide

## 🎯 Three Major Improvements

### 1. ✅ Discord File Upload Support

Automatically handles Discord file attachments.

**Function:**
```python
from src.utils.code_interpreter import upload_discord_attachment

result = await upload_discord_attachment(
    attachment=discord_attachment,
    user_id=user_id,
    db_handler=db
)
# Returns: {"success": True, "file_id": "...", "metadata": {...}}
```

**Supported file types:**
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- JSON (`.json`)
- Text (`.txt`)
- Python (`.py`)

### 2. ✅ Auto-Install Missing Packages

Automatically detects and installs missing packages during execution.

**How it works:**
1. Code fails with `ModuleNotFoundError`
2. System extracts module name from error
3. Checks if approved (62 data science packages)
4. Auto-installs and retries execution

**Example:**
```python
# User code:
import seaborn as sns  # Not installed yet
sns.load_dataset('tips')

# System automatically:
# 1. Detects seaborn is missing
# 2. Installs it
# 3. Retries execution
# 4. Returns success with installed_packages=['seaborn']
```

**Detected error patterns:**
- `ModuleNotFoundError: No module named 'xxx'`
- `ImportError: No module named xxx`
- `cannot import name 'yyy' from 'xxx'`

### 3. ✅ Automatic Cleanup Task

Built-in scheduler for maintenance.

**Quick Setup:**
```python
# In bot.py
from src.utils.code_interpreter import create_discord_cleanup_task

cleanup_task = create_discord_cleanup_task(bot, db_handler)

@bot.event
async def on_ready():
    cleanup_task.start()  # Runs every hour
    print("Cleanup task started!")
```

**What it cleans:**
- Files older than 48 hours
- Empty user directories  
- Recreates venv every 7 days

## 📦 Integration Example

### Complete bot.py Setup

```python
import discord
from discord.ext import commands
from src.database.db_handler import DatabaseHandler
from src.utils.code_interpreter import (
    create_discord_cleanup_task,
    upload_discord_attachment,
    execute_code
)

bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
db = DatabaseHandler(MONGODB_URI)

# Setup cleanup
cleanup_task = create_discord_cleanup_task(bot, db)

@bot.event
async def on_ready():
    print(f'Bot ready: {bot.user}')
    cleanup_task.start()
    print("✅ Cleanup running (every hour)")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    # Handle file uploads
    if message.attachments:
        for att in message.attachments:
            if att.filename.endswith(('.csv', '.xlsx', '.json')):
                result = await upload_discord_attachment(
                    attachment=att,
                    user_id=message.author.id,
                    db_handler=db
                )
                
                if result['success']:
                    await message.channel.send(
                        f"✅ Uploaded: `{att.filename}`\n"
                        f"📁 ID: `{result['file_id']}`\n"
                        f"⏰ Expires in 48h"
                    )
    
    await bot.process_commands(message)

bot.run(TOKEN)
```

## 🔍 Usage Examples

### Example 1: User Uploads CSV

```
User: *uploads sales.csv*
Bot:  ✅ Uploaded: sales.csv
      📁 ID: user_123_1234567890_abc123
      ⏰ Expires in 48h

User: Analyze this sales data
AI:   *calls execute_code with:*
      - code: "df = load_file('user_123_1234567890_abc123')"
      - user_files: ['user_123_1234567890_abc123']

Bot:  📊 Analysis Results:
      Shape: (1000, 5)
      Total Sales: $125,432.50
      *chart.png*
```

### Example 2: Missing Package Auto-Install

```
User: Create a correlation heatmap
AI:   *calls execute_code with:*
      code: "import seaborn as sns..."

System: ❌ ModuleNotFoundError: No module named 'seaborn'
        ℹ️  Detected missing: seaborn
        📦 Installing seaborn...
        ✅ Installed successfully
        🔄 Retrying execution...
        ✅ Success!

Bot:  📊 Here's your heatmap
      *heatmap.png*
      
      📦 Auto-installed: seaborn, matplotlib
```

### Example 3: Cleanup in Action

```
[Every hour automatically]

System: [Cleanup] Starting...
        [Cleanup] Found 3 expired files
        [Cleanup] Deleted: sales.csv (expired 2h ago)
        [Cleanup] Deleted: data.xlsx (expired 5h ago)  
        [Cleanup] Deleted: test.json (expired 1h ago)
        [Cleanup] Removed 3 files
        [Cleanup] Cleaned 2 empty directories
        [Cleanup] Completed in 0.5s
```

## ⚙️ Configuration Options

### Customize Cleanup Interval

```python
# Default: 1 hour
cleanup_task = create_discord_cleanup_task(bot, db)

# Or use manual interval:
from src.utils.code_interpreter import CleanupScheduler

scheduler = CleanupScheduler(db)
await scheduler.start_periodic_cleanup(interval_hours=2)  # Every 2 hours
```

### Check Status

```python
from src.utils.code_interpreter import get_interpreter_status

status = await get_interpreter_status(db_handler=db)

print(f"Venv ready: {status['venv_exists']}")
print(f"Packages: {status['package_count']}")
print(f"User files: {status['total_user_files']}")
print(f"Total size: {status['total_file_size_mb']} MB")
```

### Manual Cleanup

```python
from src.utils.code_interpreter import cleanup_expired_files

# Run anytime
deleted = await cleanup_expired_files(db_handler=db)
print(f"Cleaned {deleted} files")
```

## 🛡️ Security Features

All features maintain security:

✅ **File Upload**: Max 50MB, 48h expiration  
✅ **Packages**: Only 62 approved packages  
✅ **Cleanup**: Automatic, no manual intervention needed  
✅ **Execution**: Sandboxed, blocked operations enforced

## 📊 Benefits

| Feature | Before | After |
|---------|--------|-------|
| File Upload | Manual file management | Auto Discord integration |
| Missing Packages | Manual install commands | Auto-detect and install |
| Cleanup | Manual scripts | Automatic every hour |
| User Experience | Complex setup | Seamless, automatic |

## 🚀 Next Steps

1. **Add cleanup task** to `bot.py` (see example above)
2. **Test file upload** - upload a CSV in Discord
3. **Test auto-install** - use seaborn without installing
4. **Monitor logs** - watch cleanup run every hour

## 📝 Summary

✅ **Discord file uploads** - Automatic, seamless integration  
✅ **Missing packages** - Auto-detect and install on-the-fly  
✅ **Cleanup task** - Runs hourly, maintains system health  

**All features are production-ready and tested!** 🎉
