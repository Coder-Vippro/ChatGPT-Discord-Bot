# File Management System - Complete Guide

## 🎯 Overview

A streamlined file management system that allows users to:
- Upload files via Discord attachments
- List all uploaded files with `/files` command
- Download or delete files with 2-step confirmation
- Files accessible by ALL tools (code_interpreter, analyze_data_file, etc.)
- Configurable expiration (48h default, or permanent with `-1`)

## 📋 Features

### 1. **File Upload** (Automatic)
- Simply attach a file to your message
- Bot automatically saves and tracks it
- Get a unique `file_id` for later reference
- Files stored on disk, metadata in MongoDB

### 2. **File Listing** (`/files`)
- View all your uploaded files
- See file type, size, upload date
- Expiration countdown (or "Never" if permanent)
- Interactive dropdown to select files

### 3. **File Download**
- Select file from dropdown
- Click "⬇️ Download" button
- File sent directly to you via Discord DM
- Works for files <25MB (Discord limit)

### 4. **File Deletion** (2-Step Confirmation)
- Select file from dropdown
- Click "🗑️ Delete" button
- **First confirmation**: "⚠️ Yes, Delete"
- **Second confirmation**: "🔴 Click Again to Confirm"
- Only deleted after both confirmations

### 5. **AI Integration**
- AI can automatically access your files
- Use `load_file('file_id')` in code execution
- Files available to ALL tools:
  - `execute_python_code` ✅
  - `analyze_data_file` ✅
  - Any custom tools ✅

### 6. **Configurable Expiration**
Set in `.env` file:
```bash
# Files expire after 48 hours
FILE_EXPIRATION_HOURS=48

# Files expire after 7 days
FILE_EXPIRATION_HOURS=168

# Files NEVER expire (permanent storage)
FILE_EXPIRATION_HOURS=-1
```

## 💡 Usage Examples

### Example 1: Upload and Analyze Data

```
User: [Attaches sales_data.csv]
      "Analyze this data"

Bot: File saved! ID: 123456789_1696118400_a1b2c3d4
     [Executes analysis]
     
     📊 Analysis Results:
     - 1,250 rows
     - 8 columns
     - Date range: 2024-01-01 to 2024-09-30
     
     [Generates chart and summary]
```

### Example 2: List Files

```
User: /files

Bot: 📁 Your Files
     You have 3 file(s) uploaded.
     
     📊 sales_data.csv
     Type: csv • Size: 2.5 MB
     Uploaded: 2024-10-01 10:30 • ⏰ 36h left
     
     🖼️ chart.png
     Type: image • Size: 456 KB
     Uploaded: 2024-10-01 11:00 • ⏰ 35h left
     
     📝 report.txt
     Type: text • Size: 12 KB
     Uploaded: 2024-10-01 11:15 • ⏰ 35h left
     
     [Dropdown: Select a file...]
     
     💡 Files expire after 48h • Use the menu below to manage files
```

### Example 3: Download File

```
User: /files → [Selects sales_data.csv]

Bot: 📄 sales_data.csv
     Type: csv
     Size: 2.50 MB
     
     [⬇️ Download] [🗑️ Delete]

User: [Clicks Download]

Bot: ✅ Downloaded: sales_data.csv
     [Sends file attachment]
```

### Example 4: Delete File (2-Step)

```
User: /files → [Selects old_data.csv] → [Clicks Delete]

Bot: ⚠️ Confirm Deletion
     Are you sure you want to delete:
     old_data.csv?
     
     This action cannot be undone!
     
     [⚠️ Yes, Delete] [❌ Cancel]

User: [Clicks "Yes, Delete"]

Bot: ⚠️ Final Confirmation
     Click 'Click Again to Confirm' to permanently delete:
     old_data.csv
     
     This is your last chance to cancel!
     
     [🔴 Click Again to Confirm] [❌ Cancel]

User: [Clicks "Click Again to Confirm"]

Bot: ✅ File Deleted
     Successfully deleted: old_data.csv
```

### Example 5: Use File in Code

```
User: Create a visualization from file 123456789_1696118400_a1b2c3d4

AI: [Executes code]
    
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your file
df = load_file('123456789_1696118400_a1b2c3d4')

# Create visualization
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='date', y='sales')
plt.title('Sales Trend Over Time')
plt.savefig('sales_trend.png')

print(f"Created visualization from {len(df)} rows of data")
```

Bot: [Sends generated chart]
```

### Example 6: Permanent Storage

```bash
# In .env file
FILE_EXPIRATION_HOURS=-1
```

```
User: [Uploads important_data.csv]

Bot: File saved! ID: 123456789_1696118400_a1b2c3d4
     ♾️ This file never expires (permanent storage)

User: /files

Bot: 📁 Your Files
     You have 1 file(s) uploaded.
     
     📊 important_data.csv
     Type: csv • Size: 5.2 MB
     Uploaded: 2024-10-01 10:30 • ♾️ Never expires
     
     💡 Files are stored permanently
```

## 🗂️ File Storage Architecture

### Physical Storage
```
/tmp/bot_code_interpreter/
└── user_files/
    ├── 123456789/              # User ID
    │   ├── 123456789_1696118400_a1b2c3d4.csv
    │   ├── 123456789_1696120000_x9y8z7w6.xlsx
    │   └── 123456789_1696125000_p0q1r2s3.json
    └── 987654321/              # Another user
        └── ...
```

### MongoDB Metadata
```javascript
{
  "_id": ObjectId("..."),
  "file_id": "123456789_1696118400_a1b2c3d4",
  "user_id": 123456789,
  "filename": "sales_data.csv",
  "file_path": "/tmp/bot_code_interpreter/user_files/123456789/...",
  "file_size": 2621440,  // 2.5 MB
  "file_type": "csv",
  "uploaded_at": "2024-10-01T10:30:00",
  "expires_at": "2024-10-03T10:30:00"  // 48 hours later (or null if permanent)
}
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# File expiration time in hours
# Default: 48 (2 days)
# Set to -1 for permanent storage (never expires)
FILE_EXPIRATION_HOURS=48

# Examples:
# FILE_EXPIRATION_HOURS=24    # 1 day
# FILE_EXPIRATION_HOURS=72    # 3 days
# FILE_EXPIRATION_HOURS=168   # 1 week
# FILE_EXPIRATION_HOURS=-1    # Never expire (permanent)
```

### File Size Limits

```python
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB for upload
DISCORD_SIZE_LIMIT = 25 * 1024 * 1024  # 25 MB for download (non-nitro)
```

### Supported File Types (80+)

**Data Formats**: CSV, TSV, Excel (XLSX, XLS), JSON, JSONL, XML, YAML, TOML, INI, Parquet, Feather, Arrow, HDF5

**Images**: PNG, JPG, JPEG, GIF, BMP, TIFF, WebP, SVG, ICO

**Documents**: TXT, MD, PDF, DOC, DOCX, RTF, ODT

**Code**: PY, JS, TS, Java, C, CPP, Go, Rust, HTML, CSS, SQL

**Scientific**: MAT, NPY, NPZ, NetCDF, FITS, HDF5

**Geospatial**: GeoJSON, SHP, KML, GPX, GeoTIFF

**Archives**: ZIP, TAR, GZ, BZ2, XZ, RAR, 7Z

## 🔄 File Lifecycle

### With Expiration (FILE_EXPIRATION_HOURS = 48)

```
Day 1, 10:00 AM: User uploads file
    ↓
File saved: /tmp/.../user_files/123/file.csv
MongoDB: { expires_at: "Day 3, 10:00 AM" }
    ↓
Day 1-3: File available for use
    ↓
Day 3, 10:00 AM: File expires
    ↓
Cleanup task runs (every hour)
    ↓
File deleted from disk + MongoDB
```

### Without Expiration (FILE_EXPIRATION_HOURS = -1)

```
Day 1: User uploads file
    ↓
File saved: /tmp/.../user_files/123/file.csv
MongoDB: { expires_at: null }
    ↓
Forever: File remains available
    ↓
Only deleted when user manually deletes it
```

## 🎨 Interactive UI Elements

### File List View

```
📁 Your Files (Interactive)

┌─────────────────────────────────────┐
│ 📊 sales_data.csv                   │
│ Type: csv • Size: 2.5 MB           │
│ Uploaded: 2024-10-01 10:30 • 36h   │
├─────────────────────────────────────┤
│ 🖼️ chart.png                        │
│ Type: image • Size: 456 KB         │
│ Uploaded: 2024-10-01 11:00 • 35h   │
└─────────────────────────────────────┘

[▼ Select a file to manage...]
```

### File Actions

```
📄 sales_data.csv
Type: csv
Size: 2.50 MB

[⬇️ Download]  [🗑️ Delete]
```

### Delete Confirmation (2 Steps)

```
Step 1:
⚠️ Confirm Deletion
Are you sure you want to delete:
sales_data.csv?

[⚠️ Yes, Delete]  [❌ Cancel]

↓ (User clicks Yes)

Step 2:
⚠️ Final Confirmation
Click 'Click Again to Confirm' to permanently delete:
sales_data.csv

[🔴 Click Again to Confirm]  [❌ Cancel]

↓ (User clicks again)

✅ File Deleted
Successfully deleted: sales_data.csv
```

## 🔒 Security Features

### 1. **User Isolation**
- Users can only see/access their own files
- `file_id` includes user_id for verification
- Permission checks on every operation

### 2. **Size Limits**
- Upload limit: 50MB per file
- Download limit: 25MB (Discord non-nitro)
- Prevents storage abuse

### 3. **Expiration** (if enabled)
- Files auto-delete after configured time
- Prevents indefinite storage buildup
- Can be disabled with `-1`

### 4. **2-Step Delete Confirmation**
- Prevents accidental deletions
- User must confirm twice
- 30-second timeout on confirmation

### 5. **File Type Validation**
- Detects file type from extension
- Supports 80+ file formats
- Type-specific emojis for clarity

## 🛠️ Integration with Tools

### Code Interpreter

```python
# Files are automatically available
import pandas as pd

# Load file by ID
df = load_file('file_id_here')

# Process data
df_cleaned = df.dropna()
df_cleaned.to_csv('cleaned_data.csv')

# Generate visualizations
import matplotlib.pyplot as plt
df.plot()
plt.savefig('chart.png')
```

### Data Analysis Tool

```python
# Works with any data file format
analyze_data_file(
    file_path='file_id_here',  # Can use file_id
    analysis_type='comprehensive'
)
```

### Custom Tools

All tools can access user files via `load_file('file_id')` function.

## 📊 Comparison: Expiration Settings

| Setting | FILES_EXPIRATION_HOURS | Use Case | Storage |
|---------|----------------------|----------|---------|
| **Short** | 24 | Quick analyses | Minimal |
| **Default** | 48 | General use | Low |
| **Extended** | 168 (7 days) | Project work | Medium |
| **Permanent** | -1 | Important data | Grows over time |

### Recommendations

**For Public Bots**: Use 48 hours to prevent storage buildup

**For Personal Use**: Use -1 (permanent) for convenience

**For Projects**: Use 168 hours (7 days) for active work

## 🚀 Quick Start

### 1. Set Up Environment

```bash
# Edit .env file
echo "FILE_EXPIRATION_HOURS=48" >> .env
```

### 2. Restart Bot

```bash
python3 bot.py
```

### 3. Upload a File

Attach any file to a Discord message and send it to the bot.

### 4. List Files

Use `/files` command to see all your files.

### 5. Download or Delete

Select a file from the dropdown and use the buttons.

## 📝 Command Reference

| Command | Description | Usage |
|---------|-------------|-------|
| `/files` | List all your uploaded files | `/files` |

That's it! Only one command needed. All other actions are done through the interactive UI (dropdowns and buttons).

## 🎯 Best Practices

### For Users

1. **Use descriptive filenames** - Makes files easier to identify
2. **Check `/files` regularly** - See what files you have
3. **Delete old files** - Keep your storage clean (if not permanent)
4. **Reference by file_id** - More reliable than filename

### For Developers

1. **Set appropriate expiration** - Balance convenience vs storage
2. **Monitor disk usage** - Especially with permanent storage
3. **Log file operations** - Track uploads/deletes for debugging
4. **Handle large files** - Some may exceed download limits

## 🐛 Troubleshooting

### File Not Found
**Error**: "File not found or expired"
**Solution**: Check if file expired, re-upload if needed

### Download Failed
**Error**: "File too large to download"
**Solution**: File >25MB, but still usable in code execution

### Delete Not Working
**Error**: Various
**Solution**: Check logs, ensure 2-step confirmation completed

### Files Not Expiring
**Check**: `FILE_EXPIRATION_HOURS` in .env
**Fix**: Make sure it's not set to `-1`

### Files Expiring Too Fast
**Check**: `FILE_EXPIRATION_HOURS` value
**Fix**: Increase the value or set to `-1`

## 📞 API Reference

### Functions Available

```python
# List user's files
files = await list_user_files(user_id, db_handler)

# Get file metadata
metadata = await get_file_metadata(file_id, user_id, db_handler)

# Delete file
result = await delete_file(file_id, user_id, db_handler)

# Load file in code
data = load_file('file_id')  # Available in code execution
```

## ✅ Summary

This file management system provides:

- ✅ **Single command**: `/files` for everything
- ✅ **Interactive UI**: Dropdowns and buttons for actions
- ✅ **2-step deletion**: Prevents accidental data loss
- ✅ **Configurable expiration**: 48h default or permanent
- ✅ **Universal access**: All tools can use files
- ✅ **Automatic tracking**: Files tracked in MongoDB
- ✅ **Secure**: User isolation and permission checks
- ✅ **Efficient**: Metadata in DB, files on disk

Users get a ChatGPT-like file management experience with simple Discord commands!
