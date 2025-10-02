# Generated Files - Quick Reference

## 🎯 What Changed?

✅ **ALL file types** are now captured (not just images)  
✅ **48-hour expiration** for generated files  
✅ **file_id** for accessing files later  
✅ **80+ file extensions** supported  

---

## 📊 Execution Result Structure

```python
result = {
    "success": True,
    "output": "Analysis complete!",
    "error": "",
    "execution_time": 2.5,
    "return_code": 0,
    "generated_files": [          # Immediate data for Discord
        {
            "filename": "report.txt",
            "data": b"...",         # Binary content
            "type": "text",          # File category
            "size": 1234,           # Bytes
            "file_id": "123_..."    # For later access ← NEW!
        }
    ],
    "generated_file_ids": [       # Quick reference ← NEW!
        "123_1696118400_abc123",
        "123_1696118401_def456"
    ]
}
```

---

## 🔧 Key Functions

### **Execute Code**
```python
result = await execute_code(
    code="df.to_csv('data.csv')",
    user_id=123,
    db_handler=db
)
# Generated files automatically saved with 48h expiration
```

### **Load Generated File (Within 48h)**
```python
file_data = await load_file(
    file_id="123_1696118400_abc123",
    user_id=123,
    db_handler=db
)
# Returns: {"success": True, "data": b"...", "filename": "data.csv"}
```

### **List All Files**
```python
files = await list_user_files(user_id=123, db_handler=db)
# Returns all non-expired files (uploaded + generated)
```

### **Use File in Code**
```python
code = """
# Load previously generated file
df = load_file('123_1696118400_abc123')
print(f'Loaded {len(df)} rows')
"""

result = await execute_code(
    code=code,
    user_id=123,
    user_files=["123_1696118400_abc123"]
)
```

---

## 📁 Supported File Types (80+)

| Type | Extensions | Category |
|------|-----------|----------|
| **Images** | `.png`, `.jpg`, `.gif`, `.svg` | `"image"` |
| **Data** | `.csv`, `.xlsx`, `.parquet`, `.feather` | `"data"` |
| **Text** | `.txt`, `.md`, `.log` | `"text"` |
| **Structured** | `.json`, `.xml`, `.yaml` | `"structured"` |
| **Code** | `.py`, `.js`, `.sql`, `.r` | `"code"` |
| **Archive** | `.zip`, `.tar`, `.gz` | `"archive"` |
| **Scientific** | `.npy`, `.pickle`, `.hdf5` | Various |
| **HTML** | `.html`, `.htm` | `"html"` |
| **PDF** | `.pdf` | `"pdf"` |

Full list: See `GENERATED_FILES_GUIDE.md`

---

## ⏰ File Lifecycle

```
Create → Save → Available 48h → Auto-Delete
  ↓       ↓          ↓              ↓
Code   Database   Use file_id    Cleanup
runs    record    to access       task
```

**Timeline Example:**
- Day 1, 10:00 AM: File created
- Day 1-3: File accessible via `file_id`
- Day 3, 10:01 AM: File expires and is auto-deleted

---

## 💡 Common Patterns

### **Pattern 1: Multi-Format Export**
```python
code = """
df.to_csv('data.csv')
df.to_json('data.json')
df.to_excel('data.xlsx')
print('Exported to 3 formats!')
"""
```

### **Pattern 2: Reuse Generated File**
```python
# Step 1: Generate
result1 = await execute_code(
    code="df.to_csv('results.csv')",
    user_id=123
)
file_id = result1["generated_file_ids"][0]

# Step 2: Reuse (within 48h)
result2 = await execute_code(
    code=f"df = load_file('{file_id}')",
    user_id=123,
    user_files=[file_id]
)
```

### **Pattern 3: Multi-Step Analysis**
```python
# Day 1: Generate dataset
code1 = "df.to_parquet('dataset.parquet')"
result1 = await execute_code(code1, user_id=123)

# Day 2: Analyze (file still valid)
code2 = """
df = load_file('123_...')  # Use file_id from result1
# Perform analysis
"""
result2 = await execute_code(code2, user_id=123, user_files=['123_...'])
```

---

## 🎨 Discord Integration

```python
# Send files to user
for gen_file in result["generated_files"]:
    file_bytes = io.BytesIO(gen_file["data"])
    discord_file = discord.File(file_bytes, filename=gen_file["filename"])
    
    # Include file_id for user reference
    await message.channel.send(
        f"📎 `{gen_file['filename']}` (ID: `{gen_file['file_id']}`)",
        file=discord_file
    )
```

**User sees:**
```
📎 analysis.csv (ID: 123_1696118400_abc123) [downloadable]
📊 chart.png (ID: 123_1696118401_def456) [downloadable]
📝 report.txt (ID: 123_1696118402_ghi789) [downloadable]

💾 Files available for 48 hours
```

---

## 🧹 Cleanup

**Automatic (Every Hour):**
```python
# In bot.py
cleanup_task = create_discord_cleanup_task(bot, db_handler)

@bot.event
async def on_ready():
    cleanup_task.start()
```

**Manual:**
```python
deleted = await cleanup_expired_files(db_handler)
print(f"Deleted {deleted} expired files")
```

---

## 🔒 Security

✅ User isolation (can't access other users' files)  
✅ 50MB max file size  
✅ 48-hour auto-expiration  
✅ User-specific directories  
✅ No permanent storage  

---

## 📚 Full Documentation

- **GENERATED_FILES_GUIDE.md** - Complete usage guide
- **GENERATED_FILES_UPDATE_SUMMARY.md** - Technical changes
- **CODE_INTERPRETER_GUIDE.md** - General code interpreter docs
- **NEW_FEATURES_GUIDE.md** - All new features

---

## ✅ Status

- [x] All file types captured
- [x] 48-hour persistence implemented
- [x] file_id system working
- [x] Database integration complete
- [x] Automatic cleanup configured
- [x] Documentation created
- [ ] **Ready for production testing!**

---

## 🚀 Quick Start

```python
# 1. Execute code that generates files
result = await execute_code(
    code="""
    import pandas as pd
    df = pd.DataFrame({'x': [1,2,3]})
    df.to_csv('data.csv')
    df.to_json('data.json')
    print('Files created!')
    """,
    user_id=123,
    db_handler=db
)

# 2. Files are automatically:
#    - Saved to database (48h expiration)
#    - Sent to Discord
#    - Accessible via file_id

# 3. Use later (within 48h)
code2 = f"df = load_file('{result['generated_file_ids'][0]}')"
result2 = await execute_code(code2, user_id=123, user_files=[...])
```

That's it! Your code interpreter now handles **all file types** with **48-hour persistence**! 🎉
