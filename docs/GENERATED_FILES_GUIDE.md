# Generated Files - Complete Guide

## 📝 Overview

The code interpreter now captures **ALL file types** generated during code execution, not just images. All generated files:
- ✅ Are saved with **48-hour expiration** (same as uploaded files)
- ✅ Are **user-specific** (only accessible by the creator)
- ✅ Can be **referenced by file_id** in subsequent code executions
- ✅ Are **automatically sent to Discord** after execution
- ✅ Are **cleaned up automatically** after 48 hours

---

## 🎯 Key Features

### **1. Comprehensive File Type Support**

The system now captures **80+ file extensions** across all categories:

| Category | File Types | Use Cases |
|----------|-----------|-----------|
| **Images** | `.png`, `.jpg`, `.gif`, `.svg`, `.bmp` | Charts, plots, diagrams |
| **Data** | `.csv`, `.xlsx`, `.tsv`, `.parquet` | Exported datasets, analysis results |
| **Text** | `.txt`, `.md`, `.log`, `.out` | Reports, logs, documentation |
| **Structured** | `.json`, `.xml`, `.yaml`, `.toml` | Config files, API responses |
| **HTML** | `.html`, `.htm` | Interactive reports, dashboards |
| **PDF** | `.pdf` | Formatted reports |
| **Code** | `.py`, `.js`, `.sql`, `.r` | Generated scripts |
| **Archive** | `.zip`, `.tar`, `.gz` | Bundled outputs |
| **Database** | `.db`, `.sqlite`, `.sql` | Database files |
| **Scientific** | `.npy`, `.npz`, `.hdf5`, `.pickle` | NumPy arrays, ML models |

### **2. 48-Hour File Lifecycle**

```
Code Execution → File Created → Saved to Database → Available for 48h → Auto-deleted
       ↓              ↓                ↓                    ↓               ↓
  User runs code   file.txt      file_id created    User can access    Cleanup removes
                   generated     in MongoDB         via file_id         expired file
```

### **3. File Access Methods**

#### **Method A: Immediate Access (Discord)**
Files are automatically sent to Discord right after execution:
```python
# User gets files immediately as Discord attachments
# No need to do anything - automatic!
```

#### **Method B: Access by file_id (Within 48 hours)**
Users can reference generated files in subsequent code:
```python
# First execution - generates file
result1 = await execute_code(
    code="df.to_csv('analysis.csv', index=False)",
    user_id=123
)
# result1["generated_file_ids"] = ["123_1696118400_a1b2c3d4"]

# Second execution - loads previously generated file
result2 = await execute_code(
    code="""
    # Load the file we generated earlier
    df = load_file('123_1696118400_a1b2c3d4')
    print(df.head())
    """,
    user_id=123,
    user_files=["123_1696118400_a1b2c3d4"]
)
```

#### **Method C: List User Files**
```python
files = await list_user_files(user_id=123, db_handler=db)
# Returns all non-expired files (uploaded + generated)
```

#### **Method D: Load File Manually**
```python
file_data = await load_file(
    file_id="123_1696118400_a1b2c3d4",
    user_id=123,
    db_handler=db
)
# Returns: {"success": True, "data": b"...", "filename": "analysis.csv", ...}
```

---

## 💡 Usage Examples

### **Example 1: Generate Multiple File Types**

```python
code = """
import pandas as pd
import matplotlib.pyplot as plt
import json

# Create sample data
df = pd.DataFrame({
    'product': ['A', 'B', 'C', 'D'],
    'sales': [1000, 1500, 1200, 1800],
    'profit': [200, 300, 240, 360]
})

# 1. Generate CSV export
df.to_csv('sales_data.csv', index=False)

# 2. Generate JSON summary
summary = {
    'total_sales': df['sales'].sum(),
    'total_profit': df['profit'].sum(),
    'avg_profit_margin': (df['profit'].sum() / df['sales'].sum()) * 100
}
with open('summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# 3. Generate chart
plt.figure(figsize=(10, 6))
plt.bar(df['product'], df['sales'])
plt.title('Sales by Product')
plt.xlabel('Product')
plt.ylabel('Sales ($)')
plt.tight_layout()
plt.savefig('sales_chart.png', dpi=150)

# 4. Generate detailed report
with open('report.txt', 'w') as f:
    f.write('SALES ANALYSIS REPORT\\n')
    f.write('=' * 50 + '\\n\\n')
    f.write(f'Total Sales: ${summary["total_sales"]:,.2f}\\n')
    f.write(f'Total Profit: ${summary["total_profit"]:,.2f}\\n')
    f.write(f'Profit Margin: {summary["avg_profit_margin"]:.2f}%\\n\\n')
    f.write('Product Details:\\n')
    f.write(df.to_string(index=False))

print('Analysis complete! Generated 4 files.')
"""

result = await execute_code(code=code, user_id=123, db_handler=db)

# Result contains:
{
    "success": True,
    "output": "Analysis complete! Generated 4 files.",
    "generated_files": [
        {"filename": "sales_data.csv", "type": "data", "size": 142, "file_id": "123_..."},
        {"filename": "summary.json", "type": "structured", "size": 189, "file_id": "123_..."},
        {"filename": "sales_chart.png", "type": "image", "size": 28456, "file_id": "123_..."},
        {"filename": "report.txt", "type": "text", "size": 523, "file_id": "123_..."}
    ],
    "generated_file_ids": ["123_...", "123_...", "123_...", "123_..."]
}
```

**User receives in Discord:**
```
✅ Execution succeeded!
```
Analysis complete! Generated 4 files.
```

📎 Generated 4 file(s):
• sales_data.csv (data, 0.1 KB)
• summary.json (structured, 0.2 KB)
• sales_chart.png (image, 27.8 KB)
• report.txt (text, 0.5 KB)

📊 sales_data.csv [downloadable]
📋 summary.json [downloadable]
🖼️ sales_chart.png [downloadable]
📝 report.txt [downloadable]

⏱️ Executed in 2.45s
```

### **Example 2: Reuse Generated Files**

```python
# Day 1, 10:00 AM - User generates analysis
code1 = """
import pandas as pd
df = pd.DataFrame({'x': range(100), 'y': range(100, 200)})
df.to_csv('dataset.csv', index=False)
print('Dataset created!')
"""

result1 = await execute_code(code=code1, user_id=123)
# result1["generated_file_ids"] = ["123_1696118400_abc123"]

# Day 1, 11:30 AM - User wants to continue working with that file
code2 = """
# Load the previously generated file
df = load_file('123_1696118400_abc123')
print(f'Loaded dataset with {len(df)} rows')

# Create visualization
import matplotlib.pyplot as plt
plt.scatter(df['x'], df['y'])
plt.title('X vs Y')
plt.savefig('scatter_plot.png')
print('Chart created!')
"""

result2 = await execute_code(
    code=code2,
    user_id=123,
    user_files=["123_1696118400_abc123"]  # Pass the file_id
)

# Day 3, 10:01 AM - File expires (48 hours passed)
# User tries to load it again
result3 = await execute_code(
    code="df = load_file('123_1696118400_abc123')",
    user_id=123,
    user_files=["123_1696118400_abc123"]
)
# Returns error: "File not found or expired"
```

### **Example 3: Export Complex Data**

```python
code = """
import pandas as pd
import numpy as np

# Generate complex dataset
np.random.seed(42)
data = {
    'date': pd.date_range('2024-01-01', periods=365),
    'sales': np.random.randint(1000, 5000, 365),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
    'product': np.random.choice(['A', 'B', 'C'], 365)
}
df = pd.DataFrame(data)

# Export in multiple formats for different use cases

# 1. CSV for Excel users
df.to_csv('sales_2024.csv', index=False)

# 2. Parquet for data scientists (smaller, faster)
df.to_parquet('sales_2024.parquet')

# 3. JSON for web developers
df.to_json('sales_2024.json', orient='records', indent=2)

# 4. Excel with multiple sheets
with pd.ExcelWriter('sales_2024.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='All Sales', index=False)
    df.groupby('region').sum().to_excel(writer, sheet_name='By Region')
    df.groupby('product').sum().to_excel(writer, sheet_name='By Product')

# 5. Summary statistics as text
with open('summary.txt', 'w') as f:
    f.write(df.describe().to_string())

print('Exported to 5 different formats!')
"""

result = await execute_code(code=code, user_id=123)
# All 5 files are captured, saved with 48h expiration, and sent to Discord
```

---

## 🔧 Integration with Message Handler

### **Update Your Message Handler:**

```python
async def handle_code_execution_result(message, exec_result):
    """Send execution results and generated files to Discord."""
    
    if not exec_result["success"]:
        await message.channel.send(f"❌ Error: {exec_result['error']}")
        return
    
    # Send output
    if exec_result.get("output"):
        output = exec_result["output"]
        if len(output) > 1900:
            # Too long, send as file
            output_file = io.BytesIO(output.encode('utf-8'))
            await message.channel.send(
                "📄 Output:",
                file=discord.File(output_file, filename="output.txt")
            )
        else:
            await message.channel.send(f"```\n{output}\n```")
    
    # Send generated files
    generated_files = exec_result.get("generated_files", [])
    
    if generated_files:
        # Summary
        summary = f"📎 **Generated {len(generated_files)} file(s):**\n"
        for gf in generated_files:
            size_kb = gf['size'] / 1024
            summary += f"• `{gf['filename']}` ({gf['type']}, {size_kb:.1f} KB)\n"
        summary += f"\n💾 Files available for 48 hours (expires {get_expiry_time()})"
        await message.channel.send(summary)
        
        # Send each file
        emojis = {
            "image": "🖼️", "data": "📊", "text": "📝",
            "structured": "📋", "html": "🌐", "pdf": "📄",
            "code": "💻", "archive": "📦", "file": "📎"
        }
        
        for gf in generated_files:
            try:
                file_bytes = io.BytesIO(gf["data"])
                discord_file = discord.File(file_bytes, filename=gf["filename"])
                emoji = emojis.get(gf["type"], "📎")
                
                # Include file_id for user reference
                await message.channel.send(
                    f"{emoji} `{gf['filename']}` (ID: `{gf['file_id']}`)",
                    file=discord_file
                )
            except Exception as e:
                logger.error(f"Failed to send {gf['filename']}: {e}")
    
    # Execution stats
    stats = f"⏱️ Executed in {exec_result['execution_time']:.2f}s"
    if exec_result.get("installed_packages"):
        stats += f"\n📦 Auto-installed: {', '.join(exec_result['installed_packages'])}"
    await message.channel.send(stats)
```

---

## 🗂️ File Management Commands

### **List User Files**

```python
@bot.command(name="myfiles")
async def list_files_command(ctx):
    """List all user's files (uploaded + generated)."""
    files = await list_user_files(ctx.author.id, db_handler=db)
    
    if not files:
        await ctx.send("📁 You have no files.")
        return
    
    msg = f"📁 **Your Files ({len(files)} total):**\n\n"
    for f in files:
        size_kb = f['file_size'] / 1024
        expires = datetime.fromisoformat(f['expires_at'])
        hours_left = (expires - datetime.now()).total_seconds() / 3600
        
        msg += f"• `{f['filename']}`\n"
        msg += f"  ID: `{f['file_id']}`\n"
        msg += f"  Type: {f['file_type']} | Size: {size_kb:.1f} KB\n"
        msg += f"  ⏰ Expires in {hours_left:.1f} hours\n\n"
    
    await ctx.send(msg)
```

### **Download Specific File**

```python
@bot.command(name="download")
async def download_file_command(ctx, file_id: str):
    """Download a specific file by ID."""
    result = await load_file(file_id, ctx.author.id, db_handler=db)
    
    if not result["success"]:
        await ctx.send(f"❌ {result['error']}")
        return
    
    file_bytes = io.BytesIO(result["data"])
    discord_file = discord.File(file_bytes, filename=result["filename"])
    
    await ctx.send(
        f"📎 `{result['filename']}` ({result['file_type']}, {result['file_size']/1024:.1f} KB)",
        file=discord_file
    )
```

---

## 🧹 Automatic Cleanup

### **How It Works**

1. **Hourly Cleanup Task** (runs automatically)
   ```python
   # In bot.py
   cleanup_task = create_discord_cleanup_task(bot, db_handler)
   
   @bot.event
   async def on_ready():
       cleanup_task.start()
   ```

2. **What Gets Cleaned**
   - All files older than 48 hours (uploaded + generated)
   - Empty user directories
   - Stale database records

3. **Cleanup Logs**
   ```
   [Cleanup] Starting cleanup at 2024-10-01 12:00:00
   [Cleanup] Removed 15 expired files
   [Cleanup] Cleaned 3 empty directories
   [Cleanup] Cleanup completed in 1.23s
   ```

---

## 📊 System Status

### **Check Interpreter Status**

```python
status = await get_interpreter_status(db_handler=db)

# Returns:
{
    "venv_exists": True,
    "python_path": "/tmp/bot_code_interpreter/venv/bin/python",
    "installed_packages": ["numpy", "pandas", "matplotlib"],
    "package_count": 62,
    "last_cleanup": "2024-10-01T11:00:00",
    "total_user_files": 142,
    "total_file_size_mb": 256.7,
    "file_expiration_hours": 48,
    "max_file_size_mb": 50
}
```

---

## 🔒 Security Notes

1. **User Isolation**: Users can only access their own files
2. **Size Limits**: Max 50MB per file
3. **Auto-Expiration**: All files deleted after 48 hours
4. **No Permanent Storage**: Generated files are temporary
5. **Secure Paths**: Files stored in user-specific directories

---

## 🎯 Best Practices

1. **Reference Files by ID**: Save file_ids from execution results for later use
2. **Work Within 48 Hours**: Plan multi-step analysis within the expiration window
3. **Download Important Files**: Download files from Discord if you need them long-term
4. **Use Appropriate Formats**: Choose file formats based on use case (CSV for sharing, Parquet for performance)
5. **Clean Up Early**: Delete files you don't need with `delete_user_file()`

---

## 🚀 Summary

✅ **ALL file types** are now captured (80+ extensions)  
✅ **48-hour lifecycle** for generated files (same as uploads)  
✅ **User-specific** storage and access  
✅ **Automatic cleanup** every hour  
✅ **File IDs** for referencing in future executions  
✅ **Discord integration** for immediate file delivery  

Your code interpreter now works exactly like ChatGPT/Claude Code Interpreter! 🎉
