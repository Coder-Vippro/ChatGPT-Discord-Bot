# Quick Reference: File Management

## 📱 Single Command

```
/files → List + Download + Delete
```

## 🎯 Key Features

✅ **Upload**: Attach file to message (automatic)
✅ **List**: `/files` command (interactive UI)
✅ **Download**: Select file → Click download button
✅ **Delete**: Select file → Click delete (2-step confirmation)
✅ **AI Access**: All tools can use `load_file('file_id')`

## ⚙️ Configuration (.env)

```bash
# Expire after 48 hours (default)
FILE_EXPIRATION_HOURS=48

# Never expire (permanent storage)
FILE_EXPIRATION_HOURS=-1

# Custom duration
FILE_EXPIRATION_HOURS=168  # 7 days
```

## 💡 Quick Examples

### Upload & Use
```
1. Attach data.csv to message
2. Get file_id: 123456789_...
3. In code: df = load_file('123456789_...')
```

### List Files
```
/files
→ Shows all files with dropdown menu
→ Click file → Download or Delete
```

### Delete (2-Step)
```
/files → Select file → Delete
→ Confirm #1: "Yes, Delete"
→ Confirm #2: "Click Again to Confirm"
→ Deleted!
```

### Reset All
```
/reset
→ Clears conversation history
→ Resets token statistics
→ Deletes ALL files (disk + database)
→ Complete fresh start!
```

## 🔄 File Lifecycle

**With Expiration (48h)**:
```
Upload → 48h Available → Auto-Delete
```

**Permanent Storage (-1)**:
```
Upload → Forever Available → Manual Delete Only
```

## 📊 Supported Files (80+)

- 📊 Data: CSV, Excel, JSON, Parquet
- 🖼️ Images: PNG, JPG, GIF, SVG
- 📝 Text: TXT, MD, PDF, DOCX
- 💻 Code: PY, JS, TS, HTML, SQL
- 🗄️ Database: SQLite, SQL files
- 📦 Archives: ZIP, TAR, GZ

## 🔒 Security

- ✅ User isolation (can't see others' files)
- ✅ Size limits (50MB upload, 25MB download)
- ✅ 2-step delete confirmation
- ✅ Optional auto-expiration

## 🎨 UI Flow

```
/files Command
    ↓
📁 Your Files List
    ↓
[Dropdown: Select file]
    ↓
[Download Button] [Delete Button]
    ↓
Action completed!
```

## 🛠️ Integration

**In Python Code**:
```python
df = load_file('file_id')  # Load user file
```

**Available to ALL tools**:
- execute_python_code ✅
- analyze_data_file ✅
- Custom tools ✅

## 📝 Best Practices

1. Use `/files` to check what you have
2. Delete old files you don't need
3. Set appropriate expiration in .env
4. Use descriptive filenames
5. Reference by file_id in code

## 🎯 Summary

**Command**: `/files`
**Actions**: List, Download, Delete (2-step)
**Storage**: Disk (files) + MongoDB (metadata)
**Expiration**: Configurable (.env)
**Access**: All tools via `load_file()`

---

**See full guide**: `docs/FILE_MANAGEMENT_GUIDE.md`
