# Quick Reference: File Storage & Context Management

## 📁 File Storage TL;DR

```
Non-Images → Disk (/tmp/bot_code_interpreter/user_files/)
MongoDB → Only metadata (file_id, path, size, timestamps)
Images → Discord CDN links only
Expiration → 48 hours, auto-cleanup
```

## 🔢 Token Limits (config.py)

```python
gpt-4o: 8000
gpt-4.1: 8000
o1/o3/o4: 4000
gpt-5: 4000
Default: 4000
```

## 🔄 Context Management

**Strategy**: Sliding window (like ChatGPT)
- Keep: System prompt + recent messages
- Group: User+Assistant pairs together
- Trim: Oldest-first when over limit
- No summarization: Zero extra API calls

**Token Budget**:
- System: Always included
- Conversation: 80% of available
- Response: 20% reserved

## 📊 Key Improvements

| Metric | Old | New | Improvement |
|--------|-----|-----|-------------|
| DB Size (100 files) | 200MB | 50KB | 99.97% ↓ |
| Context Method | Fixed limits | Model-specific | Configurable |
| Pairing | None | User+Asst | Coherent |
| API Calls | Extra for summary | None | Free |

## 💻 Code Examples

### Upload File
```python
result = await upload_discord_attachment(attachment, user_id, db)
# Returns: {"file_id": "...", "file_path": "..."}
```

### Use in Code
```python
df = load_file('file_id')  # Auto-loads from disk
df.to_csv('output.csv')    # Auto-captured
```

### Generated Files
```python
result["generated_files"] = [
    {
        "filename": "chart.png",
        "data": b"...",
        "type": "image",
        "file_id": "..."
    }
]
```

## ⚙️ Configuration

Edit `src/config/config.py`:
```python
MODEL_TOKEN_LIMITS = {
    "openai/gpt-4.1": 8000,  # Adjust here
}
```

## 🔍 Monitoring

```bash
# Log output shows:
Sliding window trim: 45 → 28 messages (17 removed, ~3200/4000 tokens)
Saved file sales.csv for user 123: file_id
```

## 🚨 Common Issues

**File expired**: Re-upload (48h limit)
**Context too large**: Automatic trim
**Disk full**: Check cleanup task

## 📖 Full Documentation

See: `docs/FILE_STORAGE_AND_CONTEXT_MANAGEMENT.md`
