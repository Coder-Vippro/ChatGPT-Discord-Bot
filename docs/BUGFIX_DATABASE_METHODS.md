# Bug Fix: Missing Database Methods

## Issue
The bot was crashing with the error:
```
'DatabaseHandler' object has no attribute 'get_user_files'
```

## Root Cause
The `message_handler.py` was calling `db.get_user_files()` but this method didn't exist in the `DatabaseHandler` class. The database had a `user_files` collection with indexes defined, but no methods to interact with it.

## Solution
Added four new methods to `DatabaseHandler` class in `src/database/db_handler.py`:

### 1. `get_user_files(user_id: int) -> List[Dict[str, Any]]`
**Purpose**: Retrieve all non-expired files for a specific user

**Features**:
- Filters out expired files (expires_at < current_time)
- Handles files with no expiration (expires_at = None)
- Returns empty list on error

**Usage**:
```python
user_files = await db.get_user_files(user_id)
file_ids = [f['file_id'] for f in user_files]
```

### 2. `save_user_file(file_data: Dict[str, Any]) -> None`
**Purpose**: Save or update a user file record in the database

**Features**:
- Uses upsert (update or insert)
- Updates by file_id
- Stores complete file metadata

**Expected file_data format**:
```python
{
    "file_id": "unique_file_id",
    "user_id": 123456789,
    "filename": "data.csv",
    "file_type": "csv",
    "file_path": "/tmp/bot_code_interpreter/user_files/123456789/data.csv",
    "size": 1024,
    "created_at": datetime.now(),
    "expires_at": datetime.now() + timedelta(hours=48)  # or None
}
```

### 3. `delete_user_file(file_id: str) -> bool`
**Purpose**: Delete a specific file record from the database

**Returns**: True if file was deleted, False otherwise

**Usage**:
```python
success = await db.delete_user_file(file_id)
```

### 4. `delete_expired_files() -> int`
**Purpose**: Cleanup task to remove all expired file records

**Returns**: Number of deleted records

**Usage** (for scheduled cleanup):
```python
deleted_count = await db.delete_expired_files()
logging.info(f"Cleaned up {deleted_count} expired files")
```

## Files Modified

### src/database/db_handler.py
- **Lines Added**: ~60 lines (4 new methods)
- **Location**: After `reset_user_token_stats()` method
- **Dependencies**: Uses existing `datetime`, `timedelta`, `logging` imports

### src/module/message_handler.py
- **Lines 299-302**: Added variable assignments for display purposes
  ```python
  packages_to_install = install_packages  # For display
  input_data = args.get("input_data", "")  # For display
  ```

## Testing

### Verification Commands
```bash
# Compile check
python3 -m py_compile src/database/db_handler.py
python3 -m py_compile src/module/message_handler.py

# Run bot
python3 bot.py
```

### Test Cases
1. ✅ Upload a file to Discord
   - File should be saved with file_id
   - Record stored in user_files collection
   
2. ✅ Execute Python code with file access
   - `get_user_files()` retrieves all user files
   - Code can use `load_file(file_id)` 
   
3. ✅ File expiration
   - Files older than FILE_EXPIRATION_HOURS are filtered out
   - `delete_expired_files()` can clean up old records

4. ✅ User file limit
   - When MAX_FILES_PER_USER is reached
   - Oldest file is deleted before new upload

## Database Schema

### user_files Collection
```javascript
{
  "_id": ObjectId("..."),
  "file_id": "file_123456789_1234567890",  // Unique identifier
  "user_id": 123456789,                    // Discord user ID
  "filename": "data.csv",                  // Original filename
  "file_type": "csv",                      // Detected file type
  "file_path": "/tmp/.../file.csv",        // Full file path
  "size": 1024,                            // File size in bytes
  "created_at": ISODate("..."),           // Upload timestamp
  "expires_at": ISODate("...")            // Expiration time (or null)
}
```

### Indexes
```javascript
// Compound index for user queries with expiration
{ "user_id": 1, "expires_at": -1 }

// Unique index for file_id lookups
{ "file_id": 1 } // unique: true

// Index for cleanup queries
{ "expires_at": 1 }
```

## Configuration

### Environment Variables (.env)
```bash
FILE_EXPIRATION_HOURS=48   # Files expire after 48 hours (-1 = never)
MAX_FILES_PER_USER=20      # Maximum files per user
```

### How It Works
1. **Upload**: User uploads file → `save_user_file()` creates record
2. **Access**: Code execution → `get_user_files()` retrieves file_ids
3. **Load**: Python code calls `load_file(file_id)` → file loaded into memory
4. **Expire**: After 48 hours → file filtered out by `get_user_files()`
5. **Cleanup**: Periodic task → `delete_expired_files()` removes old records

## Impact
- ✅ **Fixed**: `'DatabaseHandler' object has no attribute 'get_user_files'` error
- ✅ **Added**: Complete file management system
- ✅ **Enabled**: Per-user file limits with automatic cleanup
- ✅ **Enabled**: File expiration system
- ✅ **Enabled**: Code interpreter file access

## Related Documentation
- [FILE_STORAGE_AND_CONTEXT_MANAGEMENT.md](FILE_STORAGE_AND_CONTEXT_MANAGEMENT.md)
- [UNIFIED_FILE_SYSTEM_SUMMARY.md](UNIFIED_FILE_SYSTEM_SUMMARY.md)
- [CODE_INTERPRETER_GUIDE.md](CODE_INTERPRETER_GUIDE.md)
