# Data Analysis Fix - UnboundLocalError

## 🐛 Problem

```
UnboundLocalError: cannot access local variable 'file_path' where it is not associated with a value
```

Occurred at line 557 in `message_handler.py` during data file analysis.

## 🔍 Root Cause

Variable `file_path` was used **before** it was assigned:

```python
# Line 557: Used here ❌
if file_path and not file_path.startswith('/tmp/bot_code_interpreter'):

# Line 583: Assigned here ❌
file_path = args.get("file_path", "")
```

The variable was referenced 26 lines before being defined!

## ✅ Solution

### Fix 1: Reorder Variable Assignments

**Before:**
```python
from src.utils.code_interpreter import execute_code

# ❌ Using file_path before assignment
if file_path and not file_path.startswith('/tmp/bot_code_interpreter'):
    # migration code...

# ❌ Assignment comes too late
file_path = args.get("file_path", "")
```

**After:**
```python
from src.utils.code_interpreter import execute_code

# ✅ Assign variables first
file_path = args.get("file_path", "")
analysis_type = args.get("analysis_type", "")
custom_analysis = args.get("custom_analysis", "")

# ✅ Now can safely use file_path
if file_path and not file_path.startswith('/tmp/bot_code_interpreter'):
    # migration code...
```

### Fix 2: Smart File Type Detection

Added automatic detection of file types for proper loading:

```python
# Detect file type based on extension
file_ext = os.path.splitext(file_path)[1].lower()

if file_ext in ['.xlsx', '.xls']:
    load_statement = f"df = pd.read_excel('{file_path}')"
elif file_ext == '.json':
    load_statement = f"df = pd.read_json('{file_path}')"
elif file_ext == '.parquet':
    load_statement = f"df = pd.read_parquet('{file_path}')"
else:  # Default to CSV
    load_statement = f"df = pd.read_csv('{file_path}')"
```

## 📊 Supported File Types

| Extension | Pandas Reader | Status |
|-----------|---------------|--------|
| `.csv` | `pd.read_csv()` | ✅ Working |
| `.xlsx`, `.xls` | `pd.read_excel()` | ✅ Working |
| `.json` | `pd.read_json()` | ✅ Working |
| `.parquet` | `pd.read_parquet()` | ✅ Working |
| Other | `pd.read_csv()` | ✅ Default |

## 🔄 Execution Flow

```
User uploads data.xlsx
    ↓
Bot receives file
    ↓
Assigns file_path variable ✅
    ↓
Checks if migration needed
    ↓
Detects file type (.xlsx)
    ↓
Generates: df = pd.read_excel(file_path)
    ↓
Executes via code_interpreter
    ↓
Returns analysis results
```

## 🧪 Testing

### Test Case 1: CSV File
```
1. Upload data.csv
2. Ask for analysis
3. ✅ Loads with pd.read_csv()
4. ✅ Shows statistics
```

### Test Case 2: Excel File
```
1. Upload report.xlsx
2. Ask for analysis
3. ✅ Detects .xlsx extension
4. ✅ Loads with pd.read_excel()
5. ✅ Shows statistics
```

### Test Case 3: JSON File
```
1. Upload data.json
2. Ask for analysis
3. ✅ Detects .json extension
4. ✅ Loads with pd.read_json()
5. ✅ Shows statistics
```

## 🎯 Result

✅ **Fixed UnboundLocalError**
✅ **All file types supported**
✅ **Proper file type detection**
✅ **Clean execution through code_interpreter**

---

**Date**: October 2, 2025
**File**: `src/module/message_handler.py`
**Lines**: 555-598
**Status**: ✅ Fixed
