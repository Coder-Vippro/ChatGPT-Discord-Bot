# Discord Message Length Fix

## Problem

Discord has a **2000 character limit** for messages. The bot was displaying code execution results without properly checking the total message length, causing this error:

```
400 Bad Request (error code: 50035): Invalid Form Body
In content: Must be 2000 or fewer in length.
```

## Root Cause

The code was truncating individual parts (code, output, errors) but not checking the **combined total length** before sending. Even with truncated parts, the message could exceed 2000 characters when combined.

### Example of the Issue:

```python
# Each part was truncated individually:
execution_display += packages  # 100 chars
execution_display += input_data[:500]  # 500 chars
execution_display += code  # 800 chars
execution_display += output[:1000]  # 1000 chars
# Total: 2400 chars → EXCEEDS LIMIT! ❌
```

## Solution

Implemented **dynamic length calculation** that:

1. **Calculates remaining space** before adding output/errors
2. **Adjusts content length** based on what's already in the message
3. **Final safety check** ensures total message < 2000 chars

### Changes Made

**File**: `src/module/message_handler.py`

#### Before:
```python
# Fixed truncation without considering total length
execution_display += output[:1000]  # ❌ Doesn't consider existing content
```

#### After:
```python
# Dynamic truncation based on remaining space
remaining = 1900 - len(execution_display)  # ✅ Calculate available space
if remaining > 100:
    execution_display += output[:remaining]
    if len(output) > remaining:
        execution_display += "\n... (output truncated)"
else:
    execution_display += "(output too long)"

# Final safety check
if len(execution_display) > 1990:
    execution_display = execution_display[:1980] + "\n...(truncated)"
```

## Implementation Details

### Two Display Scenarios:

#### 1. **Normal Display** (code < 3000 chars)
```python
execution_display = "🐍 Python Code Execution\n\n"
+ packages (if any)
+ input_data (max 500 chars)
+ code (full, up to 3000 chars)
+ output (remaining space, min 100 chars)
+ final_check (ensure < 2000 total)
```

#### 2. **File Attachment Display** (code >= 3000 chars)
```python
execution_display = "🐍 Python Code Execution\n\n"
+ packages (if any)
+ input_data (max 500 chars)
+ "Code: *Attached as file*"
+ output (remaining space, min 100 chars)
+ final_check (ensure < 2000 total)
# Code sent as separate .py file attachment
```

### Smart Truncation Strategy:

1. **Priority Order** (most to least important):
   - Header & metadata (packages, input info)
   - Code (inline or file attachment)
   - Output/Errors (dynamically sized)

2. **Space Allocation**:
   - Reserve 1900 chars (100 char buffer)
   - Calculate: `remaining = 1900 - len(current_content)`
   - Only add output/errors if `remaining > 100`

3. **Safety Net**:
   - Final check: `if len(message) > 1990`
   - Hard truncate at 1980 with "...(truncated)"

## Benefits

✅ **No More Discord Errors**: Messages never exceed 2000 char limit
✅ **Smart Truncation**: Prioritizes most important information
✅ **Better UX**: Users see as much as possible within limits
✅ **Graceful Degradation**: Long content becomes file attachments
✅ **Clear Indicators**: Shows when content is truncated

## Testing

To test the fix:

1. **Short code + long output**: Should display inline with truncated output
2. **Long code + short output**: Code as file, output inline
3. **Long code + long output**: Code as file, output truncated
4. **Very long error messages**: Should truncate gracefully

Example test case:
```python
# Generate long output
for i in range(1000):
    print(f"Line {i}: " + "x" * 100)
```

Before: ❌ Discord 400 error
After: ✅ Displays with "(output truncated)" indicator

## Related Files

- `src/module/message_handler.py` (Lines 400-480)
  - Fixed both normal display and file attachment display
  - Added dynamic length calculation
  - Added final safety check

## Prevention

To prevent similar issues in the future:

1. **Always calculate remaining space** before adding variable-length content
2. **Use final safety check** before sending to Discord
3. **Test with extreme cases** (very long code, output, errors)
4. **Consider file attachments** for content that might exceed limits

## Discord Limits Reference

- **Message content**: 2000 characters max
- **Embed description**: 4096 characters max
- **Embed field value**: 1024 characters max
- **Code blocks**: Count toward message limit

**Note**: We use 1990 as safe limit (10 char buffer) to account for markdown formatting and edge cases.
