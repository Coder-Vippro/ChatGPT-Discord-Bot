# Current Time in Chat Context

## Feature Overview

The AI model now always knows the current date and time in every conversation! The system automatically includes the current datetime with your configured timezone at the beginning of each message context.

## How It Works

### Dynamic Time Injection

On **every user message**, the system:
1. Gets the current date and time in your configured timezone
2. Formats it in a readable format (e.g., "Thursday, October 02, 2025 at 09:30:45 PM ICT")
3. Prepends it to the system prompt
4. Sends the updated context to the AI model

### Implementation

The time is added via the `_get_system_prompt_with_time()` method in `message_handler.py`:

```python
def _get_system_prompt_with_time(self) -> str:
    """Get the system prompt with current time and timezone information."""
    from src.config.config import NORMAL_CHAT_PROMPT, TIMEZONE
    
    # Get current time in configured timezone
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(TIMEZONE)
        current_time = datetime.now(tz)
        time_str = current_time.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
    except ImportError:
        # Fallback to pytz if zoneinfo not available
        import pytz
        tz = pytz.timezone(TIMEZONE)
        current_time = datetime.now(tz)
        time_str = current_time.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
    except Exception:
        # Final fallback to UTC
        current_time = datetime.utcnow()
        time_str = current_time.strftime("%A, %B %d, %Y at %I:%M:%S %p UTC")
    
    # Prepend current time to system prompt
    time_prefix = f"Current date and time: {time_str}\n\n"
    return time_prefix + NORMAL_CHAT_PROMPT
```

### Timezone Configuration

Set your timezone in the `.env` file:

```bash
TIMEZONE=Asia/Ho_Chi_Minh
```

**Supported Timezone Formats:**
- IANA timezone names: `Asia/Ho_Chi_Minh`, `America/New_York`, `Europe/London`, `UTC`
- Default: `UTC` (if not specified)

## What the Model Sees

### Example Context

When you send a message, the AI sees:

```
Current date and time: Thursday, October 02, 2025 at 09:30:45 PM ICT

You're ChatGPT for Discord. Be concise, helpful, safe. Reply in user's language...
[rest of system prompt]
```

### Time Format

- **Day**: Full name (e.g., Thursday)
- **Date**: Month name, day, year (e.g., October 02, 2025)
- **Time**: 12-hour format with AM/PM (e.g., 09:30:45 PM)
- **Timezone**: Abbreviation (e.g., ICT, EST, PST, UTC)

## Use Cases

### 1. Time-Aware Responses

**User**: "What time is it?"

**AI**: "It's currently 9:30 PM on Thursday, October 2nd, 2025."

### 2. Relative Time Calculations

**User**: "How many days until Christmas?"

**AI**: "It's October 2nd today, so there are 84 days until Christmas (December 25th)."

### 3. Scheduling and Reminders

**User**: "Remind me in 2 hours to take a break"

**AI**: "I'll remind you at 11:30 PM (in 2 hours from now at 9:30 PM)."

### 4. Time-Based Greetings

**User**: "Hello!"

**AI**: "Good evening! How can I help you tonight?"

### 5. Historical Context

**User**: "Tell me about today's events in history"

**AI**: "On October 2nd in history:
- 1869: Mahatma Gandhi was born
- 1950: The Peanuts comic strip first appeared
..."

### 6. Business Hours Context

**User**: "Is the support team available?"

**AI**: "It's currently 9:30 PM on Thursday. Our support team operates from 9 AM to 6 PM on weekdays, so they're currently offline. They'll be available again tomorrow morning at 9 AM."

## Benefits

### ✅ Always Accurate

- Time is fetched **dynamically** on every request
- No stale timestamps
- Always reflects the actual current time

### ✅ Timezone Aware

- Respects your configured timezone
- Shows proper timezone abbreviation (ICT, EST, PST, etc.)
- Handles daylight saving time automatically

### ✅ Works with All Models

- **Regular models** (GPT-4, GPT-5, etc.): Time added to system prompt
- **o1 models** (o1-mini, o1-preview): Time added to Instructions message
- Both approaches ensure the model always knows the current time

### ✅ Low Overhead

- Minimal token cost (~15-20 tokens)
- Negligible performance impact
- Only generated once per message

## Technical Details

### Timezone Libraries

The implementation uses multiple fallback mechanisms:

1. **Primary**: `zoneinfo` (Python 3.9+, built-in)
2. **Fallback**: `pytz` (if zoneinfo not available)
3. **Final Fallback**: UTC (if both fail)

### Docker Support

The Dockerfile includes `tzdata` package for timezone support:

```dockerfile
RUN apk add --no-cache \
    ...
    tzdata \
    ...
```

This ensures timezone information is available in Alpine Linux containers.

### Database Storage

The system prompt with time is:
- ✅ **Generated fresh** on every request
- ✅ **Not stored** in database (only base prompt stored)
- ✅ **Always up-to-date** when model receives it

The stored history contains the base system prompt without time. Time is added dynamically when messages are sent to the API.

## Configuration

### .env Settings

```bash
# Timezone configuration (IANA timezone name)
TIMEZONE=Asia/Ho_Chi_Minh

# Examples:
# TIMEZONE=America/New_York
# TIMEZONE=Europe/London
# TIMEZONE=Asia/Tokyo
# TIMEZONE=UTC
```

### Finding Your Timezone

Find your IANA timezone name:
- **Website**: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
- **Python command**:
  ```python
  import zoneinfo
  print(zoneinfo.available_timezones())
  ```

### Common Timezones

| Region | Timezone String |
|--------|----------------|
| Vietnam | `Asia/Ho_Chi_Minh` |
| US East Coast | `America/New_York` |
| US West Coast | `America/Los_Angeles` |
| UK | `Europe/London` |
| Japan | `Asia/Tokyo` |
| Australia (Sydney) | `Australia/Sydney` |
| UTC | `UTC` |

## Testing

### Verify Current Time

Ask the bot:
```
What's the current date and time?
```

Expected response should include the current time in your timezone.

### Verify Timezone

Ask the bot:
```
What timezone are you using?
```

It should respond with your configured timezone.

### Verify Time-Based Logic

Ask the bot:
```
Is it morning, afternoon, or evening right now?
```

It should correctly identify the current time of day based on the actual time.

## Troubleshooting

### Issue: Bot shows wrong time

**Solution 1**: Check `.env` configuration
```bash
grep TIMEZONE .env
# Should show: TIMEZONE=Your/Timezone
```

**Solution 2**: Verify timezone is valid
```bash
python3 -c "from zoneinfo import ZoneInfo; print(ZoneInfo('Asia/Ho_Chi_Minh'))"
```

**Solution 3**: Restart the bot to reload configuration
```bash
# Local
python3 bot.py

# Docker
docker-compose restart
```

### Issue: Timezone not found error

**Cause**: Missing `tzdata` package (Alpine Linux)

**Solution**: Rebuild Docker image
```bash
docker-compose build --no-cache
docker-compose up -d
```

### Issue: Bot shows UTC instead of configured timezone

**Cause**: Timezone configuration not loaded or invalid

**Check**:
1. Verify `.env` file exists and contains `TIMEZONE=...`
2. Check logs for timezone-related warnings
3. Ensure timezone name is in IANA format (e.g., `Asia/Ho_Chi_Minh`, not `ICT`)

## Performance Impact

### Token Cost

Adding current time to system prompt:
- **Base prompt**: ~500-600 tokens (unchanged)
- **Time prefix**: ~15-20 tokens
- **Total increase**: ~3% token overhead

### Latency

Time generation adds:
- **Typical**: <1ms per request
- **Impact**: Negligible (less than network latency)

### Memory

No additional memory usage:
- Time string generated on-the-fly
- Not stored in memory or database
- Garbage collected after request

## Future Enhancements

Potential improvements:

1. **User-Specific Timezones**: Allow each user to set their own timezone
2. **Time Format Preferences**: Let users choose 12-hour vs 24-hour format
3. **Multiple Timezone Display**: Show time in multiple timezones simultaneously
4. **Calendar Integration**: Connect to calendar APIs for event-aware responses

## Summary

✅ **Implemented**: Current time dynamically added to every conversation

✅ **Timezone Support**: Respects configured timezone from .env

✅ **All Models**: Works with both system prompt and Instructions format

✅ **Docker Ready**: Includes tzdata package for Alpine Linux

✅ **Low Overhead**: Minimal token cost and performance impact

The AI model now has full temporal awareness and can provide time-sensitive responses! 🕒
