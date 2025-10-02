# Discord Bot Improvements Summary

## Overview
Comprehensive improvements to the ChatGPT Discord Bot focusing on token counting, cost tracking, and handling Discord image links with 24-hour expiration.

## 1. Token Counter Utility (`src/utils/token_counter.py`)

### Features
✅ **Accurate text token counting** using tiktoken with proper encoding support
✅ **Image token calculation** based on OpenAI's vision model pricing
✅ **Discord image URL handling** with automatic download and dimension detection
✅ **24-hour expiration support** for Discord CDN links
✅ **Context limit checking** before API calls
✅ **Cost estimation** with detailed breakdown

### Encoding Support
- **o200k_base** for: gpt-4o, gpt-4.1 (all variants), gpt-5 (all variants), o1/o3/o4 families
- **cl100k_base** for: gpt-4 (original), gpt-3.5-turbo

### Image Token Calculation
- **Low detail**: 85 tokens (fixed)
- **High detail**: 170 base + (170 × number of 512×512 tiles)
- Automatically downloads Discord images to determine dimensions
- Handles base64 encoded images
- Graceful fallback for unavailable images

## 2. Database Handler Updates (`src/database/db_handler.py`)

### Enhanced Token Tracking
```python
await db_handler.save_token_usage(
    user_id=user_id,
    model="openai/gpt-4o",
    input_tokens=1000,
    output_tokens=500,
    cost=0.0125,
    text_tokens=950,      # NEW
    image_tokens=50       # NEW
)
```

### Features
✅ **Separate text/image token tracking**
✅ **Per-model statistics** with request count
✅ **Automatic image expiration filtering** (23-hour threshold)
✅ **Detailed usage breakdown** by model

### Image Expiration Handling
- Automatically filters images older than 23 hours
- Checks timestamps on every `get_history()` call
- Proactive history trimming (keeps last 50 messages)
- Replaces expired images with placeholder text

## 3. Commands Integration (`src/commands/commands.py`)

### Updated Search Command
✅ **Token counting before API call**
✅ **Context limit checking**
✅ **Cost display in responses**
✅ **Detailed logging** with text/image breakdown

### Enhanced User Stats Command
```
📊 User Statistics
Current Model: `openai/gpt-4o`

Token Usage:
• Total Input: `10,500` tokens
  ├─ Text: `9,800` tokens
  └─ Images: `700` tokens
• Total Output: `5,200` tokens
• Combined: `15,700` tokens

💰 Total Cost: `$0.156000`

Per-Model Breakdown:
`gpt-4o`
  • 25 requests, $0.125000
  • In: 8,000 (7,500 text + 500 img)
  • Out: 4,000
```

## 4. Documentation

### TOKEN_COUNTING_GUIDE.md
Comprehensive guide covering:
- Token encoding by model
- Text and image token counting
- Discord image handling
- 24-hour expiration system
- Cost estimation
- Database integration
- Complete integration examples
- Best practices
- Troubleshooting

## Key Features

### 1. Accurate Token Counting
- Uses tiktoken for precise text token counting
- Proper encoding selection per model family
- Handles multi-byte characters efficiently

### 2. Image Token Calculation
- Based on OpenAI's official pricing methodology
- Automatic dimension detection via download
- Tile-based calculation for high-detail images
- Supports Discord CDN URLs, base64, and HTTP URLs

### 3. Discord Image Expiration
- **23-hour threshold** (safer than 24 hours)
- Timestamps stored with each image
- Automatic filtering on history load
- Token counter skips expired images
- Prevents counting/sending expired links

### 4. Cost Tracking
- Real-time cost calculation
- Displayed to users after each operation
- Separate tracking for text vs image tokens
- Per-model cost breakdown
- Historical usage tracking

### 5. Context Management
- Pre-flight context limit checking
- Prevents API errors from oversized requests
- Clear error messages with token counts
- Automatic history trimming

## Model Support

### Full Token Counting Support
- ✅ gpt-4o (o200k_base)
- ✅ gpt-4o-mini (o200k_base)
- ✅ gpt-4.1 (o200k_base) ⭐ NEW
- ✅ gpt-4.1-mini (o200k_base) ⭐ NEW
- ✅ gpt-4.1-nano (o200k_base) ⭐ NEW
- ✅ gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-chat (o200k_base)
- ✅ o1, o1-mini, o1-preview (o200k_base)
- ✅ o3, o3-mini (o200k_base)
- ✅ o4, o4-mini (o200k_base)
- ✅ gpt-4 (cl100k_base)
- ✅ gpt-3.5-turbo (cl100k_base)

## Usage Examples

### Basic Text Counting
```python
from src.utils.token_counter import token_counter

tokens = token_counter.count_text_tokens("Hello world!", "openai/gpt-4o")
# Result: ~3 tokens
```

### Image Token Counting
```python
# From Discord URL
tokens = await token_counter.count_image_tokens(
    image_url="https://cdn.discordapp.com/attachments/123/456/image.png",
    detail="auto"
)
# Result: 170-1700 tokens depending on size
```

### Message Counting with Images
```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {
        "role": "user", 
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://...", "detail": "auto"},
                "timestamp": "2025-10-01T12:00:00"
            }
        ]
    }
]

counts = await token_counter.count_message_tokens(messages, "openai/gpt-4o")
# Returns: {"text_tokens": 50, "image_tokens": 500, "total_tokens": 550}
```

### Context Checking
```python
check = await token_counter.check_context_limit(messages, "openai/gpt-4o")

if not check["within_limit"]:
    print(f"Too large! {check['input_tokens']} > {check['max_tokens']}")
else:
    print(f"OK! {check['available_output_tokens']} tokens available for response")
```

## Benefits

### For Users
- 📊 **Transparent cost tracking** - see exactly what you're spending
- 💰 **Cost display** after each operation
- 📈 **Detailed statistics** with text/image breakdown
- ⚠️ **Proactive warnings** when approaching context limits
- 🖼️ **Smart image handling** with automatic expiration

### For Developers
- 🎯 **Accurate token estimation** before API calls
- 🛡️ **Error prevention** via context limit checking
- 📝 **Detailed logging** for debugging
- 🔧 **Easy integration** with existing commands
- 📚 **Comprehensive documentation**

### For Operations
- 💾 **Efficient storage** with automatic cleanup
- 🔍 **Detailed analytics** per user and per model
- 🚨 **Early warning** for context limit issues
- 📊 **Usage patterns** tracking
- 💸 **Cost monitoring** and forecasting

## Implementation Checklist

### ✅ Completed
- [x] Token counter utility with tiktoken
- [x] Image token calculation
- [x] Discord image URL handling
- [x] 24-hour expiration system
- [x] Database schema updates
- [x] Command integration (search)
- [x] Enhanced user stats
- [x] Cost tracking and display
- [x] Context limit checking
- [x] Comprehensive documentation

### 🔄 Next Steps (Optional)
- [ ] Integrate token counting in `web` command
- [ ] Add token counting to message handler
- [ ] Implement token budget system per user
- [ ] Add admin dashboard for usage analytics
- [ ] Create cost alerts for high usage
- [ ] Add token usage graphs/charts
- [ ] Implement automatic context trimming
- [ ] Add token counting to all commands

## Performance Considerations

### Memory Optimization
- ✅ Async image downloading (non-blocking)
- ✅ Automatic session management
- ✅ Connection pooling via aiohttp
- ✅ Lazy encoder loading
- ✅ Automatic history trimming

### Network Optimization
- ✅ Timeout handling for image downloads
- ✅ Fallback estimates when download fails
- ✅ Connection reuse via persistent session
- ✅ Graceful degradation

### Database Optimization
- ✅ Indexed queries on user_id and timestamp
- ✅ Atomic updates with $inc operators
- ✅ Escaped field names for MongoDB
- ✅ Batch operations where possible

## Testing Recommendations

### Unit Tests
```python
# Test text token counting
assert token_counter.count_text_tokens("Hello", "openai/gpt-4o") > 0

# Test image token estimation
tokens = await token_counter.count_image_tokens(detail="low")
assert tokens == 85

# Test expiration filtering
# ... (see TOKEN_COUNTING_GUIDE.md for examples)
```

### Integration Tests
- Send message with images
- Verify timestamps are added
- Check token counting accuracy
- Verify cost calculation
- Test expiration filtering
- Validate context limit checking

## Migration Notes

### For Existing Data
No migration needed! The system is backward compatible:
- Old records without text_tokens/image_tokens still work
- New fields are added incrementally via $inc
- Existing history is filtered automatically

### For Existing Code
Minimal changes required:
```python
# Old
await db_handler.save_token_usage(user_id, model, input, output, cost)

# New (backward compatible)
await db_handler.save_token_usage(
    user_id, model, input, output, cost,
    text_tokens=0,  # Optional
    image_tokens=0  # Optional
)
```

## Troubleshooting

### Common Issues

**Issue**: Token counts seem inaccurate
- **Solution**: Verify model name matches encoding map
- **Check**: Model uses correct encoding (o200k_base vs cl100k_base)

**Issue**: Images not being counted
- **Solution**: Check image URL is accessible
- **Check**: Verify timestamp format is ISO 8601
- **Check**: Ensure image hasn't expired (>23 hours)

**Issue**: Context limit errors
- **Solution**: Enable automatic history trimming
- **Check**: Verify context limits in token_counter.py
- **Try**: Reduce image detail to "low"

**Issue**: Cost seems wrong
- **Solution**: Verify MODEL_PRICING has correct values
- **Check**: Ensure per 1M token calculation
- **Check**: Use actual usage from API response

## Conclusion

This comprehensive token counting system provides:
- ✅ **Accuracy** via tiktoken and proper encoding
- ✅ **Transparency** with detailed cost tracking
- ✅ **Reliability** through context limit checking
- ✅ **Efficiency** with automatic image expiration
- ✅ **Scalability** via optimized database operations

The system is production-ready and fully documented for easy maintenance and extension.
