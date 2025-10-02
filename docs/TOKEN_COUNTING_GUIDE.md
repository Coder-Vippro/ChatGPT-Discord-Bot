# Token Counting Guide

## Overview

This bot implements comprehensive token counting for both text and images, with special handling for Discord image links stored in MongoDB with 24-hour expiration.

## Token Encoding by Model

### o200k_base (200k vocabulary) - Newer Models
Used for:
- ✅ **gpt-4o** and **gpt-4o-mini**
- ✅ **gpt-4.1**, **gpt-4.1-mini**, **gpt-4.1-nano** (NEW!)
- ✅ **gpt-5**, **gpt-5-mini**, **gpt-5-nano**, **gpt-5-chat**
- ✅ **o1**, **o1-mini**, **o1-preview**
- ✅ **o3**, **o3-mini**
- ✅ **o4**, **o4-mini**

### cl100k_base (100k vocabulary) - Older Models
Used for:
- ✅ **gpt-4** (original, not 4o or 4.1)
- ✅ **gpt-3.5-turbo**

## Token Counting Features

### 1. Text Token Counting
```python
from src.utils.token_counter import token_counter

# Count text tokens
tokens = token_counter.count_text_tokens("Hello, world!", "openai/gpt-4o")
print(f"Text uses {tokens} tokens")
```

### 2. Image Token Counting

Images consume tokens based on their dimensions and detail level:

#### Low Detail
- **85 tokens** (fixed cost)

#### High Detail
- **Base cost**: 170 tokens
- **Tile cost**: 170 tokens per 512x512 tile
- Images are scaled to fit 2048x2048
- Shortest side scaled to 768px
- Divided into 512x512 tiles

```python
# Count image tokens from Discord URL
tokens = await token_counter.count_image_tokens(
    image_url="https://cdn.discordapp.com/attachments/...",
    detail="auto"
)
print(f"Image uses {tokens} tokens")

# Count image tokens from bytes
with open("image.png", "rb") as f:
    image_data = f.read()
tokens = await token_counter.count_image_tokens(
    image_data=image_data,
    detail="high"
)
```

### 3. Message Token Counting

Count tokens for complete message arrays including text and images:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

token_counts = await token_counter.count_message_tokens(messages, "openai/gpt-4o")
print(f"Total: {token_counts['total_tokens']} tokens")
print(f"Text: {token_counts['text_tokens']} tokens")
print(f"Images: {token_counts['image_tokens']} tokens")
```

### 4. Context Limit Checking

Check if messages fit within model's context window:

```python
context_check = await token_counter.check_context_limit(
    messages=messages,
    model="openai/gpt-4o",
    max_output_tokens=4096
)

if not context_check["within_limit"]:
    print(f"⚠️ Messages too large: {context_check['input_tokens']} tokens")
    print(f"Maximum: {context_check['max_tokens']} tokens")
else:
    print(f"✅ Within limit. Available for output: {context_check['available_output_tokens']} tokens")
```

## Discord Image Handling

### Image Storage in MongoDB

When users send images in Discord:

1. **Image URL Captured**: Discord CDN URL is stored
2. **Timestamp Added**: Current datetime is recorded
3. **Saved to History**: Stored in message content array

```python
content = [
    {"type": "text", "text": "Look at this image"},
    {
        "type": "image_url",
        "image_url": {
            "url": "https://cdn.discordapp.com/attachments/...",
            "detail": "auto"
        },
        "timestamp": "2025-10-01T12:00:00"  # Added automatically
    }
]
```

### 24-Hour Expiration

Discord CDN links expire after ~24 hours. The system:

1. **Filters Expired Images**: When loading history, images older than 23 hours are removed
2. **Token Counting Skips Expired**: Token counter checks timestamps and skips expired images
3. **Automatic Cleanup**: Database handler filters expired images on every `get_history()` call

```python
# In db_handler.py
def _filter_expired_images(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out image links that are older than 23 hours"""
    current_time = datetime.now()
    expiration_time = current_time - timedelta(hours=23)
    
    # Checks timestamp and removes expired images
    # ...
```

### Token Counter Expiration Handling

The token counter automatically skips expired images:

```python
# In token_counter.py count_message_tokens()
timestamp_str = part.get("timestamp")
if timestamp_str:
    timestamp = datetime.fromisoformat(timestamp_str)
    if timestamp <= expiration_time:
        logging.info(f"Skipping expired image (added at {timestamp_str})")
        continue  # Don't count tokens for expired images
```

## Cost Estimation

Calculate costs based on token usage:

```python
cost = token_counter.estimate_cost(
    input_tokens=1000,
    output_tokens=500,
    model="openai/gpt-4o"
)
print(f"Estimated cost: ${cost:.6f}")
```

### Model Pricing (per 1M tokens)

| Model | Input | Output |
|-------|-------|--------|
| gpt-4o | $5.00 | $20.00 |
| gpt-4o-mini | $0.60 | $2.40 |
| gpt-4.1 | $2.00 | $8.00 |
| gpt-4.1-mini | $0.40 | $1.60 |
| gpt-4.1-nano | $0.10 | $0.40 |
| gpt-5 | $1.25 | $10.00 |
| gpt-5-mini | $0.25 | $2.00 |
| gpt-5-nano | $0.05 | $0.40 |
| o1-preview | $15.00 | $60.00 |
| o1-mini | $1.10 | $4.40 |

## Database Token Tracking

### Save Token Usage

```python
await db_handler.save_token_usage(
    user_id=user_id,
    model="openai/gpt-4o",
    input_tokens=1000,
    output_tokens=500,
    cost=0.0125,
    text_tokens=950,
    image_tokens=50
)
```

### Get User Statistics

```python
# Get total usage
stats = await db_handler.get_user_token_usage(user_id)
print(f"Total input: {stats['total_input_tokens']}")
print(f"Total text: {stats['total_text_tokens']}")
print(f"Total images: {stats['total_image_tokens']}")
print(f"Total cost: ${stats['total_cost']:.6f}")

# Get usage by model
model_usage = await db_handler.get_user_token_usage_by_model(user_id)
for model, usage in model_usage.items():
    print(f"{model}: {usage['requests']} requests, ${usage['cost']:.6f}")
    print(f"  Text: {usage['text_tokens']}, Images: {usage['image_tokens']}")
```

## Integration Example

Complete example of using token counting in a command:

```python
from src.utils.token_counter import token_counter

async def process_user_message(interaction, user_message, image_urls=None):
    user_id = interaction.user.id
    model = await db_handler.get_user_model(user_id) or DEFAULT_MODEL
    history = await db_handler.get_history(user_id)
    
    # Build message content
    content = [{"type": "text", "text": user_message}]
    
    # Add images with timestamps
    if image_urls:
        for url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": "auto"},
                "timestamp": datetime.now().isoformat()
            })
    
    # Add to messages
    messages = history + [{"role": "user", "content": content}]
    
    # Check context limit
    context_check = await token_counter.check_context_limit(messages, model)
    if not context_check["within_limit"]:
        await interaction.followup.send(
            f"⚠️ Context too large: {context_check['input_tokens']:,} tokens. "
            f"Maximum: {context_check['max_tokens']:,} tokens.",
            ephemeral=True
        )
        return
    
    # Count input tokens
    input_count = await token_counter.count_message_tokens(messages, model)
    
    # Call API
    response = await openai_client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    reply = response.choices[0].message.content
    
    # Get actual usage from API
    usage = response.usage
    actual_input = usage.prompt_tokens if usage else input_count['total_tokens']
    actual_output = usage.completion_tokens if usage else token_counter.count_text_tokens(reply, model)
    
    # Calculate cost
    cost = token_counter.estimate_cost(actual_input, actual_output, model)
    
    # Save to database
    await db_handler.save_token_usage(
        user_id=user_id,
        model=model,
        input_tokens=actual_input,
        output_tokens=actual_output,
        cost=cost,
        text_tokens=input_count['text_tokens'],
        image_tokens=input_count['image_tokens']
    )
    
    # Send response with cost
    await interaction.followup.send(f"{reply}\n\n💰 Cost: ${cost:.6f}")
```

## Best Practices

### 1. Always Check Context Limits
Before making API calls, check if the messages fit within the model's context window.

### 2. Add Timestamps to Images
When storing images from Discord, always add a timestamp:
```python
"timestamp": datetime.now().isoformat()
```

### 3. Filter History on Load
The database handler automatically filters expired images when loading history.

### 4. Count Before API Call
Count tokens before calling the API to provide accurate estimates and warnings.

### 5. Use Actual Usage from API
Prefer `response.usage` over estimates when available:
```python
actual_input = usage.prompt_tokens if usage else estimated_tokens
```

### 6. Track Text and Image Separately
Store both text_tokens and image_tokens for detailed analytics.

### 7. Show Cost to Users
Always display the cost after operations so users are aware of usage.

## Context Window Limits

| Model | Context Limit |
|-------|--------------|
| gpt-4o | 128,000 tokens |
| gpt-4o-mini | 128,000 tokens |
| gpt-4.1 | 128,000 tokens |
| gpt-4.1-mini | 128,000 tokens |
| gpt-4.1-nano | 128,000 tokens |
| gpt-5 | 200,000 tokens |
| gpt-5-mini | 200,000 tokens |
| gpt-5-nano | 200,000 tokens |
| o1 | 200,000 tokens |
| o1-mini | 128,000 tokens |
| o3 | 200,000 tokens |
| o3-mini | 200,000 tokens |
| gpt-4 | 8,192 tokens |
| gpt-3.5-turbo | 16,385 tokens |

## Troubleshooting

### Image Token Count Seems Wrong
- Check if image was downloaded successfully
- Verify image dimensions
- Remember: high detail images use tile-based calculation

### Expired Images Still Counted
- Check that timestamps are in ISO format
- Verify expiration threshold (23 hours)
- Ensure `_filter_expired_images()` is called

### Cost Calculation Incorrect
- Verify model name matches MODEL_PRICING keys exactly
- Check that pricing is per 1M tokens
- Ensure input/output tokens are correct

### Context Limit Exceeded
- Trim conversation history (keep last N messages)
- Reduce image detail level to "low"
- Remove old images from history
- Use a model with larger context window

## Cleanup

Don't forget to close the token counter session when shutting down:

```python
await token_counter.close()
```

This is typically done in the bot's cleanup/shutdown handler.
