# Quick Reference: Token Counting System

## Import
```python
from src.utils.token_counter import token_counter
```

## Text Tokens
```python
tokens = token_counter.count_text_tokens("Hello!", "openai/gpt-4o")
```

## Image Tokens
```python
# From URL (Discord CDN)
tokens = await token_counter.count_image_tokens(
    image_url="https://cdn.discordapp.com/...",
    detail="auto"  # or "low" or "high"
)

# From bytes
tokens = await token_counter.count_image_tokens(
    image_data=image_bytes,
    detail="auto"
)
```

## Message Tokens
```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Look at this"},
            {
                "type": "image_url",
                "image_url": {"url": "https://...", "detail": "auto"},
                "timestamp": "2025-10-01T12:00:00"  # Add for 24h expiration
            }
        ]
    }
]

counts = await token_counter.count_message_tokens(messages, "openai/gpt-4o")
# Returns: {
#     "text_tokens": 50,
#     "image_tokens": 500,
#     "total_tokens": 550
# }
```

## Context Check
```python
check = await token_counter.check_context_limit(messages, "openai/gpt-4o")

if not check["within_limit"]:
    print(f"⚠️ Too large: {check['input_tokens']} tokens")
    print(f"Max: {check['max_tokens']} tokens")
else:
    print(f"✅ OK! {check['available_output_tokens']} tokens available")
```

## Cost Estimate
```python
cost = token_counter.estimate_cost(
    input_tokens=1000,
    output_tokens=500,
    model="openai/gpt-4o"
)
print(f"Cost: ${cost:.6f}")
```

## Save Usage (Database)
```python
await db_handler.save_token_usage(
    user_id=123456789,
    model="openai/gpt-4o",
    input_tokens=1000,
    output_tokens=500,
    cost=0.0125,
    text_tokens=950,
    image_tokens=50
)
```

## Get User Stats
```python
# Total usage
stats = await db_handler.get_user_token_usage(user_id)
print(f"Total: {stats['total_cost']:.6f}")
print(f"Text: {stats['total_text_tokens']:,}")
print(f"Images: {stats['total_image_tokens']:,}")

# By model
model_usage = await db_handler.get_user_token_usage_by_model(user_id)
for model, usage in model_usage.items():
    print(f"{model}: ${usage['cost']:.6f}, {usage['requests']} reqs")
```

## Model Encodings

### o200k_base (200k vocabulary)
- gpt-4o, gpt-4o-mini
- **gpt-4.1, gpt-4.1-mini, gpt-4.1-nano** ⭐
- gpt-5 (all variants)
- o1, o3, o4 (all variants)

### cl100k_base (100k vocabulary)
- gpt-4 (original)
- gpt-3.5-turbo

## Image Token Costs

| Detail | Cost |
|--------|------|
| Low | 85 tokens |
| High | 170 + (170 × tiles) |

Tiles = ceil(width/512) × ceil(height/512) after scaling to 2048×2048 and 768px shortest side.

## Context Limits

| Model | Tokens |
|-------|--------|
| gpt-4o, gpt-4o-mini, gpt-4.1* | 128,000 |
| gpt-5*, o1-mini, o1-preview | 128,000-200,000 |
| o1, o3, o4 | 200,000 |
| gpt-4 | 8,192 |
| gpt-3.5-turbo | 16,385 |

## Discord Image Timestamps

Always add when storing images:
```python
{
    "type": "image_url",
    "image_url": {"url": discord_url, "detail": "auto"},
    "timestamp": datetime.now().isoformat()  # ← Important!
}
```

Images >23 hours old are automatically filtered.

## Complete Integration Pattern

```python
async def handle_message(interaction, text, image_urls=None):
    user_id = interaction.user.id
    model = await db_handler.get_user_model(user_id) or "openai/gpt-4o"
    history = await db_handler.get_history(user_id)
    
    # Build content
    content = [{"type": "text", "text": text}]
    if image_urls:
        for url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": "auto"},
                "timestamp": datetime.now().isoformat()
            })
    
    messages = history + [{"role": "user", "content": content}]
    
    # Check context
    check = await token_counter.check_context_limit(messages, model)
    if not check["within_limit"]:
        await interaction.followup.send(
            f"⚠️ Too large: {check['input_tokens']:,} tokens",
            ephemeral=True
        )
        return
    
    # Count tokens
    input_count = await token_counter.count_message_tokens(messages, model)
    
    # Call API
    response = await openai_client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    reply = response.choices[0].message.content
    
    # Get usage
    usage = response.usage
    actual_in = usage.prompt_tokens if usage else input_count['total_tokens']
    actual_out = usage.completion_tokens if usage else token_counter.count_text_tokens(reply, model)
    
    # Calculate cost
    cost = token_counter.estimate_cost(actual_in, actual_out, model)
    
    # Save
    await db_handler.save_token_usage(
        user_id=user_id,
        model=model,
        input_tokens=actual_in,
        output_tokens=actual_out,
        cost=cost,
        text_tokens=input_count['text_tokens'],
        image_tokens=input_count['image_tokens']
    )
    
    # Respond
    await interaction.followup.send(f"{reply}\n\n💰 ${cost:.6f}")
```

## Cleanup

At bot shutdown:
```python
await token_counter.close()
```

## Key Points

✅ **Always add timestamps** to Discord images
✅ **Check context limits** before API calls
✅ **Use actual usage** from API response when available
✅ **Track text/image separately** for analytics
✅ **Show cost** to users
✅ **Filter expired images** automatically (done by db_handler)

## Troubleshooting

**Tokens seem wrong?**
→ Check model name and encoding

**Images not counted?**
→ Verify URL is accessible and timestamp is valid

**Context errors?**
→ Trim history or use "low" detail for images

**Cost incorrect?**
→ Check MODEL_PRICING and use actual API usage
