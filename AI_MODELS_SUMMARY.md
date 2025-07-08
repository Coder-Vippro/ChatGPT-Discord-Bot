# AI Models Summary for ChatGPT Discord Bot

## Quick Answer

The ChatGPT Discord Bot Python package supports **11 OpenAI models** through the official `openai` Python library:

## Complete Model List

1. **openai/gpt-4o** - Premium multimodal model (8,000 tokens)
2. **openai/gpt-4o-mini** - Lightweight version (8,000 tokens)
3. **openai/gpt-4.1** - Enhanced GPT-4 (8,000 tokens)
4. **openai/gpt-4.1-nano** - Ultra-lightweight (8,000 tokens)
5. **openai/gpt-4.1-mini** - Compact version (8,000 tokens)
6. **openai/o1-preview** - Reasoning model preview (4,000 tokens)
7. **openai/o1-mini** - Compact reasoning (4,000 tokens)
8. **openai/o1** - Full reasoning model (4,000 tokens)
9. **openai/o3-mini** - Next-gen compact (4,000 tokens)
10. **openai/o3** - Next-gen full model (4,000 tokens)
11. **openai/o4-mini** - Future model preview (4,000 tokens)

## Key Python Dependencies

- **openai** - Official OpenAI Python client library
- **tiktoken** - Token counting for OpenAI models

## Model Features

- **All models**: Text generation, conversation, code assistance
- **PDF Support**: Only GPT-4o and GPT-4.1 series models
- **System Prompts**: Supported by GPT-4o and GPT-4.1 series (not O1/O3/O4)
- **Tools/Functions**: Available on most models except o1-mini and o1-preview

## Default Model

The bot defaults to **openai/gpt-4.1-mini** when no model is selected by the user.