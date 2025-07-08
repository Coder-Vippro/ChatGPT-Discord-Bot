# Supported AI Models

This document provides a comprehensive overview of all AI models supported by the ChatGPT Discord Bot Python package.

## Overview

The ChatGPT Discord Bot supports **11 different OpenAI models** with varying capabilities and token limits. All models are accessed through OpenAI's API using the `openai` Python package.

## Complete List of Supported Models

### GPT-4o Series
- **openai/gpt-4o** - Premium multimodal model with 8,000 token limit
- **openai/gpt-4o-mini** - Lightweight version with 8,000 token limit

### GPT-4.1 Series
- **openai/gpt-4.1** - Enhanced GPT-4 model with 8,000 token limit
- **openai/gpt-4.1-nano** - Ultra-lightweight version with 8,000 token limit
- **openai/gpt-4.1-mini** - Compact version with 8,000 token limit

### O1 Series (Reasoning Models)
- **openai/o1-preview** - Preview of reasoning model with 4,000 token limit
- **openai/o1-mini** - Compact reasoning model with 4,000 token limit
- **openai/o1** - Full reasoning model with 4,000 token limit

### O3 Series (Next Generation)
- **openai/o3-mini** - Compact next-gen model with 4,000 token limit
- **openai/o3** - Full next-gen model with 4,000 token limit

### O4 Series (Future Models)
- **openai/o4-mini** - Compact future model with 4,000 token limit

## Model Capabilities Matrix

| Model | Token Limit | PDF Support | System Prompts | Special Features |
|-------|-------------|-------------|----------------|------------------|
| openai/gpt-4o | 8,000 | ✅ | ✅ | Multimodal capabilities |
| openai/gpt-4o-mini | 8,000 | ✅ | ✅ | Lightweight, cost-effective |
| openai/gpt-4.1 | 8,000 | ✅ | ✅ | Enhanced performance |
| openai/gpt-4.1-nano | 8,000 | ✅ | ✅ | Ultra-lightweight |
| openai/gpt-4.1-mini | 8,000 | ✅ | ✅ | Balanced performance |
| openai/o1-preview | 4,000 | ❌ | ❌ | Advanced reasoning |
| openai/o1-mini | 4,000 | ❌ | ❌ | Compact reasoning |
| openai/o1 | 4,000 | ❌ | ❌ | Full reasoning capabilities |
| openai/o3-mini | 4,000 | ❌ | ❌ | Next-generation compact |
| openai/o3 | 4,000 | ❌ | ❌ | Next-generation full |
| openai/o4-mini | 4,000 | ❌ | ❌ | Future model preview |

## Model Selection

Users can select their preferred model using the `/choose_model` Discord command. The bot provides a dropdown menu with all available models.

## Model-Specific Limitations

### PDF Processing
Only the following models support PDF document analysis:
- openai/gpt-4o
- openai/gpt-4o-mini
- openai/gpt-4.1
- openai/gpt-4.1-nano
- openai/gpt-4.1-mini

### System Prompts
The O1, O3, and O4 series models do not support system prompts. For these models, the bot automatically converts system prompts into user messages.

### Token Management
- **High Token Models** (8,000 tokens): GPT-4o and GPT-4.1 series
- **Conservative Models** (4,000 tokens): O1, O3, and O4 series
- **Default Fallback**: 60,000 tokens for unknown models

## Python Package Dependencies

The bot uses the following key Python packages for AI model integration:

```
openai - Official OpenAI Python client
tiktoken - Token counting for OpenAI models
```

## Usage Examples

### Model Selection
```python
# Users can select models via Discord command
/choose_model
```

### API Integration
```python
# The bot uses the OpenAI client configured in the codebase
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
```

## Configuration

All model configurations are defined in `src/config/config.py`:

- `MODEL_OPTIONS`: List of all available models
- `MODEL_TOKEN_LIMITS`: Token limits for each model
- `PDF_ALLOWED_MODELS`: Models that support PDF processing
- `DEFAULT_TOKEN_LIMIT`: Fallback for unknown models

## Model Updates

The bot supports both current and future OpenAI models, with the configuration easily updatable to include new model releases from OpenAI.