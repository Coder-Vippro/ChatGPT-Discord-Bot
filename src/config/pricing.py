"""
Centralized pricing configuration for OpenAI models.

This module provides a single source of truth for model pricing,
eliminating duplication across the codebase.
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing information for a model (per 1M tokens in USD)."""
    input: float
    output: float
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost for given token counts."""
        input_cost = (input_tokens / 1_000_000) * self.input
        output_cost = (output_tokens / 1_000_000) * self.output
        return input_cost + output_cost


# Model pricing per 1M tokens (in USD)
# Centralized location - update prices here only
MODEL_PRICING: Dict[str, ModelPricing] = {
    # GPT-4o Family
    "openai/gpt-4o": ModelPricing(input=5.00, output=20.00),
    "openai/gpt-4o-mini": ModelPricing(input=0.60, output=2.40),
    
    # GPT-4.1 Family
    "openai/gpt-4.1": ModelPricing(input=2.00, output=8.00),
    "openai/gpt-4.1-mini": ModelPricing(input=0.40, output=1.60),
    "openai/gpt-4.1-nano": ModelPricing(input=0.10, output=0.40),
    
    # GPT-5 Family
    "openai/gpt-5": ModelPricing(input=1.25, output=10.00),
    "openai/gpt-5-mini": ModelPricing(input=0.25, output=2.00),
    "openai/gpt-5-nano": ModelPricing(input=0.05, output=0.40),
    "openai/gpt-5-chat": ModelPricing(input=1.25, output=10.00),
    
    # o1 Family (Reasoning models)
    "openai/o1-preview": ModelPricing(input=15.00, output=60.00),
    "openai/o1-mini": ModelPricing(input=1.10, output=4.40),
    "openai/o1": ModelPricing(input=15.00, output=60.00),
    
    # o3 Family
    "openai/o3-mini": ModelPricing(input=1.10, output=4.40),
    "openai/o3": ModelPricing(input=2.00, output=8.00),
    
    # o4 Family
    "openai/o4-mini": ModelPricing(input=2.00, output=8.00),
    
    # Claude Family (Anthropic)
    "claude/claude-3-5-sonnet": ModelPricing(input=3.00, output=15.00),
    "claude/claude-3-5-haiku": ModelPricing(input=0.80, output=4.00),
    "claude/claude-3-opus": ModelPricing(input=15.00, output=75.00),
}


def get_model_pricing(model: str) -> Optional[ModelPricing]:
    """
    Get pricing for a specific model.
    
    Args:
        model: The model name (e.g., "openai/gpt-4o")
        
    Returns:
        ModelPricing object or None if model not found
    """
    return MODEL_PRICING.get(model)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost for a given model and token counts.
    
    Args:
        model: The model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Total cost in USD, or 0.0 if model not found
    """
    pricing = get_model_pricing(model)
    if pricing:
        return pricing.calculate_cost(input_tokens, output_tokens)
    return 0.0


def get_all_models() -> list:
    """Get list of all available models with pricing."""
    return list(MODEL_PRICING.keys())


def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1.00:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"
