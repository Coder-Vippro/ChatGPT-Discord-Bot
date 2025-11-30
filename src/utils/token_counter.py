"""
Token counter utility for OpenAI API requests including text and images.
Handles Discord image links stored in MongoDB with 24-hour expiration.
"""

import tiktoken
import logging
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta

class TokenCounter:
    """
    Token counter for OpenAI API requests including text and images.
    Based on OpenAI's token counting methodology with support for Discord image links.
    """
    
    # Image token costs based on OpenAI's vision pricing
    IMAGE_TOKEN_COSTS = {
        "low": 85,      # Low detail image
        "high": 170,    # Base cost for high detail
        "tile": 170     # Cost per 512x512 tile for high detail
    }
    
    def __init__(self):
        self.encoders = {}
        self._load_encoders()
        self.session: Optional[aiohttp.ClientSession] = None
        logging.info("TokenCounter initialized")
    
    def _load_encoders(self):
        """Pre-load tiktoken encoders for different models"""
        try:
            self.encoders = {
                # o200k_base encoding (200k vocabulary) - newer models
                "gpt-4o": tiktoken.get_encoding("o200k_base"),
                "gpt-4o-mini": tiktoken.get_encoding("o200k_base"),
                "gpt-4.1": tiktoken.get_encoding("o200k_base"),  # GPT-4.1 uses o200k_base
                "gpt-4.1-mini": tiktoken.get_encoding("o200k_base"),
                "gpt-4.1-nano": tiktoken.get_encoding("o200k_base"),
                "gpt-5": tiktoken.get_encoding("o200k_base"),
                "gpt-5-mini": tiktoken.get_encoding("o200k_base"),
                "gpt-5-nano": tiktoken.get_encoding("o200k_base"),
                "gpt-5-chat": tiktoken.get_encoding("o200k_base"),
                "o1": tiktoken.get_encoding("o200k_base"),
                "o1-mini": tiktoken.get_encoding("o200k_base"),
                "o1-preview": tiktoken.get_encoding("o200k_base"),
                "o3": tiktoken.get_encoding("o200k_base"),
                "o3-mini": tiktoken.get_encoding("o200k_base"),
                "o4": tiktoken.get_encoding("o200k_base"),
                "o4-mini": tiktoken.get_encoding("o200k_base"),
                
                # cl100k_base encoding (100k vocabulary) - older models
                "gpt-4": tiktoken.get_encoding("cl100k_base"),
                "gpt-3.5-turbo": tiktoken.get_encoding("cl100k_base"),
            }
            logging.info("Tiktoken encoders loaded successfully")
        except Exception as e:
            logging.error(f"Error loading tiktoken encoders: {e}")
    
    def _get_encoder(self, model: str):
        """Get appropriate encoder for model"""
        model_key = model.replace("openai/", "")
        
        # o200k_base models (newer)
        o200k_prefixes = ["gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4"]
        for prefix in o200k_prefixes:
            if model_key.startswith(prefix):
                return self.encoders.get(model_key.split('-')[0] if '-' in model_key else model_key, 
                                       self.encoders.get("gpt-4o"))
        
        # cl100k_base models (older)
        if model_key.startswith("gpt-4") and not any(model_key.startswith(x) for x in ["gpt-4o", "gpt-4.1"]):
            return self.encoders.get("gpt-4")
        if model_key.startswith("gpt-3.5"):
            return self.encoders.get("gpt-3.5-turbo")
        
        # Default to newer encoding
        return self.encoders.get("gpt-4o")
    
    def count_text_tokens(self, text: str, model: str) -> int:
        """Count tokens in text using tiktoken"""
        try:
            encoder = self._get_encoder(model)
            if encoder:
                return len(encoder.encode(text))
            else:
                # Fallback: rough estimate (1 token ≈ 4 characters)
                return len(text) // 4
        except Exception as e:
            logging.error(f"Error counting tokens: {e}")
            return len(text) // 4
    
    async def _get_image_from_url(self, url: str) -> Optional[bytes]:
        """Download image from URL (Discord CDN link)"""
        try:
            if not self.session:
                timeout = aiohttp.ClientTimeout(total=10, connect=5)
                self.session = aiohttp.ClientSession(timeout=timeout)
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logging.warning(f"Failed to download image: HTTP {response.status}")
                    return None
        except Exception as e:
            logging.error(f"Error downloading image from {url}: {e}")
            return None
    
    async def count_image_tokens(
        self, 
        image_data: Optional[bytes] = None,
        image_url: Optional[str] = None,
        detail: str = "auto"
    ) -> int:
        """
        Count tokens for an image based on OpenAI's vision model pricing.
        
        Args:
            image_data: Raw image bytes
            image_url: URL to image (Discord CDN link)
            detail: "low", "high", or "auto"
        
        Returns:
            Number of tokens the image will consume
        """
        try:
            # If detail is low, return fixed cost
            if detail == "low":
                return self.IMAGE_TOKEN_COSTS["low"]
            
            # Get image dimensions
            if image_data:
                img = Image.open(BytesIO(image_data))
                width, height = img.size
            elif image_url:
                # Try to download and get dimensions
                image_data = await self._get_image_from_url(image_url)
                if image_data:
                    try:
                        img = Image.open(BytesIO(image_data))
                        width, height = img.size
                    except Exception as e:
                        logging.error(f"Error opening image: {e}")
                        # Conservative high estimate if we can't determine size
                        return self.IMAGE_TOKEN_COSTS["high"] + (self.IMAGE_TOKEN_COSTS["tile"] * 4)
                else:
                    # If download fails, use conservative estimate
                    return self.IMAGE_TOKEN_COSTS["high"] + (self.IMAGE_TOKEN_COSTS["tile"] * 4)
            else:
                return self.IMAGE_TOKEN_COSTS["high"]
            
            # For high detail images, calculate tile-based cost
            # Scale image to fit within 2048x2048
            max_dim = 2048
            if width > max_dim or height > max_dim:
                scale = min(max_dim / width, max_dim / height)
                width = int(width * scale)
                height = int(height * scale)
            
            # Scale shortest side to 768
            if width < height:
                scale = 768 / width
                width = 768
                height = int(height * scale)
            else:
                scale = 768 / height
                height = 768
                width = int(width * scale)
            
            # Calculate number of 512x512 tiles needed
            tiles_width = (width + 511) // 512
            tiles_height = (height + 511) // 512
            num_tiles = tiles_width * tiles_height
            
            # Base cost + (tile cost * number of tiles)
            total_tokens = self.IMAGE_TOKEN_COSTS["high"] + (self.IMAGE_TOKEN_COSTS["tile"] * num_tiles)
            
            return total_tokens
            
        except Exception as e:
            logging.error(f"Error counting image tokens: {e}")
            # Return conservative estimate
            return self.IMAGE_TOKEN_COSTS["high"] + (self.IMAGE_TOKEN_COSTS["tile"] * 4)
    
    async def count_message_tokens(
        self, 
        messages: List[Dict[str, Any]], 
        model: str
    ) -> Dict[str, int]:
        """
        Count total tokens in a message list including text and images.
        Handles Discord image links stored in MongoDB with timestamps.
        
        Returns:
            Dict with 'text_tokens', 'image_tokens', 'total_tokens'
        """
        text_tokens = 0
        image_tokens = 0
        
        # Tokens for message formatting (varies by model)
        tokens_per_message = 3  # <|start|>role/name\n{content}<|end|>\n
        tokens_per_name = 1
        
        # Current time for checking image expiration
        current_time = datetime.now()
        expiration_time = current_time - timedelta(hours=23)
        
        for message in messages:
            text_tokens += tokens_per_message
            
            # Count role tokens
            if "role" in message:
                text_tokens += self.count_text_tokens(message["role"], model)
            
            if "name" in message:
                text_tokens += tokens_per_name
                text_tokens += self.count_text_tokens(message["name"], model)
            
            # Handle content
            content = message.get("content", "")
            
            # Content can be string or array of content parts
            if isinstance(content, str):
                text_tokens += self.count_text_tokens(content, model)
            
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type", "")
                        
                        if part_type == "text":
                            text_tokens += self.count_text_tokens(part.get("text", ""), model)
                        
                        elif part_type == "image_url":
                            image_info = part.get("image_url", {})
                            detail = image_info.get("detail", "auto")
                            url = image_info.get("url", "")
                            
                            # Check timestamp if present (for Discord images)
                            timestamp_str = part.get("timestamp")
                            if timestamp_str:
                                try:
                                    timestamp = datetime.fromisoformat(timestamp_str)
                                    # Skip expired images
                                    if timestamp <= expiration_time:
                                        logging.info(f"Skipping expired image (added at {timestamp_str})")
                                        continue
                                except Exception as e:
                                    logging.warning(f"Error parsing timestamp {timestamp_str}: {e}")
                            
                            # Check if it's base64 data
                            if url.startswith("data:image"):
                                try:
                                    # Extract base64 data
                                    base64_data = url.split(",")[1]
                                    image_data = base64.b64decode(base64_data)
                                    tokens = await self.count_image_tokens(
                                        image_data=image_data,
                                        detail=detail
                                    )
                                    image_tokens += tokens
                                except Exception as e:
                                    logging.error(f"Error processing base64 image: {e}")
                                    image_tokens += self.IMAGE_TOKEN_COSTS["high"]
                            elif url.startswith("http"):
                                # Discord CDN URL or other HTTP URL
                                tokens = await self.count_image_tokens(
                                    image_url=url,
                                    detail=detail
                                )
                                image_tokens += tokens
                            else:
                                # Unknown format, use default
                                image_tokens += self.IMAGE_TOKEN_COSTS["high"]
        
        # Add tokens for reply formatting
        text_tokens += 3  # For assistant reply priming
        
        return {
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
            "total_tokens": text_tokens + image_tokens
        }
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ) -> float:
        """
        Estimate cost based on token usage.
        
        Args:
            input_tokens: Number of input tokens (including images)
            output_tokens: Number of output tokens
            model: Model name
        
        Returns:
            Estimated cost in USD
        """
        # Import from centralized pricing module
        from src.config.pricing import MODEL_PRICING
        
        if model not in MODEL_PRICING:
            model = "openai/gpt-4o"  # Default fallback
        
        pricing = MODEL_PRICING[model]
        
        # Pricing is per 1M tokens
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    async def check_context_limit(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_output_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Check if messages will exceed context window.
        
        Returns:
            Dict with 'within_limit' (bool), 'total_tokens' (int), 
            'max_tokens' (int), 'available_output_tokens' (int)
        """
        # Model context limits
        CONTEXT_LIMITS = {
            "openai/gpt-4o": 128000,
            "openai/gpt-4o-mini": 128000,
            "openai/gpt-4.1": 128000,
            "openai/gpt-4.1-mini": 128000,
            "openai/gpt-4.1-nano": 128000,
            "openai/gpt-5": 200000,
            "openai/gpt-5-mini": 200000,
            "openai/gpt-5-nano": 200000,
            "openai/gpt-5-chat": 200000,
            "openai/o1-preview": 128000,
            "openai/o1-mini": 128000,
            "openai/o1": 200000,
            "openai/o3-mini": 200000,
            "openai/o3": 200000,
            "openai/o4-mini": 200000,
            "openai/gpt-4": 8192,
            "openai/gpt-3.5-turbo": 16385,
        }
        
        max_tokens = CONTEXT_LIMITS.get(model, 128000)
        token_counts = await self.count_message_tokens(messages, model)
        total_input_tokens = token_counts["total_tokens"]
        
        # Reserve space for output
        available_for_output = max_tokens - total_input_tokens
        within_limit = available_for_output >= max_output_tokens
        
        return {
            "within_limit": within_limit,
            "input_tokens": total_input_tokens,
            "text_tokens": token_counts["text_tokens"],
            "image_tokens": token_counts["image_tokens"],
            "max_tokens": max_tokens,
            "available_output_tokens": available_for_output,
            "needed_output_tokens": max_output_tokens
        }
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
            logging.info("TokenCounter session closed")

# Global instance
token_counter = TokenCounter()
