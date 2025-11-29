"""
Image Generation Utilities - Runware API Integration
======================================================
Comprehensive image generation, editing, and manipulation tools using the Runware SDK.
Configuration is loaded from config/image_config.json for easy model management.
"""

import io
import aiohttp
import logging
import tempfile
import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from runware import (
    Runware, 
    IImageInference, 
    IPromptEnhance, 
    IImageBackgroundRemoval, 
    IImageCaption, 
    IImageUpscale, 
    IPhotoMaker
)


def load_image_config() -> Dict[str, Any]:
    """Load image configuration from JSON file"""
    config_paths = [
        Path(__file__).parent.parent.parent / "config" / "image_config.json",
        Path(__file__).parent.parent / "config" / "image_config.json",
        Path("config/image_config.json"),
        Path("image_config.json")
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    logging.info(f"Loaded image config from {config_path}")
                    return config
            except Exception as e:
                logging.error(f"Error loading image config from {config_path}: {e}")
    
    logging.warning("Image config file not found, using defaults")
    return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Return default configuration if config file is not found"""
    return {
        "settings": {
            "default_model": "flux",
            "default_upscale_model": "clarity",
            "default_background_removal_model": "bria",
            "connection_timeout": 120,
            "max_retries": 3,
            "retry_delay": 2,
            "output_format": "WEBP",
            "output_quality": 95
        },
        "image_models": {
            "flux": {
                "model_id": "runware:101@1",
                "name": "FLUX.1",
                "description": "High-quality FLUX model",
                "default_width": 1024,
                "default_height": 1024,
                "max_width": 2048,
                "max_height": 2048,
                "default_steps": 30,
                "default_cfg_scale": 7.5,
                "supports_negative_prompt": True,
                "max_images": 4
            }
        },
        "upscale_models": {
            "clarity": {
                "model_id": "runware:500@1",
                "name": "Clarity",
                "supported_factors": [2, 4]
            }
        },
        "background_removal_models": {
            "bria": {
                "model_id": "runware:110@1",
                "name": "Bria RMBG 2.0"
            }
        },
        "default_negative_prompts": {
            "general": "blurry, distorted, low quality, watermark, signature, text, bad anatomy, deformed"
        }
    }


# Global config - loaded once at module import
IMAGE_CONFIG = load_image_config()


class ImageGenerator:
    """
    Image generation and manipulation using Runware API.
    
    Features:
    - Text-to-image generation with multiple models
    - Image upscaling with various algorithms
    - Background removal
    - Image captioning (image-to-text)
    - Prompt enhancement
    - PhotoMaker for reference-based generation
    
    Configuration is loaded from config/image_config.json
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the image generator with the Runware API key.
        
        Args:
            api_key: API key for Runware (optional - can use RUNWARE_API_KEY env var)
        """
        self.config = IMAGE_CONFIG
        self.settings = self.config.get("settings", {})
        
        # Initialize Runware client
        if api_key and api_key not in ("fake_key", "test_key", ""):
            self.runware = Runware(api_key=api_key)
        else:
            self.runware = Runware()
        
        self.connected = False
        self._connection_retries = 0
        self._max_retries = self.settings.get("max_retries", 3)
        
        logging.info(f"ImageGenerator initialized with {len(self.get_available_models())} models")
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Get all available image generation models"""
        return self.config.get("image_models", {})
    
    def get_model_info(self, model_key: str) -> Optional[Dict]:
        """Get information about a specific model"""
        models = self.get_available_models()
        return models.get(model_key)
    
    def get_upscale_models(self) -> Dict[str, Dict]:
        """Get all available upscale models"""
        return self.config.get("upscale_models", {})
    
    def get_background_removal_models(self) -> Dict[str, Dict]:
        """Get all available background removal models"""
        return self.config.get("background_removal_models", {})
    
    def get_default_negative_prompt(self, category: str = "general") -> str:
        """Get default negative prompt for a category"""
        prompts = self.config.get("default_negative_prompts", {})
        return prompts.get(category, prompts.get("general", "blurry, low quality"))
    
    def get_aspect_ratio_dimensions(self, aspect_ratio: str) -> Optional[Dict]:
        """Get dimensions for an aspect ratio"""
        ratios = self.config.get("aspect_ratios", {})
        return ratios.get(aspect_ratio)
    
    async def ensure_connected(self) -> bool:
        """Ensure connection to Runware API is established with retry logic"""
        if self.connected:
            return True
        
        max_retries = self._max_retries
        retry_delay = self.settings.get("retry_delay", 2)
        
        for attempt in range(max_retries):
            try:
                await self.runware.connect()
                self.connected = True
                self._connection_retries = 0
                logging.info("Successfully connected to Runware API")
                return True
            except Exception as e:
                self._connection_retries += 1
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logging.warning(f"Runware connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
                    import asyncio
                    await asyncio.sleep(wait_time)
                else:
                    logging.error(f"Failed to connect to Runware API after {max_retries} attempts: {e}")
                    return False
        
        return False
    
    async def disconnect(self):
        """Disconnect from Runware API"""
        if self.connected:
            try:
                await self.runware.disconnect()
                self.connected = False
                logging.info("Disconnected from Runware API")
            except Exception as e:
                logging.warning(f"Error disconnecting from Runware: {e}")
    
    async def generate_image(
        self, 
        args: Union[str, Dict],
        model: str = None,
        num_images: int = 1,
        negative_prompt: str = None,
        width: int = None,
        height: int = None,
        steps: int = None,
        cfg_scale: float = None,
        seed: int = None,
        aspect_ratio: str = None
    ) -> Dict[str, Any]:
        """
        Generate images based on a text prompt.
        
        Args:
            args: Either a string prompt or dict containing prompt and options
            model: Model key from config (e.g., "flux", "sdxl", "anime")
            num_images: Number of images to generate (max 4)
            negative_prompt: Things to avoid in the generated image
            width: Image width (overrides model default)
            height: Image height (overrides model default)
            steps: Number of inference steps
            cfg_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            aspect_ratio: Aspect ratio key (e.g., "16:9", "1:1")
            
        Returns:
            Dict with generated images or error information
        """
        # Parse input arguments
        if isinstance(args, dict):
            prompt = args.get('prompt', '')
            model = args.get('model', model)
            num_images = args.get('num_images', num_images)
            negative_prompt = args.get('negative_prompt', negative_prompt)
            width = args.get('width', width)
            height = args.get('height', height)
            steps = args.get('steps', steps)
            cfg_scale = args.get('cfg_scale', cfg_scale)
            seed = args.get('seed', seed)
            aspect_ratio = args.get('aspect_ratio', aspect_ratio)
        else:
            prompt = str(args)
        
        # Get model configuration
        model = model or self.settings.get("default_model", "flux")
        model_config = self.get_model_info(model)
        
        if not model_config:
            logging.warning(f"Model '{model}' not found, using default")
            model = self.settings.get("default_model", "flux")
            model_config = self.get_model_info(model) or {}
        
        model_id = model_config.get("model_id", "runware:101@1")
        
        # Handle aspect ratio
        if aspect_ratio:
            ratio_dims = self.get_aspect_ratio_dimensions(aspect_ratio)
            if ratio_dims:
                width = width or ratio_dims.get("width")
                height = height or ratio_dims.get("height")
        
        # Apply defaults from model config
        width = width or model_config.get("default_width", 1024)
        height = height or model_config.get("default_height", 1024)
        steps = steps or model_config.get("default_steps", 30)
        cfg_scale = cfg_scale or model_config.get("default_cfg_scale", 7.5)
        max_images = model_config.get("max_images", 4)
        num_images = min(num_images, max_images)
        
        # Ensure dimensions are within limits and divisible by 64
        max_width = model_config.get("max_width", 2048)
        max_height = model_config.get("max_height", 2048)
        min_width = model_config.get("min_width", 512)
        min_height = model_config.get("min_height", 512)
        step_size = model_config.get("step_size", 64)
        
        width = max(min_width, min(width, max_width))
        height = max(min_height, min(height, max_height))
        width = (width // step_size) * step_size
        height = (height // step_size) * step_size
        
        # Get negative prompt
        if negative_prompt is None:
            category = model_config.get("category", "general")
            negative_prompt = self.get_default_negative_prompt(category)
        
        try:
            if not await self.ensure_connected():
                return {
                    "success": False,
                    "error": "Failed to connect to image generation API",
                    "prompt": prompt,
                    "image_urls": [],
                    "image_count": 0
                }
            
            # Build request parameters
            request_params = {
                "positivePrompt": prompt,
                "model": model_id,
                "numberResults": num_images,
                "width": width,
                "height": height,
                "steps": steps,
                "CFGScale": cfg_scale,
                "outputFormat": self.settings.get("output_format", "WEBP")
            }
            
            if model_config.get("supports_negative_prompt", True) and negative_prompt:
                request_params["negativePrompt"] = negative_prompt
            
            if seed is not None:
                request_params["seed"] = seed
            
            request_image = IImageInference(**request_params)
            images = await self.runware.imageInference(requestImage=request_image)
            
            result = {
                "success": True,
                "prompt": prompt,
                "model": model,
                "model_name": model_config.get("name", model),
                "image_urls": [],
                "image_count": 0,
                "width": width,
                "height": height
            }
            
            if images:
                for image in images:
                    if hasattr(image, 'imageURL') and image.imageURL:
                        result["image_urls"].append(image.imageURL)
                    elif hasattr(image, 'imageDataURI') and image.imageDataURI:
                        result["image_urls"].append(image.imageDataURI)
            
            result["image_count"] = len(result["image_urls"])
            
            if result["image_count"] > 0:
                logging.info(f"Generated {result['image_count']} images with {model} for: {prompt[:50]}...")
            else:
                logging.warning(f"Image generation succeeded but no images received for: {prompt[:50]}...")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in generate_image: {e}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt,
                "model": model,
                "image_urls": [],
                "image_count": 0
            }
    
    async def upscale_image(
        self,
        args: Union[str, Dict],
        scale_factor: int = 2,
        model: str = None
    ) -> Dict[str, Any]:
        """
        Upscale an image to higher resolution.
        
        Args:
            args: Image URL or dict with image_url/image_data and options
            scale_factor: Upscale factor (2 or 4)
            model: Upscale model key (e.g., "clarity", "swinir")
            
        Returns:
            Dict with upscaled image information
        """
        image_data = None
        if isinstance(args, dict):
            image_url = args.get('image_url', '')
            scale_factor = args.get('scale_factor', scale_factor)
            model = args.get('model', model)
        else:
            image_url = str(args)
        
        # Validate URL
        is_valid, error_msg = self._validate_url(image_url)
        if not is_valid:
            return {
                "success": False,
                "error": f"Invalid image URL: {error_msg}",
                "image_urls": [],
                "image_count": 0
            }
        
        model = model or self.settings.get("default_upscale_model", "clarity")
        upscale_models = self.get_upscale_models()
        model_config = upscale_models.get(model, {})
        
        if not model_config:
            model = self.settings.get("default_upscale_model", "clarity")
            model_config = upscale_models.get(model, {})
        
        model_id = model_config.get("model_id", "runware:500@1")
        supported_factors = model_config.get("supported_factors", [2, 4])
        
        if scale_factor not in supported_factors:
            scale_factor = supported_factors[0] if supported_factors else 2
        
        try:
            if not await self.ensure_connected():
                return {
                    "success": False,
                    "error": "Failed to connect to image processing API",
                    "image_urls": [],
                    "image_count": 0
                }
            
            # Pass URL directly to Runware API (it handles downloading)
            logging.info(f"Sending image URL to Runware upscale API: {image_url}")
            upscale_payload = IImageUpscale(
                inputImage=image_url,
                upscaleFactor=scale_factor,
                model=model_id
            )
            
            upscaled_images = await self.runware.imageUpscale(upscaleGanPayload=upscale_payload)
            
            result = {
                "success": True,
                "original_url": image_url,
                "scale_factor": scale_factor,
                "model": model,
                "model_name": model_config.get("name", model),
                "image_urls": [],
                "image_count": 0
            }
            
            if upscaled_images:
                for image in upscaled_images:
                    if hasattr(image, 'imageURL') and image.imageURL:
                        result["image_urls"].append(image.imageURL)
                    elif hasattr(image, 'imageSrc') and image.imageSrc:
                        result["image_urls"].append(image.imageSrc)
            
            result["image_count"] = len(result["image_urls"])
            
            if result["image_count"] > 0:
                logging.info(f"Successfully upscaled image by {scale_factor}x with {model}")
            else:
                logging.warning("Upscaling succeeded but no images returned")
            
            return result
                
        except Exception as e:
            logging.error(f"Error in upscale_image: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_url": image_url,
                "image_urls": [],
                "image_count": 0
            }
    
    async def remove_background(
        self,
        args: Union[str, Dict],
        model: str = None
    ) -> Dict[str, Any]:
        """
        Remove background from an image.
        
        Args:
            args: Image URL or dict with image_url/image_data and options
            model: Background removal model key
            
        Returns:
            Dict with processed image information
        """
        if isinstance(args, dict):
            image_url = args.get('image_url', '')
            model = args.get('model', model)
        else:
            image_url = str(args)
        
        # Validate URL
        is_valid, error_msg = self._validate_url(image_url)
        if not is_valid:
            return {
                "success": False,
                "error": f"Invalid image URL: {error_msg}",
                "image_urls": [],
                "image_count": 0
            }
        
        model = model or self.settings.get("default_background_removal_model", "bria")
        bg_models = self.get_background_removal_models()
        model_config = bg_models.get(model, {})
        
        if not model_config:
            model = self.settings.get("default_background_removal_model", "bria")
            model_config = bg_models.get(model, {})
        
        model_id = model_config.get("model_id", "runware:110@1")
        
        try:
            if not await self.ensure_connected():
                return {
                    "success": False,
                    "error": "Failed to connect to image processing API",
                    "image_urls": [],
                    "image_count": 0
                }
            
            # Pass URL directly to Runware API (it handles downloading)
            logging.info(f"Sending image URL to Runware background removal API: {image_url}")
            bg_removal_payload = IImageBackgroundRemoval(
                inputImage=image_url,
                model=model_id,
                outputFormat="PNG"
            )
            
            processed_images = await self.runware.imageBackgroundRemoval(
                removeImageBackgroundPayload=bg_removal_payload
            )
            
            result = {
                "success": True,
                "original_url": image_url,
                "operation": "remove_background",
                "model": model,
                "model_name": model_config.get("name", model),
                "image_urls": [],
                "image_count": 0
            }
            
            if processed_images:
                for image in processed_images:
                    if hasattr(image, 'imageURL') and image.imageURL:
                        result["image_urls"].append(image.imageURL)
            
            result["image_count"] = len(result["image_urls"])
            
            if result["image_count"] > 0:
                logging.info(f"Successfully removed background with {model}")
            else:
                logging.warning("Background removal succeeded but no images returned")
            
            return result
                
        except Exception as e:
            logging.error(f"Error in remove_background: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_url": image_url,
                "operation": "remove_background",
                "image_urls": [],
                "image_count": 0
            }
    
    async def edit_image(self, args, operation: str = "remove_background") -> Dict[str, Any]:
        """Edit an image - backward compatibility alias"""
        if isinstance(args, dict):
            operation = args.get('operation', operation)
        
        if operation == "remove_background":
            return await self.remove_background(args)
        else:
            return {
                "success": False,
                "error": f"Unsupported operation: {operation}",
                "operation": operation,
                "image_urls": [],
                "image_count": 0
            }
    
    async def image_to_text(self, args: Union[str, Dict]) -> Dict[str, Any]:
        """Generate a text caption/description from an image."""
        if isinstance(args, dict):
            image_url = args.get('image_url', '')
        else:
            image_url = str(args)
        
        try:
            if not await self.ensure_connected():
                return {
                    "success": False,
                    "error": "Failed to connect to image processing API",
                    "caption": ""
                }
            
            image_data = await self._download_image(image_url)
            if image_data is None:
                return {
                    "success": False,
                    "error": "Failed to download input image",
                    "caption": ""
                }
            
            temp_path = await self._save_temp_image(image_data)
            
            try:
                caption_request = IImageCaption(inputImage=temp_path)
                caption_result = await self.runware.imageCaption(requestImageToText=caption_request)
                
                result = {
                    "success": True,
                    "image_url": image_url,
                    "caption": ""
                }
                
                if caption_result:
                    if hasattr(caption_result, 'text'):
                        result["caption"] = caption_result.text
                    elif isinstance(caption_result, list) and len(caption_result) > 0:
                        if hasattr(caption_result[0], 'text'):
                            result["caption"] = caption_result[0].text
                
                if result["caption"]:
                    logging.info(f"Generated caption: {result['caption'][:50]}...")
                
                return result
                
            finally:
                await self._cleanup_temp_file(temp_path)
                
        except Exception as e:
            logging.error(f"Error in image_to_text: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_url": image_url,
                "caption": ""
            }
    
    async def enhance_prompt(
        self,
        args: Union[str, Dict],
        num_versions: int = 3,
        max_length: int = 200
    ) -> Dict[str, Any]:
        """Enhance a text prompt with AI for better image generation results."""
        if isinstance(args, dict):
            prompt = args.get('prompt', '')
            num_versions = args.get('num_versions', num_versions)
            max_length = args.get('max_length', max_length)
        else:
            prompt = str(args)
        
        try:
            if not await self.ensure_connected():
                return {
                    "success": False,
                    "error": "Failed to connect to API",
                    "enhanced_prompts": [],
                    "prompt_count": 0
                }
            
            enhance_request = IPromptEnhance(
                prompt=prompt,
                promptVersions=num_versions,
                promptMaxLength=max_length
            )
            
            enhanced = await self.runware.promptEnhance(promptEnhancer=enhance_request)
            
            result = {
                "success": True,
                "original_prompt": prompt,
                "enhanced_prompts": [],
                "prompt_count": 0
            }
            
            if enhanced:
                for item in enhanced:
                    if hasattr(item, 'text') and item.text:
                        result["enhanced_prompts"].append(item.text)
            
            result["prompt_count"] = len(result["enhanced_prompts"])
            
            if result["prompt_count"] > 0:
                logging.info(f"Generated {result['prompt_count']} enhanced prompts")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in enhance_prompt: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_prompt": prompt,
                "enhanced_prompts": [],
                "prompt_count": 0
            }
    
    async def photo_maker(
        self,
        args: Union[str, Dict],
        input_images: List[str] = None,
        style: str = "No style",
        strength: int = 40,
        steps: int = 35,
        num_images: int = 1,
        width: int = 1024,
        height: int = 1024
    ) -> Dict[str, Any]:
        """Generate images based on reference photos and a text prompt."""
        if isinstance(args, dict):
            prompt = args.get('prompt', '')
            input_images = args.get('input_images', input_images or [])
            style = args.get('style', style)
            strength = args.get('strength', strength)
            steps = args.get('steps', steps)
            num_images = args.get('num_images', num_images)
            width = args.get('width', width)
            height = args.get('height', height)
        else:
            prompt = str(args)
            input_images = input_images or []
        
        try:
            if not await self.ensure_connected():
                return {
                    "success": False,
                    "error": "Failed to connect to API",
                    "image_urls": [],
                    "image_count": 0
                }
            
            photo_request = IPhotoMaker(
                positivePrompt=prompt,
                inputImages=input_images,
                style=style,
                strength=strength,
                steps=steps,
                numberResults=num_images,
                width=width,
                height=height,
                outputFormat=self.settings.get("output_format", "WEBP"),
                taskUUID=str(uuid.uuid4())
            )
            
            photos = await self.runware.photoMaker(requestPhotoMaker=photo_request)
            
            result = {
                "success": True,
                "prompt": prompt,
                "style": style,
                "image_urls": [],
                "image_count": 0
            }
            
            if photos:
                for photo in photos:
                    if hasattr(photo, 'imageURL') and photo.imageURL:
                        result["image_urls"].append(photo.imageURL)
            
            result["image_count"] = len(result["image_urls"])
            
            if result["image_count"] > 0:
                logging.info(f"Generated {result['image_count']} photos with PhotoMaker")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in photo_maker: {e}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt,
                "image_urls": [],
                "image_count": 0
            }
    
    async def generate_image_with_refiner(
        self,
        args: Union[str, Dict],
        model: str = "sdxl",
        num_images: int = 1,
        negative_prompt: str = None
    ) -> Dict[str, Any]:
        """Generate high-quality images with refiner model."""
        if isinstance(args, dict):
            args['model'] = args.get('model', model)
        else:
            args = {
                'prompt': str(args),
                'model': model,
                'num_images': num_images,
                'negative_prompt': negative_prompt
            }
        
        return await self.generate_image(args)
    
    # ================== Helper Methods ==================
    
    def _validate_url(self, url: str) -> tuple[bool, str]:
        """Validate if a string is a valid image URL"""
        if not url or not isinstance(url, str):
            return False, "No URL provided"
        
        url = url.strip()
        
        # Check for valid URL scheme
        if not url.startswith(('http://', 'https://')):
            return False, f"Invalid URL scheme. URL must start with http:// or https://. Got: {url[:50]}..."
        
        # Check for common image extensions or known image hosts
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff')
        image_hosts = ('cdn.discordapp.com', 'media.discordapp.net', 'i.imgur.com', 
                       'imgur.com', 'cloudinary.com', 'unsplash.com', 'pexels.com',
                       'runware.ai', 'replicate.delivery')
        
        url_lower = url.lower()
        has_image_ext = any(ext in url_lower for ext in image_extensions)
        is_image_host = any(host in url_lower for host in image_hosts)
        
        # URLs with query params might not have extension visible
        if not has_image_ext and not is_image_host and '?' not in url:
            logging.warning(f"URL may not be an image: {url[:100]}")
        
        return True, "OK"
    
    async def _download_image(self, url: str) -> Optional[bytes]:
        """Download image from URL with validation and Discord CDN support"""
        # Validate URL first
        is_valid, error_msg = self._validate_url(url)
        if not is_valid:
            logging.error(f"Invalid image URL: {error_msg}")
            return None
        
        url = url.strip()
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'image/*,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            # For Discord CDN URLs, add bot authorization if available
            if 'cdn.discordapp.com' in url or 'media.discordapp.net' in url:
                try:
                    from src.config.config import DISCORD_TOKEN
                    if DISCORD_TOKEN:
                        headers['Authorization'] = f'Bot {DISCORD_TOKEN}'
                        logging.debug("Using Discord bot token for CDN access")
                except ImportError:
                    pass
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30), headers=headers) as resp:
                    if resp.status == 200:
                        content_type = resp.headers.get('Content-Type', '')
                        if not content_type.startswith('image/') and 'octet-stream' not in content_type:
                            logging.warning(f"Response may not be an image. Content-Type: {content_type}")
                        return await resp.read()
                    elif resp.status == 404:
                        logging.error(f"Image not found (404). URL: {url[:100]}...")
                        return None
                    elif resp.status == 403:
                        logging.error(f"Access denied (403). The image URL may have expired or requires re-uploading. URL: {url[:100]}...")
                        return None
                    else:
                        logging.error(f"Failed to download image: HTTP {resp.status} for {url[:100]}...")
                        return None
        except aiohttp.ClientError as e:
            logging.error(f"Network error downloading image: {e}")
            return None
        except Exception as e:
            logging.error(f"Error downloading image: {e}")
            return None
    
    async def _save_temp_image(self, image_data: bytes, suffix: str = '.jpg') -> str:
        """Save image data to temporary file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(image_data)
            return temp_file.name
    
    async def _cleanup_temp_file(self, file_path: str):
        """Clean up temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logging.warning(f"Failed to clean up temp file {file_path}: {e}")
    
    def list_models(self) -> str:
        """Get a formatted string listing all available models"""
        models = self.get_available_models()
        lines = ["**Available Image Models:**"]
        for key, config in models.items():
            name = config.get("name", key)
            desc = config.get("description", "")
            lines.append(f"• `{key}` - {name}: {desc}")
        return "\n".join(lines)
    
    def list_upscale_models(self) -> str:
        """Get a formatted string listing all upscale models"""
        models = self.get_upscale_models()
        lines = ["**Available Upscale Models:**"]
        for key, config in models.items():
            name = config.get("name", key)
            factors = config.get("supported_factors", [2])
            lines.append(f"• `{key}` - {name} (factors: {factors})")
        return "\n".join(lines)
    
    def reload_config(self):
        """Reload configuration from file"""
        global IMAGE_CONFIG
        IMAGE_CONFIG = load_image_config()
        self.config = IMAGE_CONFIG
        self.settings = self.config.get("settings", {})
        logging.info("Image configuration reloaded")
