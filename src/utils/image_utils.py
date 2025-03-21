import io
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from runware import IImageInference
import os
import time

class ImageGenerator:
    def __init__(self, api_key: str):
        """
        Initialize the image generator with the Runware API key.
        
        Args:
            api_key: API key for Runware
        """
        from runware import Runware
        self.runware = Runware(api_key=api_key)
        self.connected = False
    
    async def ensure_connected(self):
        """Ensure connection to Runware API is established"""
        if not self.connected:
            await self.runware.connect()
            self.connected = True
        
    async def generate_image(self, prompt: str, num_images: int = 1, negative_prompt: str = "blurry, distorted, low quality"):
        """
        Generate images based on a text prompt
        
        Args:
            prompt: The text prompt for image generation
            num_images: Number of images to generate (max 4)
            negative_prompt: Things to avoid in the generated image
            
        Returns:
            Dict with generated images or error information
        """
        num_images = min(num_images, 4)
        
        try:
            # Ensure connection is established
            await self.ensure_connected()
            
            # Configure request for Runware
            request_image = IImageInference(
                positivePrompt=prompt,
                numberResults=num_images,
                model="runware:5@1",  # Specify the model
                negativePrompt=negative_prompt,
                height=512,
                width=512,
            )
            
            # Generate images
            images = await self.runware.imageInference(requestImage=request_image)
            
            result = {
                "success": True,
                "prompt": prompt,
                "binary_images": [],
                "image_urls": [],  # Initialize empty image URLs list
                "image_count": 0
            }
            
            # Process generated images - handle different response formats
            if images:
                # Extract image URLs based on response structure
                image_urls = []
                
                # Case 1: Response is a direct list/iterable of image objects
                if hasattr(images, '__iter__') and not hasattr(images, 'images'):
                    for image in images:
                        if hasattr(image, 'imageURL'):
                            image_urls.append(image.imageURL)
                
                # Case 2: Response has an 'images' attribute with URLs
                elif hasattr(images, 'images') and images.images:
                    image_urls = images.images
                
                # Update result with image info
                result["image_count"] = len(image_urls)
                result["image_urls"] = image_urls  # Add image URLs to result
                
                # Get binary data for each image
                for img_url in image_urls:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(img_url) as resp:
                                if resp.status == 200:
                                    image_data = await resp.read()
                                    result["binary_images"].append(image_data)
                    except Exception as e:
                        logging.error(f"Error downloading image {img_url}: {str(e)}")
            
            # Log success or failure
            if result["image_count"] > 0:
                logging.info(f"Generated {result['image_count']} images for prompt: {prompt[:50]}...")
            else:
                logging.warning(f"Image generation succeeded but no images were received for prompt: {prompt[:50]}...")
            
            return result
                
        except Exception as e:
            error_message = f"Error in generate_image: {str(e)}"
            logging.error(error_message)
            return {
                "success": False, 
                "error": str(e),
                "prompt": prompt,
                "image_urls": [],  # Include empty image_urls even in error case
                "image_count": 0,
                "binary_images": []
            }