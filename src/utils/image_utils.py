import io
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from runware import IImageInference

class ImageGenerator:
    def __init__(self, api_key: str):
        """
        Initialize the image generator with the Runware API key.
        
        Args:
            api_key: API key for Runware
        """
        from runware import Runware
        self.runware = Runware(api_key=api_key)
        
    async def generate_image(self, prompt: str, num_images: int = 1) -> Dict[str, Any]:
        """
        Generate images using AI from a text prompt.
        
        Args:
            prompt (str): The text prompt describing the image to generate
            num_images (int): Number of images to generate (default: 1, max: 4)
        
        Returns:
            dict: Dictionary containing URLs and binary data of generated images
        """
        try:
            # Limit the number of images to maximum 4
            num_images = min(max(1, num_images), 4)
            
            # Create an image generation request
            request_image = IImageInference(
                positivePrompt=prompt,
                model="runware:100@1",
                numberResults=num_images,
                height=512,
                width=512
            )
            
            # Call the API to get the results
            images = await self.runware.imageInference(requestImage=request_image)
            
            # Check the API's return value
            if images is None:
                return {"success": False, "error": "Image generation failed - API returned no results"}
            
            # Format the results with URLs
            result = {
                "success": True,
                "prompt": prompt,
                "image_count": len(images),
                "image_urls": [image.imageURL for image in images],
                "binary_images": []
            }
            
            # Download images for sending as attachments
            async with aiohttp.ClientSession() as session:
                for image_url in result["image_urls"]:
                    try:
                        async with session.get(image_url) as resp:
                            if resp.status == 200:
                                image_data = await resp.read()
                                result["binary_images"].append(image_data)
                            else:
                                logging.error(f"Failed to download image: {image_url} with status {resp.status}")
                    except Exception as e:
                        logging.error(f"Error downloading image {image_url}: {str(e)}")
            
            return result
            
        except Exception as e:
            error_message = f"Error generating images: {str(e)}"
            logging.error(error_message)
            return {"success": False, "error": error_message}