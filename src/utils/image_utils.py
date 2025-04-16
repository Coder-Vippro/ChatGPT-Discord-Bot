import io
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from runware import IImageInference
import os
import time
import uuid
from runware import IPromptEnhance, IImageBackgroundRemoval, IImageCaption, IImageUpscale, IPhotoMaker, IRefiner

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
    
    async def edit_image(self, image_url: str, operation: str = "remove_background"):
        """
        Edit an image using various operations like background removal
        
        Args:
            image_url: URL of the image to edit
            operation: Type of edit operation (currently supports 'remove_background')
            
        Returns:
            Dict with edited image information
        """
        try:
            # Ensure connection is established
            await self.ensure_connected()
            
            # Download the image first
            image_data = None
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        return {
                            "success": False,
                            "error": f"Failed to download image, status: {resp.status}",
                            "operation": operation
                        }
                    image_data = await resp.read()
            
            if operation == "remove_background":
                # Import the necessary class from runware
                from runware import IImageBackgroundRemoval
                
                # Create a temporary file to store the image
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file.write(image_data)
                    temp_path = temp_file.name
                
                try:
                    # Configure background removal request
                    background_removal_payload = IImageBackgroundRemoval(
                        image_initiator=temp_path
                    )
                    
                    # Process the image
                    processed_images = await self.runware.imageBackgroundRemoval(
                        removeImageBackgroundPayload=background_removal_payload
                    )
                    
                    # Clean up temporary file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    
                    # Handle the response
                    result = {
                        "success": True,
                        "operation": operation,
                        "original_url": image_url,
                        "image_urls": [],
                        "binary_images": []
                    }
                    
                    # Extract image URLs from response
                    if processed_images:
                        for image in processed_images:
                            if hasattr(image, 'imageURL'):
                                result["image_urls"].append(image.imageURL)
                                
                        # Download the edited images
                        for img_url in result["image_urls"]:
                            try:
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(img_url) as resp:
                                        if resp.status == 200:
                                            edited_image_data = await resp.read()
                                            result["binary_images"].append(edited_image_data)
                            except Exception as e:
                                logging.error(f"Error downloading edited image {img_url}: {str(e)}")
                    
                    result["image_count"] = len(result["image_urls"])
                    
                    if result["image_count"] > 0:
                        logging.info(f"Successfully removed background from image")
                    else:
                        logging.warning("Background removal succeeded but no images were returned")
                    
                    return result
                    
                except Exception as e:
                    logging.error(f"Error in background removal: {str(e)}")
                    return {
                        "success": False,
                        "error": f"Error in background removal: {str(e)}",
                        "operation": operation
                    }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported edit operation: {operation}",
                    "operation": operation
                }
                
        except Exception as e:
            error_message = f"Error in edit_image: {str(e)}"
            logging.error(error_message)
            return {
                "success": False, 
                "error": str(e),
                "operation": operation,
                "image_urls": [],
                "image_count": 0,
                "binary_images": []
            }
    
    async def enhance_prompt(self, prompt: str, num_versions: int = 3, max_length: int = 64) -> Dict[str, Any]:
        """
        Enhance a text prompt with AI to create more detailed/creative versions
        
        Args:
            prompt: The original prompt text
            num_versions: Number of enhanced versions to generate
            max_length: Maximum length of each enhanced prompt
            
        Returns:
            Dict with enhanced prompt information
        """
        try:
            # Ensure connection is established
            await self.ensure_connected()
            
            # Configure prompt enhancement request
            prompt_enhancer = IPromptEnhance(
                prompt=prompt,
                promptVersions=num_versions,
                promptMaxLength=max_length,
            )
            
            # Get enhanced prompts
            enhanced_prompts = await self.runware.promptEnhance(promptEnhancer=prompt_enhancer)
            
            result = {
                "success": True,
                "original_prompt": prompt,
                "enhanced_prompts": [],
                "prompt_count": 0
            }
            
            # Extract enhanced prompts from the response
            if enhanced_prompts:
                for enhanced_prompt in enhanced_prompts:
                    if hasattr(enhanced_prompt, 'text') and enhanced_prompt.text:
                        result["enhanced_prompts"].append(enhanced_prompt.text)
            
            result["prompt_count"] = len(result["enhanced_prompts"])
            
            # Log success or failure
            if result["prompt_count"] > 0:
                logging.info(f"Generated {result['prompt_count']} enhanced prompts for: {prompt[:50]}...")
            else:
                logging.warning(f"Prompt enhancement succeeded but no prompts were received")
            
            return result
                
        except Exception as e:
            error_message = f"Error in enhance_prompt: {str(e)}"
            logging.error(error_message)
            return {
                "success": False, 
                "error": str(e),
                "original_prompt": prompt,
                "enhanced_prompts": [],
                "prompt_count": 0
            }
    
    async def image_to_text(self, image_url: str) -> Dict[str, Any]:
        """
        Convert an image to a text description
        
        Args:
            image_url: URL of the image to analyze
            
        Returns:
            Dict with image caption information
        """
        try:
            # Ensure connection is established
            await self.ensure_connected()
            
            # Download the image first
            image_data = None
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        return {
                            "success": False,
                            "error": f"Failed to download image, status: {resp.status}"
                        }
                    image_data = await resp.read()
            
            # Create a temporary file to store the image
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name
                
            try:
                # Configure image caption request
                request_image_to_text = IImageCaption(
                    image_initiator=temp_path
                )
                
                # Get image caption
                image_caption = await self.runware.imageCaption(
                    requestImageToText=request_image_to_text
                )
                
                # Clean up temporary file
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                result = {
                    "success": True,
                    "image_url": image_url,
                    "caption": ""
                }
                
                # Extract caption from the response
                if image_caption and hasattr(image_caption, 'text'):
                    result["caption"] = image_caption.text
                
                # Log success or failure
                if result["caption"]:
                    logging.info(f"Generated caption for image: {result['caption'][:50]}...")
                else:
                    logging.warning(f"Image caption generation succeeded but no text was received")
                
                return result
                
            except Exception as e:
                # Clean up temporary file
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
                logging.error(f"Error in image captioning: {str(e)}")
                return {
                    "success": False,
                    "error": f"Error in image captioning: {str(e)}",
                    "image_url": image_url
                }
                
        except Exception as e:
            error_message = f"Error in image_to_text: {str(e)}"
            logging.error(error_message)
            return {
                "success": False,
                "error": str(e),
                "image_url": image_url,
                "caption": ""
            }
    
    async def upscale_image(self, image_url: str, scale_factor: int = 4) -> Dict[str, Any]:
        """
        Upscale an image to a higher resolution
        
        Args:
            image_url: URL of the image to upscale
            scale_factor: Factor by which to upscale the image (2-4)
            
        Returns:
            Dict with upscaled image information
        """
        # Ensure scale factor is within valid range
        scale_factor = max(2, min(scale_factor, 4))
        
        try:
            # Ensure connection is established
            await self.ensure_connected()
            
            # Download the image first
            image_data = None
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        return {
                            "success": False,
                            "error": f"Failed to download image, status: {resp.status}",
                            "image_urls": [],
                            "image_count": 0,
                            "binary_images": []
                        }
                    image_data = await resp.read()
            
            # Create a temporary file to store the image
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name
                
            try:
                # Configure upscale request
                upscale_payload = IImageUpscale(
                    inputImage=temp_path,
                    upscaleFactor=scale_factor
                )
                
                # Get upscaled images
                upscaled_images = await self.runware.imageUpscale(
                    upscaleGanPayload=upscale_payload
                )
                
                # Clean up temporary file
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                result = {
                    "success": True,
                    "original_url": image_url,
                    "scale_factor": scale_factor,
                    "image_urls": [],
                    "image_count": 0,
                    "binary_images": []
                }
                
                # Extract image URLs from response
                if upscaled_images:
                    for image in upscaled_images:
                        if hasattr(image, 'imageSrc'):
                            result["image_urls"].append(image.imageSrc)
                
                result["image_count"] = len(result["image_urls"])
                
                # Get binary data for each image
                for img_url in result["image_urls"]:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(img_url) as resp:
                                if resp.status == 200:
                                    image_data = await resp.read()
                                    result["binary_images"].append(image_data)
                    except Exception as e:
                        logging.error(f"Error downloading upscaled image {img_url}: {str(e)}")
                
                # Log success or failure
                if result["image_count"] > 0:
                    logging.info(f"Successfully upscaled image by factor {scale_factor}")
                else:
                    logging.warning(f"Image upscaling succeeded but no images were returned")
                
                return result
                
            except Exception as e:
                # Clean up temporary file
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
                logging.error(f"Error in image upscaling: {str(e)}")
                return {
                    "success": False,
                    "error": f"Error in image upscaling: {str(e)}",
                    "image_urls": [],
                    "image_count": 0,
                    "binary_images": []
                }
                
        except Exception as e:
            error_message = f"Error in upscale_image: {str(e)}"
            logging.error(error_message)
            return {
                "success": False, 
                "error": str(e),
                "original_url": image_url,
                "image_urls": [],
                "image_count": 0,
                "binary_images": []
            }
    
    async def photo_maker(self, prompt: str, input_images: List[str], style: str = "No style", 
                          strength: int = 40, steps: int = 35, num_images: int = 1, 
                          height: int = 512, width: int = 512) -> Dict[str, Any]:
        """
        Generate images based on reference photos and a text prompt
        
        Args:
            prompt: The text prompt describing what to generate
            input_images: List of reference image URLs to use as input
            style: Style to apply to the generated image
            strength: Strength of the input images' influence (0-100)
            steps: Number of generation steps
            num_images: Number of images to generate
            height: Output image height
            width: Output image width
            
        Returns:
            Dict with generated image information
        """
        try:
            # Ensure connection is established
            await self.ensure_connected()
            
            # Configure request for photo maker
            request_photo = IPhotoMaker(
                positivePrompt=prompt,
                steps=steps,
                numberResults=num_images,
                height=height,
                width=width,
                style=style,
                strength=strength,
                outputFormat="WEBP",
                includeCost=True,
                taskUUID=str(uuid.uuid4()),
                inputImages=input_images,
            )
            
            # Generate photos
            photos = await self.runware.photoMaker(requestPhotoMaker=request_photo)
            
            result = {
                "success": True,
                "prompt": prompt,
                "binary_images": [],
                "image_urls": [],
                "image_count": 0
            }
            
            # Extract image URLs from response
            if photos:
                for photo in photos:
                    if hasattr(photo, 'imageURL'):
                        result["image_urls"].append(photo.imageURL)
            
            result["image_count"] = len(result["image_urls"])
            
            # Get binary data for each image
            for img_url in result["image_urls"]:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(img_url) as resp:
                            if resp.status == 200:
                                image_data = await resp.read()
                                result["binary_images"].append(image_data)
                except Exception as e:
                    logging.error(f"Error downloading photo maker image {img_url}: {str(e)}")
            
            # Log success or failure
            if result["image_count"] > 0:
                logging.info(f"Generated {result['image_count']} photos with PhotoMaker for prompt: {prompt[:50]}...")
            else:
                logging.warning(f"PhotoMaker succeeded but no images were received for prompt: {prompt[:50]}...")
            
            return result
                
        except Exception as e:
            error_message = f"Error in photo_maker: {str(e)}"
            logging.error(error_message)
            return {
                "success": False, 
                "error": str(e),
                "prompt": prompt,
                "image_urls": [],
                "image_count": 0,
                "binary_images": []
            }
    
    async def generate_image_with_refiner(self, prompt: str, num_images: int = 1, 
                                         negative_prompt: str = "blurry, distorted, low quality",
                                         model: str = "civitai:101055@128078",
                                         refiner_start_step: int = 20) -> Dict[str, Any]:
        """
        Generate images with a refiner model for better quality
        
        Args:
            prompt: The text prompt for image generation
            num_images: Number of images to generate (max 4)
            negative_prompt: Things to avoid in the generated image
            model: Model to use for generation
            refiner_start_step: Step at which to start refining
            
        Returns:
            Dict with generated images or error information
        """
        num_images = min(num_images, 4)
        
        try:
            # Ensure connection is established
            await self.ensure_connected()
            
            # Configure refiner
            refiner = IRefiner(
                model=model,
                startStep=refiner_start_step,
                startStepPercentage=None,
            )
            
            # Configure request for Runware
            request_image = IImageInference(
                positivePrompt=prompt,
                numberResults=num_images,
                model=model,
                negativePrompt=negative_prompt,
                height=512,
                width=512,
                refiner=refiner
            )
            
            # Generate images
            images = await self.runware.imageInference(requestImage=request_image)
            
            result = {
                "success": True,
                "prompt": prompt,
                "binary_images": [],
                "image_urls": [],
                "image_count": 0
            }
            
            # Process generated images
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
                result["image_urls"] = image_urls
                
                # Get binary data for each image
                for img_url in image_urls:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(img_url) as resp:
                                if resp.status == 200:
                                    image_data = await resp.read()
                                    result["binary_images"].append(image_data)
                    except Exception as e:
                        logging.error(f"Error downloading refined image {img_url}: {str(e)}")
            
            # Log success or failure
            if result["image_count"] > 0:
                logging.info(f"Generated {result['image_count']} refined images for prompt: {prompt[:50]}...")
            else:
                logging.warning(f"Refined image generation succeeded but no images were received for prompt: {prompt[:50]}...")
            
            return result
                
        except Exception as e:
            error_message = f"Error in generate_image_with_refiner: {str(e)}"
            logging.error(error_message)
            return {
                "success": False, 
                "error": str(e),
                "prompt": prompt,
                "image_urls": [],
                "image_count": 0,
                "binary_images": []
            }