import discord
import asyncio
import json
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional
import io
import aiohttp

from src.utils.openai_utils import process_tool_calls, prepare_messages_for_api, get_tools_for_model
from src.utils.pdf_utils import process_pdf, send_response
from src.utils.code_utils import extract_code_blocks, execute_code

user_tasks = {}

class MessageHandler:
    def __init__(self, bot, db_handler, openai_client, image_generator):
        """
        Initialize the message handler.
        
        Args:
            bot: Discord bot instance
            db_handler: Database handler instance
            openai_client: OpenAI client instance
            image_generator: Image generator instance
        """
        self.bot = bot
        self.db = db_handler
        self.client = openai_client
        self.image_generator = image_generator
        
        # Tool functions mapping
        self.tool_functions = {
            "google_search": self._google_search,
            "scrape_webpage": self._scrape_webpage,
            "code_interpreter": self._code_interpreter,
            "generate_image": self._generate_image
        }
        
        # Register message event handlers
        self._setup_event_handlers()
        
    def _setup_event_handlers(self):
        """Set up the Discord message event handlers."""
        @self.bot.event
        async def on_message(message: discord.Message):
            """Handle incoming messages, responding to replies, mentions, and DMs."""
            if message.author == self.bot.user:
                return

            if self._should_respond_to_message(message):
                await self._handle_user_message(message)
            else:
                await self.bot.process_commands(message)
    
    def _should_respond_to_message(self, message: discord.Message) -> bool:
        """
        Check if the bot should respond to the message.
        
        Args:
            message: The Discord message
            
        Returns:
            bool: True if the bot should respond, False otherwise
        """
        # Check if it's a reply to the bot
        is_bot_reply = (message.reference and 
                        message.reference.resolved and 
                        message.reference.resolved.author == self.bot.user)
                        
        # Check if it mentions the bot
        is_mention = self.bot.user.mentioned_in(message)
        
        # Check if it's a direct message
        is_dm = message.guild is None
        
        return is_bot_reply or is_mention or is_dm
    
    async def _handle_user_message(self, message: discord.Message):
        """
        Process an incoming user message that the bot should respond to.
        
        Args:
            message: The Discord message to process
        """
        user_id = message.author.id
        
        # Track tasks for this user
        if user_id not in user_tasks:
            user_tasks[user_id] = []
        
        # Create and track a new task for this message
        task = asyncio.create_task(self._process_user_message(message))
        user_tasks[user_id].append(task)
        task.add_done_callback(lambda t: user_tasks[user_id].remove(t) if user_id in user_tasks else None)
    
    async def _process_user_message(self, message: discord.Message):
        """
        Process the content of a user message and generate a response.
        
        Args:
            message: The Discord message to process
        """
        try:
            user_id = message.author.id
            
            # Check if user is blacklisted (skip for admins)
            if not await self.db.is_admin(user_id) and await self.db.is_user_blacklisted(user_id):
                await message.channel.send("You have been blacklisted from using this bot. Please contact the admin if you think this is a mistake.")
                return
            
            # Get history and model preferences first    
            history = await self.db.get_history(user_id)
            model = await self.db.get_user_model(user_id) or "gpt-4o"  # Default to gpt-4o if no model set

            # Handle PDF files
            if message.attachments:
                for attachment in message.attachments:
                    if attachment.filename.lower().endswith('.pdf'):
                        # Check if user is allowed to process PDFs
                        if model not in ["gpt-4o", "gpt-4o-mini"]:
                            await message.channel.send("PDF processing is only available with gpt-4o and gpt-4o-mini models. Please use /choose_model to select a supported model.")
                            return
                            
                        if not await self.db.is_admin(user_id) and not await self.db.is_user_whitelisted(user_id):
                            await message.channel.send("You are not authorized to use PDF processing. Please contact an admin for access.")
                            return
                            
                        # Get PDF content
                        pdf_content = await attachment.read()
                        user_prompt = message.content if message.content else "Analyze this PDF document and provide a summary."
                        
                        # Process PDF
                        await process_pdf(message, pdf_content, user_prompt, model, self.client)
                        return

            # Handle normal messages and non-PDF attachments
            content = []
            
            # Add message content if present
            if message.content:
                content.append({"type": "text", "text": message.content})

            # Process attachments
            if message.attachments:
                for attachment in message.attachments:
                    if any(attachment.filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                        content.append({
                            "type": "image_url", 
                            "image_url": {
                                "url": attachment.url,
                                "detail": "auto"
                            }
                        })
                    else:
                        content.append({"type": "text", "text": f"[Attachment: {attachment.filename}] - I can't process this type of file directly."})

            if not content:
                content.append({"type": "text", "text": "No content."})

            # Prepare current message
            current_message = {"role": "user", "content": content}
            
            try:
                # Let user know we're processing
                async with message.channel.typing():
                    # Process messages based on the model's capabilities
                    messages_for_api = []
                    
                    # For models that don't support system prompts
                    if model in ["o1-mini", "o1-preview"]:
                        # Convert system messages to user instructions
                        system_content = None
                        history_without_system = []
                        
                        # Extract system message content
                        for msg in history:
                            if msg.get('role') == 'system':
                                system_content = msg.get('content', '')
                            else:
                                history_without_system.append(msg)
                        
                        # Add the system content as a special user message at the beginning
                        if system_content:
                            history_without_system.insert(0, {"role": "user", "content": f"Instructions: {system_content}"})
                        
                        # Add current message and prepare for API
                        history_without_system.append(current_message)
                        messages_for_api = prepare_messages_for_api(history_without_system)
                    else:
                        # For models that support system prompts
                        from src.config.config import NORMAL_CHAT_PROMPT
                        
                        # Add system prompt if not present
                        if not any(msg.get('role') == 'system' for msg in history):
                            history.insert(0, {"role": "system", "content": NORMAL_CHAT_PROMPT})
                            
                        history.append(current_message)
                        messages_for_api = prepare_messages_for_api(history)
                    
                    # Determine which models should have tools available
                    # o1-mini and o1-preview do not support tools
                    use_tools = model in ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"]
                    
                    # Prepare API call parameters
                    api_params = {
                        "model": model,
                        "messages": messages_for_api,
                        "temperature": 0.3 if model in ["gpt-4o", "gpt-4o-mini"] else 1,
                        "top_p": 0.7 if model in ["gpt-4o", "gpt-4o-mini"] else 1
                    }
                    
                    # Add tools if using a supported model
                    if use_tools:
                        api_params["tools"] = get_tools_for_model()
                    
                    # Flag to track if image generation was used
                    image_generation_used = False
                    
                    # Make the initial API call
                    response = await self.client.chat.completions.create(**api_params)
                    
                    # Check if there are any tool calls to process
                    if use_tools and response.choices[0].finish_reason == "tool_calls":
                        # Process tools
                        tool_calls = response.choices[0].message.tool_calls
                        tool_messages = {}
                        
                        # Track which tools are being called
                        for tool_call in tool_calls:
                            if tool_call.function.name in self.tool_functions:
                                tool_messages[tool_call.function.name] = True
                                if tool_call.function.name == "generate_image":
                                    image_generation_used = True
                        
                        # Display appropriate messages based on which tools are being called
                        if tool_messages.get("google_search") or tool_messages.get("scrape_webpage"):
                            await message.channel.send("🔍 Researching information...")
                        
                        if tool_messages.get("code_interpreter"):
                            await message.channel.send("💻 Running code...")
                        
                        if tool_messages.get("generate_image"):
                            await message.channel.send("🎨 Generating images...")
                        
                        if not tool_messages:                        
                            await message.channel.send("🤔 Processing...")
                        
                        # Process any tool calls and get the updated messages
                        tool_calls_processed, updated_messages = await process_tool_calls(
                            self.client, 
                            response, 
                            messages_for_api, 
                            self.tool_functions
                        )
                        
                        # If tool calls were processed, make another API call with the updated messages
                        if tool_calls_processed:
                            response = await self.client.chat.completions.create(
                                model=model,
                                messages=updated_messages,
                                temperature=0.3 if model in ["gpt-4o", "gpt-4o-mini"] else 1
                            )
                
                reply = response.choices[0].message.content
                
                # Store the response in history for models that support it
                if model in ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"]:
                    if model in ["o1-mini", "o1-preview"]:
                        # For models without system prompt support, keep track separately
                        history_without_system.append({"role": "assistant", "content": reply})
                        # Sync back to regular history format by preserving system message
                        new_history = []
                        if system_content:
                            new_history.append({"role": "system", "content": system_content})
                        new_history.extend(history_without_system[1:])  # Skip the first "Instructions" message
                        await self.db.save_history(user_id, new_history)
                    else:
                        # For models with system prompt support, just append to regular history
                        history.append({"role": "assistant", "content": reply})
                        await self.db.save_history(user_id, history)
                
                # Check if there are any image URLs to send from the image generation tool
                image_urls = []
                for msg in locals().get('updated_messages', []):
                    if msg.get('role') == 'tool' and msg.get('name') == 'generate_image':
                        try:
                            content = msg.get('content', '')
                            if isinstance(content, str):
                                data = json.loads(content) if '{' in content else {}
                                if 'image_urls' in data:
                                    image_urls = data.get('image_urls', [])
                        except Exception as e:
                            logging.error(f"Error parsing image URLs: {str(e)}")
                
                # If image generation was used and we have image URLs, handle specially
                if image_generation_used and image_urls:
                    # Send the text response first (if it contains useful information besides just mentioning images)
                    text_to_exclude = ["here are the images", "i've generated", "generated for you", "as requested", "based on your prompt"]
                    
                    # Check if reply is just about the images or has other content
                    has_other_content = True
                    reply_lower = reply.lower()
                    
                    # Check if reply is primarily about the images
                    for phrase in text_to_exclude:
                        if phrase in reply_lower:
                            has_other_content = len(reply) > 300  # Only consider it "other content" if it's substantial
                    
                    # Only send text response if it has additional content
                    if has_other_content:
                        await send_response(message.channel, reply)
                    
                    # Download images from URLs and send as attachments
                    image_files = []
                    async with aiohttp.ClientSession() as session:
                        for image_url in image_urls:
                            try:
                                async with session.get(image_url) as resp:
                                    if resp.status == 200:
                                        image_files.append(await resp.read())
                            except Exception as e:
                                logging.error(f"Error downloading image: {str(e)}")
                    
                    # Send images as attachments
                    if image_files:
                        await message.channel.send(
                            "Generated images:",
                            files=[discord.File(io.BytesIO(img), filename=f"image_{i}.png") 
                                  for i, img in enumerate(image_files)]
                        )
                else:
                    # Normal response without image generation
                    await send_response(message.channel, reply)
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                logging.error(f"Error in message processing: {error_message}")
                await message.channel.send(error_message)
                
        except Exception as e:
            error_message = f"Error: {str(e)}"
            logging.error(f"Error in message handling: {error_message}")
            await message.channel.send(error_message)
            
    # Tool implementation functions 
    async def _google_search(self, args: Dict[str, Any]):
        """
        Execute Google search tool function.
        
        Args:
            args: Arguments containing query and optional num_results
        
        Returns:
            str: JSON string of search results
        """
        from src.utils.web_utils import google_custom_search
        
        query = args.get("query", "")
        num_results = min(int(args.get("num_results", 3)), 10)
        
        if not query:
            return json.dumps({"error": "No search query provided"})
        
        try:
            results = google_custom_search(query, num_results)
            return json.dumps(results)
        except Exception as e:
            return json.dumps({"error": str(e), "query": query})

    async def _scrape_webpage(self, args: Dict[str, Any]):
        """
        Execute web scraping tool function.
        
        Args:
            args: Arguments containing the URL to scrape
        
        Returns:
            str: Scraped content or error message
        """
        from src.utils.web_utils import scrape_web_content
        
        url = args.get("url", "")
        if not url:
            return "Error: No URL provided"
        
        content = scrape_web_content(url)
        return content

    async def _code_interpreter(self, args: Dict[str, Any]):
        """
        Execute code interpreter tool function.
        
        Args:
            args: Arguments containing code, language and optional input
        
        Returns:
            str: Code execution result
        """
        from src.utils.code_utils import execute_code
        
        code = args.get("code", "")
        language = args.get("language", "python")
        input_data = args.get("input", "")
        
        if not code:
            return "Error: No code provided"
        
        result = await execute_code(code, language, input_data=input_data)
        return result

    async def _generate_image(self, args: Dict[str, Any]):
        """
        Execute image generation tool function.
        
        Args:
            args: Arguments containing prompt and optional num_images
        
        Returns:
            str: JSON string with image information
        """
        prompt = args.get("prompt", "")
        num_images = min(int(args.get("num_images", 1)), 4)
        
        if not prompt:
            return json.dumps({"error": "No image prompt provided"})
        
        try:
            result = await self.image_generator.generate_image(prompt, num_images)
            if result["success"]:
                return json.dumps({
                    "message": f"Successfully generated {result['image_count']} images",
                    "image_urls": result["image_urls"],
                    "prompt": result["prompt"]
                })
            else:
                return json.dumps({"error": result.get("error", "Unknown error")})
        except Exception as e:
            return json.dumps({"error": str(e), "prompt": prompt})
    
    @staticmethod
    async def stop_user_tasks(user_id: int):
        """
        Stop all tasks for a specific user.
        
        Args:
            user_id: The Discord user ID
        """
        if user_id in user_tasks:
            for task in user_tasks[user_id]:
                task.cancel()
            user_tasks[user_id] = []