import discord
import asyncio
import json
import logging
import time
import functools
import concurrent.futures
from typing import Dict, Any, List
import io
import aiohttp
import os
import sys
import subprocess
import base64
import traceback
import tiktoken
from datetime import datetime, timedelta
from src.utils.openai_utils import process_tool_calls, prepare_messages_for_api, get_tools_for_model
from src.utils.pdf_utils import process_pdf, send_response
from src.utils.code_utils import extract_code_blocks
from src.utils.reminder_utils import ReminderManager
from src.config.config import PDF_ALLOWED_MODELS, MODEL_TOKEN_LIMITS, DEFAULT_TOKEN_LIMIT, DEFAULT_MODEL

# Global task and rate limiting tracking
user_tasks = {}
user_last_request = {}
RATE_LIMIT_WINDOW = 5  # seconds
MAX_REQUESTS = 3  # max requests per window

# File extensions that should be treated as text files
TEXT_FILE_EXTENSIONS = [
    '.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm', '.css', 
    '.js', '.py', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.php',
    '.rb', '.pl', '.sh', '.bat', '.ps1', '.sql', '.yaml', '.yml',
    '.ini', '.cfg', '.conf', '.log', '.ts', '.jsx', '.tsx', '.vue', 
    '.go', '.rs', '.swift', '.kt', '.kts', '.dart', '.lua'
]

# File extensions for data files
DATA_FILE_EXTENSIONS = ['.csv', '.xlsx', '.xls']

# File extensions for image files (should never be processed as data)
IMAGE_FILE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg', '.tiff', '.ico']

# Storage for user data files and charts
user_data_files = {}
user_charts = {}

# Try to import data analysis libraries early
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    if 'DISPLAY' not in os.environ:
        matplotlib.use('Agg')  # Use non-interactive backend
    PANDAS_AVAILABLE = True
    logging.info(f"Successfully imported pandas {pd.__version__} and related libraries")
except ImportError as e:
    PANDAS_AVAILABLE = False
    logging.warning(f"Data analysis libraries not available: {str(e)}")

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
        self.aiohttp_session = None
        
        # Initialize reminder manager
        self.reminder_manager = ReminderManager(bot, db_handler)
        
        # Tool mapping for API integration
        self.tool_mapping = {
            "google_search": self._google_search,
            "scrape_webpage": self._scrape_webpage,
            "execute_python_code": self._execute_python_code,
            "analyze_data_file": self._analyze_data_file,
            "generate_image": self._generate_image,
            "edit_image": self._edit_image,
            "set_reminder": self._set_reminder,
            "get_reminders": self._get_reminders,
            "enhance_prompt": self._enhance_prompt,
            "image_to_text": self._image_to_text,
            "upscale_image": self._upscale_image,
            "photo_maker": self._photo_maker,
            "generate_image_with_refiner": self._generate_image_with_refiner
        }
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        
        # Create session for HTTP requests
        asyncio.create_task(self._setup_aiohttp_session())
        
        # Register message event handlers
        self._setup_event_handlers()
        
        # Start reminder manager
        self.reminder_manager.start()
        
        # Start chart cleanup task
        self.chart_cleanup_task = asyncio.create_task(self._run_chart_cleanup())
        
        # Start data file cleanup task
        self.file_cleanup_task = asyncio.create_task(self._run_file_cleanup())
        
        # Install required packages if not available
        if not PANDAS_AVAILABLE:
            self._install_data_packages()
        
        # Initialize tiktoken encoder for token counting (using o200k_base for all models)
        self.token_encoder = tiktoken.get_encoding("o200k_base")
    
    def _find_user_id_from_current_task(self):
        """
        Utility method to find user_id from the current asyncio task.
        
        Returns:
            str or None: The user_id if found, None otherwise
        """
        current_task = asyncio.current_task()
        if not current_task:
            return None
            
        for user_id, tasks in user_tasks.items():
            if current_task in tasks:
                return user_id
        return None
    
    def _get_discord_message_from_current_task(self):
        """
        Utility method to get the Discord message from the current asyncio task.
        
        Returns:
            discord.Message or None: The Discord message if found, None otherwise
        """
        current_task = asyncio.current_task()
        if not current_task:
            return None
            
        for user_id, tasks in user_tasks.items():
            if current_task in tasks:
                task_info = tasks[current_task]
                return task_info.get('message')
        return None
            
    def _install_data_packages(self):
        """Install required data analysis packages if not available"""
        try:
            logging.info("Attempting to install data analysis packages...")
            packages = ["pandas", "numpy", "matplotlib", "seaborn", "openpyxl"]
            for package in packages:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    logging.info(f"Successfully installed {package}")
                except Exception as e:
                    logging.error(f"Failed to install {package}: {str(e)}")
        except Exception as e:
            logging.error(f"Error installing packages: {str(e)}")
    
    async def _setup_aiohttp_session(self):
        """Create a reusable aiohttp session for better performance"""
        if self.aiohttp_session is None or self.aiohttp_session.closed:
            self.aiohttp_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=240),
                connector=aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)
            )
        
    def _setup_event_handlers(self):
        """Set up the Discord message event handlers."""
        @self.bot.event
        async def on_message(message: discord.Message):
            """Handle incoming messages, responding to replies, mentions, and DMs."""
            if message.author == self.bot.user:
                return
                
            if self._should_respond_to_message(message):
                # Check rate limiting before processing
                if await self._check_rate_limit(message):
                    await self._handle_user_message(message)
            else:
                await self.bot.process_commands(message)
    
    # Note: _analyze_data function removed - replaced by execute_python_code and analyze_data_file
    
    async def _execute_python_code(self, args: Dict[str, Any]):
        """Handle general Python code execution functionality"""
        try:
            # Find user_id from current task context
            user_id = args.get("user_id")
            if not user_id:
                user_id = self._find_user_id_from_current_task()
            
            # Add file context if user has uploaded data files
            if user_id and user_id in user_data_files:
                file_info = user_data_files[user_id]
                file_context = f"\n\n# Data file available: {file_info['filename']}\n"
                file_context += f"# File path: {file_info['file_path']}\n"
                file_context += f"# You can access this file using: pd.read_csv('{file_info['file_path']}') or similar\n\n"
                
                # Prepend file context to the code
                original_code = args.get("code", "")
                args["code"] = file_context + original_code
                
                logging.info(f"Added file context to Python execution for user {user_id}")
            
            # Import and call Python executor
            from src.utils.python_executor import execute_python_code
            execute_result = await execute_python_code(args)
            
            # If there are visualizations, handle them
            if execute_result and execute_result.get("visualizations"):
                discord_message = self._get_discord_message_from_current_task()
                
                for i, viz_path in enumerate(execute_result["visualizations"]):
                    try:
                        with open(viz_path, 'rb') as f:
                            img_data = f.read()
                                
                        if discord_message:
                            # Upload chart to Discord and get URL
                            chart_url = await self._upload_and_get_chart_url(
                                img_data, 
                                f"chart_{i+1}.png",
                                discord_message.channel
                            )
                            
                            if chart_url:
                                # Store the URL in history
                                execute_result.setdefault("chart_urls", []).append(chart_url)
                                
                                # Send the chart with description
                                await discord_message.channel.send(
                                    "📊 Generated visualization:",
                                    file=discord.File(io.BytesIO(img_data), filename=f"chart_{i+1}.png")
                                )
                            
                    except Exception as e:
                        logging.error(f"Error handling visualization: {str(e)}")
                        traceback.print_exc()

            # Update history with chart URLs if available
            if execute_result and execute_result.get("chart_urls"):
                content = []
                if execute_result.get("output"):
                    content.append({
                        "type": "text",
                        "text": execute_result["output"]
                    })
                
                # Add each chart URL to content
                for chart_url in execute_result["chart_urls"]:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": chart_url},
                        "timestamp": datetime.now().isoformat()
                    })
                
                return {
                    "role": "assistant",
                    "content": content
                }
            else:
                return {
                    "role": "assistant", 
                    "content": execute_result.get("output", "No output generated") if execute_result else "No output generated"
                }

        except Exception as e:
            error_msg = f"Error in Python code execution: {str(e)}"
            logging.error(error_msg)
            traceback.print_exc()
            return {"role": "assistant", "content": error_msg}
    
    async def _analyze_data_file(self, args: Dict[str, Any]) -> str:
        """
        Analyze a data file using the new data analyzer module.
        
        Args:
            args: Dictionary containing file_path and optional parameters
            
        Returns:
            JSON string with analysis results
        """
        try:
            # Find real user_id from current task if not provided
            user_id = args.get("user_id", "default")
            if user_id == "default":
                user_id = self._find_user_id_from_current_task()
            
            # Add user_id to args for the data analyzer
            args["user_id"] = user_id
            
            # Import and call data analyzer
            from src.utils.data_analyzer import analyze_data_file
            result = await analyze_data_file(args)
            
            # If there are visualizations, handle them for Discord
            if result and result.get("visualizations"):
                discord_message = self._get_discord_message_from_current_task()
                
                for i, viz_path in enumerate(result["visualizations"]):
                    try:
                        with open(viz_path, 'rb') as f:
                            img_data = f.read()
                                
                        if discord_message:
                            # Upload chart to Discord and get URL
                            chart_url = await self._upload_and_get_chart_url(
                                img_data, 
                                f"analysis_chart_{i+1}.png",
                                discord_message.channel
                            )
                            
                            if chart_url:
                                # Store the URL in history
                                result.setdefault("chart_urls", []).append(chart_url)
                                
                                # Send the chart with description
                                await discord_message.channel.send(
                                    "📊 Data analysis visualization:",
                                    file=discord.File(io.BytesIO(img_data), filename=f"analysis_chart_{i+1}.png")
                                )
                            
                    except Exception as e:
                        logging.error(f"Error handling visualization: {str(e)}")
                        traceback.print_exc()

            # Format response for the AI model
            if result and result.get("success"):
                content = []
                if result.get("output"):
                    content.append({
                        "type": "text",
                        "text": result["output"]
                    })
                
                # Add chart URLs if available
                if result.get("chart_urls"):
                    for chart_url in result["chart_urls"]:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": chart_url},
                            "timestamp": datetime.now().isoformat()
                        })
                
                return {
                    "role": "assistant",
                    "content": content if content else result.get("output", "Analysis completed")
                }
            else:
                return {
                    "role": "assistant",
                    "content": result.get("error", "Analysis failed") if result else "Analysis failed"
                }

        except Exception as e:
            error_msg = f"Error in data analysis: {str(e)}"
            logging.error(error_msg)
            traceback.print_exc()
            return {"role": "assistant", "content": error_msg}
    
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
    
    async def _check_rate_limit(self, message: discord.Message) -> bool:
        """
        Check if the user is within rate limits.
        
        Args:
            message: The Discord message
            
        Returns:
            bool: True if within rate limits, False otherwise
        """
        user_id = message.author.id
        current_time = time.time()
        
        # Check if this is an admin (bypass rate limiting)
        if await self.db.is_admin(user_id):
            return True
        
        # Initialize if first message from user
        if user_id not in user_last_request:
            user_last_request[user_id] = []
        
        # Clean expired timestamps
        user_last_request[user_id] = [
            timestamp for timestamp in user_last_request[user_id] 
            if current_time - timestamp < RATE_LIMIT_WINDOW
        ]
        
        # Check if user has hit rate limit
        if len(user_last_request[user_id]) >= MAX_REQUESTS:
            await message.channel.send(
                f"⚠️ You're sending requests too quickly. Please wait a moment before trying again."
            )
            return False
        
        # Add current request timestamp
        user_last_request[user_id].append(current_time)
        return True
    
    async def _handle_user_message(self, message: discord.Message):
        """
        Process an incoming user message that the bot should respond to.
        
        Args:
            message: The Discord message to process
        """
        user_id = message.author.id
        
        # Track tasks for this user
        if user_id not in user_tasks:
            user_tasks[user_id] = {}
        
        # Create and track a new task for this message
        task = asyncio.create_task(self._process_user_message(message))
        
        # Store the message object with the task for future reference
        user_tasks[user_id][task] = {'message': message}
        
        # Use done callbacks to clean up and handle errors
        def task_done_callback(task):
            # Remove task from tracking dictionary
            if user_id in user_tasks and task in user_tasks[user_id]:
                del user_tasks[user_id][task]
            
            # Check for exceptions that weren't handled
            if task.done() and not task.cancelled():
                try:
                    task.result()
                except Exception as e:
                    logging.error(f"Unhandled task exception: {str(e)}")
        
        # Add the callback
        task.add_done_callback(task_done_callback)
    
    async def _download_and_save_data_file(self, attachment, user_id):
        """
        Download and save a data file attachment for future use
        
        Args:
            attachment: The Discord file attachment
            user_id: User ID for tracking
            
        Returns:
            Dict with file info and path
        """
        try:
            # Get file contents and determine file type
            file_extension = os.path.splitext(attachment.filename)[1].lower()
            file_bytes = await attachment.read()
            
            # Save file to local storage with timestamp
            from src.utils.code_utils import DATA_FILES_DIR
            temp_file_path = os.path.join(DATA_FILES_DIR, f"data_{user_id}_{int(time.time())}{file_extension}")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
            
            # Save file
            with open(temp_file_path, "wb") as f:
                f.write(file_bytes)
                
            # Store the data file in user_data_files for future reference
            file_info = {
                "bytes": file_bytes,
                "filename": attachment.filename,
                "file_path": temp_file_path,
                "timestamp": datetime.now()
            }
            user_data_files[user_id] = file_info
            
            logging.info(f"Downloaded and saved data file: {temp_file_path}")
            return {"success": True, "file_info": file_info}
            
        except Exception as e:
            error_msg = f"Error downloading data file: {str(e)}"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _detect_user_intent(self, message_content):
        """
        Detect whether user wants data analysis or general programming
        
        Args:
            message_content: The user's message content
            
        Returns:
            str: 'data_analysis' or 'general_programming'
        """
        if not message_content:
            return 'data_analysis'  # Default for file uploads without message
            
        content_lower = message_content.lower()
        
        # Data analysis keywords
        analysis_keywords = [
            'analyze', 'analysis', 'visualize', 'plot', 'graph', 'chart', 'statistics', 'stats',
            'correlation', 'distribution', 'histogram', 'summary', 'explore', 'insights',
            'trends', 'patterns', 'data', 'dataset', 'describe', 'overview', 'breakdown',
            'create visualizations', 'show', 'display'
        ]
        
        # Programming keywords
        programming_keywords = [
            'code', 'function', 'algorithm', 'calculate', 'compute', 'programming', 'script',
            'implement', 'process', 'transform', 'clean', 'filter', 'merge', 'join',
            'custom', 'specific', 'particular', 'write code', 'create function'
        ]
        
        analysis_score = sum(1 for keyword in analysis_keywords if keyword in content_lower)
        programming_score = sum(1 for keyword in programming_keywords if keyword in content_lower)
        
        # If programming intent is clearly stronger, use general programming
        if programming_score > analysis_score:
            return 'general_programming'
        else:
            return 'data_analysis'
    
    async def _handle_data_file(self, attachment, message, user_id, history, model, start_time):
        """
        Handle a data file attachment by downloading it and determining appropriate tool
        
        Args:
            attachment: The Discord file attachment
            message: The Discord message
            user_id: User ID for tracking
            history: Conversation history
            model: OpenAI model to use
            start_time: Timestamp when processing started
            
        Returns:
            Dict with processing results
        """
        try:
            # First, download and save the file
            download_result = await self._download_and_save_data_file(attachment, user_id)
            
            if not download_result["success"]:
                await message.channel.send(f"❌ {download_result['error']}")
                return download_result
            
            file_info = download_result["file_info"]
            file_path = file_info["file_path"]
            
            # Safety check: Ensure this is not an image file
            file_ext = os.path.splitext(attachment.filename)[1].lower()
            if file_ext in IMAGE_FILE_EXTENSIONS:
                await message.channel.send(
                    f"🖼️ **Image File Detected**: {attachment.filename}\n"
                    f"Images are handled directly by the AI model for visual analysis.\n"
                    f"Your image has been sent to the AI for processing."
                )
                return {"success": True, "message": "Image processed directly by AI model"}
            
            # Extract query from message if any
            content = message.content.strip()
            query = content if content else "Analyze this data file and create relevant visualizations"
            
            # Detect user intent
            intent = self._detect_user_intent(content)
            
            if intent == 'data_analysis':
                # Use the specialized data analysis tool
                await message.channel.send("📊 Analyzing data file with specialized data analysis tool...")
                
                # Determine analysis type based on query
                analysis_type = "comprehensive"  # Default
                if any(word in query.lower() for word in ['correlation', 'correlate', 'relationship']):
                    analysis_type = "correlation"
                elif any(word in query.lower() for word in ['distribution', 'histogram', 'spread']):
                    analysis_type = "distribution"
                elif any(word in query.lower() for word in ['summary', 'overview', 'basic']):
                    analysis_type = "summary"
                
                # Call the data analysis tool directly
                analysis_args = {
                    "file_path": file_path,
                    "analysis_type": analysis_type,
                    "custom_analysis": query,
                    "user_id": user_id
                }
                
                result = await self._analyze_data_file(analysis_args)
                
                # The tool already handles Discord integration, so we just return the result
                return result
                
            else:
                # For general programming, just inform the user that the file is ready
                await message.channel.send(
                    f"📁 **File Downloaded**: {attachment.filename}\n"
                    f"File saved and ready for use in Python code.\n"
                    f"You can now ask me to write Python code to process this data file."
                )
                
                # Add file info to the conversation for context
                file_context = f"\n\n[Data file uploaded: {attachment.filename} - Available at path: {file_path}]"
                
                # Add context to the current conversation
                if len(history) > 0 and history[-1]["role"] == "user":
                    if isinstance(history[-1]["content"], list):
                        history[-1]["content"].append({
                            "type": "text", 
                            "text": file_context
                        })
                    else:
                        history[-1]["content"] += file_context
                
                # Save updated history
                await self.db.save_history(user_id, history)
                
                return {
                    "success": True, 
                    "message": "File ready for Python programming",
                    "file_path": file_path,
                    "intent": intent
                }
                
        except Exception as e:
            error_msg = f"Error handling data file: {str(e)}"
            logging.error(error_msg)
            traceback.print_exc()
            await message.channel.send(f"❌ {error_msg}")
            return {"error": error_msg}

    async def _upload_and_get_chart_url(self, image_data, filename, channel):
        """
        Upload an image to Discord and get its URL for later reference
        
        Args:
            image_data: Binary image data
            filename: Filename for the image
            channel: Discord channel to upload to
            
        Returns:
            str: URL of the uploaded image or None on failure
        """
        try:
            file = discord.File(io.BytesIO(image_data), filename=filename)
            message = await channel.send(file=file)
            if message.attachments and len(message.attachments) > 0:
                return message.attachments[0].url
            return None
        except Exception as e:
            logging.error(f"Error uploading chart: {str(e)}")
            return None

    async def _process_user_message(self, message: discord.Message):
        """
        Process the content of a user message and generate a response.
        
        Args:
            message: The Discord message to process
        """
        try:
            user_id = message.author.id
            start_time = time.time()  # Track processing time
            
            # Typing indicator to show the bot is working
            async with message.channel.typing():
                # Check if user is blacklisted (skip for admins)
                if not await self.db.is_admin(user_id) and await self.db.is_user_blacklisted(user_id):
                    await message.channel.send("You have been blacklisted from using this bot. Please contact the admin if you think this is a mistake.")
                    return
                
                # Get history and model preferences first    
                history = await self.db.get_history(user_id)
                model = await self.db.get_user_model(user_id) or DEFAULT_MODEL  # Default to configured default model if no model set
                
                # Handle PDF files
                if message.attachments:
                    for attachment in message.attachments:
                        if attachment.filename.lower().endswith('.pdf'):
                            # Check if user is allowed to process PDFs
                            if model not in PDF_ALLOWED_MODELS:
                                allowed_models = ", ".join(PDF_ALLOWED_MODELS)
                                await message.channel.send(f"PDF processing is only available with these models: {allowed_models}. Please use /choose_model to select a supported model.")
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
                        
                        # Handle data files (CSV, Excel)
                        elif any(attachment.filename.lower().endswith(ext) for ext in DATA_FILE_EXTENSIONS):
                            await self._handle_data_file(attachment, message, user_id, history, model, start_time)
                            return
                            
                # Handle normal messages and non-PDF attachments
                content = []
                extracted_text_contents = []
                
                # Add message content if present
                if message.content:
                    content.append({"type": "text", "text": message.content})
                    
                # Process attachments
                if message.attachments:
                    for attachment in message.attachments:
                        if any(attachment.filename.lower().endswith(ext) for ext in TEXT_FILE_EXTENSIONS):
                            # Process text-based file attachments
                            try:
                                file_bytes = await attachment.read()
                                try:
                                    # Try to decode as UTF-8 first
                                    file_content = file_bytes.decode('utf-8')
                                except UnicodeDecodeError:
                                    # Try UTF-8 with error replacement
                                    try:
                                        file_content = file_bytes.decode('utf-8', errors='replace')
                                    except:
                                        file_content = f"[Unable to decode file content: {attachment.filename}]"
                                
                                # Add formatted text to extracted contents
                                extracted_text = f"\n\n--- Content of {attachment.filename} ---\n{file_content}\n--- End of {attachment.filename} ---\n\n"
                                extracted_text_contents.append(extracted_text)
                                
                                # Add a reference in the content
                                content.append({"type": "text", "text": f"[Attached file: {attachment.filename}]"})
                                
                                logging.info(f"Extracted text from {attachment.filename} ({len(file_content)} chars)")
                                
                            except Exception as e:
                                error_msg = f"Error processing text file {attachment.filename}: {str(e)}"
                                logging.error(error_msg)
                                content.append({"type": "text", "text": f"[Error processing {attachment.filename}: {str(e)}]"})
                                
                        elif any(attachment.filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                            content.append({
                                "type": "image_url", 
                                "image_url": {
                                    "url": attachment.url,
                                    "detail": "high"
                                },
                                "timestamp": datetime.now().isoformat()  # Add timestamp to track image expiration
                            })
                        else:
                            content.append({"type": "text", "text": f"[Attachment: {attachment.filename}] - I can't process this type of file directly."})
                
                # If we have extracted text contents, append them to the user's message
                if extracted_text_contents:
                    # Add the file content(s) after the user's message
                    for text in extracted_text_contents:
                        content.append({"type": "text", "text": text})
                
                if not content:
                    content.append({"type": "text", "text": "No content."})
                    
                # Prepare current message
                current_message = {"role": "user", "content": content}
                
                # Pass the message to the text processing function with the start_time
                await self._process_text_message(message, user_id, current_message, model, history, start_time)
                
        except asyncio.CancelledError:
            # Re-raise cancellation
            raise
        except Exception as e:
            error_message = f"Error: {str(e)}"
            logging.error(f"Error in message handling: {error_message}")
            try:
                await message.channel.send(error_message)
            except:
                pass
    
    async def _process_text_message(self, message, user_id, current_message, model, history, start_time):
        """
        Process text messages and generate AI responses.
        
        Args:
            message: Original Discord message
            user_id: User ID
            current_message: Current message (string or dict)
            model: AI model to use
            history: Conversation history
            start_time: Time when processing started for tracking duration
        """
        try:
            # Convert string messages to message format if needed
            if isinstance(current_message, str):
                current_message = {"role": "user", "content": current_message}
            
            # Process messages based on the model's capabilities
            messages_for_api = []
            
            # For models that don't support system prompts
            if model in ["openai/o1-mini", "openai/o1-preview"]:
                # Convert system messages to user instructions
                system_content = None
                history_without_system = []
                
                # Extract system message content
                for msg in history:
                    if (msg.get('role') == 'system'):
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
            
            # Proactively trim history to avoid context overload while preserving system prompt
            current_tokens = self._count_tokens(messages_for_api)
            token_limit = MODEL_TOKEN_LIMITS.get(model, DEFAULT_TOKEN_LIMIT)
            max_tokens = int(token_limit * 0.8)  # Use 80% of limit to leave room for response
            
            if current_tokens > max_tokens:
                logging.info(f"Proactively trimming history: {current_tokens} tokens > {max_tokens} limit for {model}")
                
                if model in ["openai/o1-mini", "openai/o1-preview"]:
                    # For o1 models, trim the history without system prompt
                    trimmed_history_without_system = self._trim_history_to_token_limit(history_without_system, model, max_tokens)
                    messages_for_api = prepare_messages_for_api(trimmed_history_without_system)
                    
                    # Update the history tracking
                    history_without_system = trimmed_history_without_system
                else:
                    # For regular models, trim the full history (preserving system prompt)
                    trimmed_history = self._trim_history_to_token_limit(history, model, max_tokens)
                    messages_for_api = prepare_messages_for_api(trimmed_history)
                    
                    # Update the history tracking
                    history = trimmed_history
                
                # Save the trimmed history immediately to keep it in sync
                if model in ["openai/o1-mini", "openai/o1-preview"]:
                    new_history = []
                    if system_content:
                        new_history.append({"role": "system", "content": system_content})
                    new_history.extend(history_without_system[1:])  # Skip the "Instructions" message
                    await self.db.save_history(user_id, new_history)
                else:
                    await self.db.save_history(user_id, history)
                
                final_tokens = self._count_tokens(messages_for_api)
                logging.info(f"History trimmed from {current_tokens} to {final_tokens} tokens")
            
            # Determine which models should have tools available
            # openai/o1-mini and openai/o1-preview do not support tools
            use_tools = model in ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-5", "openai/gpt-5-nano", "openai/gpt-5-mini", "openai/gpt-5-chat", "openai/o1", "openai/o3-mini", "openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4.1-nano", "openai/o3", "openai/o4-mini"]
            
            # Prepare API call parameters
            api_params = {
                "model": model,
                "messages": messages_for_api,
                "timeout": 240  # Increased timeout for better response handling
            }
            
            # Add temperature and top_p only for models that support them (exclude GPT-5 family)
            if model in ["openai/gpt-4o", "openai/gpt-4o-mini"]:
                api_params["temperature"] = 0.3
                api_params["top_p"] = 0.7
            elif model not in ["openai/gpt-5", "openai/gpt-5-nano", "openai/gpt-5-mini", "openai/gpt-5-chat"]:
                # For other models (not GPT-4o family and not GPT-5 family)
                api_params["temperature"] = 1
                api_params["top_p"] = 1
            
            # Add tools if using a supported model
            if use_tools:
                api_params["tools"] = get_tools_for_model()
            
            # Initialize variables to track tool responses
            image_generation_used = False
            chart_id = None
            image_urls = []  # Will store unique image URLs
            
            # Make the initial API call
            try:
                response = await self.client.chat.completions.create(**api_params)
            except Exception as e:
                # Handle 413 Request Entity Too Large error with a user-friendly message
                if "413" in str(e) or "tokens_limit_reached" in str(e) or "Request body too large" in str(e):
                    await message.channel.send(
                        f"❌ **Request too large for {model}**\n"
                        f"Your conversation history or message is too large for this model.\n"
                        f"Try:\n"
                        f"• Using `/reset` to start fresh\n"
                        f"• Using a model with higher token limits\n"
                        f"• Reducing the size of your current message\n"
                        f"• Breaking up large files into smaller pieces"
                    )
                    return
                else:
                    # Re-raise other errors
                    raise e
            
            # Process tool calls if any
            updated_messages = None
            if use_tools and response.choices[0].finish_reason == "tool_calls":
                # Process tools
                tool_calls = response.choices[0].message.tool_calls
                tool_messages = {}
                
                # Track which tools are being called
                for tool_call in tool_calls:
                    if tool_call.function.name in self.tool_mapping:
                        tool_messages[tool_call.function.name] = True
                        if tool_call.function.name == "generate_image":
                            image_generation_used = True
                        elif tool_call.function.name == "edit_image":
                            # Display appropriate message for image editing
                            await message.channel.send("🖌️ Editing image...")
                
                # Display appropriate messages based on which tools are being called
                if tool_messages.get("google_search") or tool_messages.get("scrape_webpage"):
                    await message.channel.send("🔍 Researching information...")
                
                if tool_messages.get("execute_python_code") or tool_messages.get("analyze_data_file"):
                    await message.channel.send("💻 Running code...")
                
                if tool_messages.get("generate_image"):
                    await message.channel.send("🎨 Generating images...")
                    
                if tool_messages.get("set_reminder") or tool_messages.get("get_reminders"):
                    await message.channel.send("📅 Processing reminders...")
                
                if not tool_messages:                        
                    await message.channel.send("🤔 Processing...")
                
                # Process any tool calls and get the updated messages
                tool_calls_processed, updated_messages = await process_tool_calls(
                    self.client, 
                    response, 
                    messages_for_api, 
                    self.tool_mapping
                )
                
                # Process tool responses to extract important data (images, charts)
                if updated_messages:
                    # Look for image generation and code interpreter tool responses
                    for msg in updated_messages:
                        if msg.get('role') == 'tool' and msg.get('name') == 'generate_image':
                            try:
                                tool_result = json.loads(msg.get('content', '{}'))
                                if tool_result.get('image_urls'):
                                    image_urls.extend(tool_result['image_urls'])
                            except:
                                pass
                        
                        elif msg.get('role') == 'tool' and msg.get('name') == 'edit_image':
                            try:
                                tool_result = json.loads(msg.get('content', '{}'))
                                if tool_result.get('image_url'):
                                    image_urls.append(tool_result['image_url'])
                            except:
                                pass
                        
                        elif msg.get('role') == 'tool' and msg.get('name') in ['execute_python_code', 'analyze_data_file']:
                            try:
                                tool_result = json.loads(msg.get('content', '{}'))
                                if tool_result.get('chart_id'):
                                    chart_id = tool_result['chart_id']
                            except:
                                pass
                
                # If tool calls were processed, make another API call with the updated messages
                if tool_calls_processed and updated_messages:
                    # Prepare API parameters for follow-up call
                    follow_up_params = {
                        "model": model,
                        "messages": updated_messages,
                        "timeout": 240
                    }
                    
                    # Add temperature only for models that support it (exclude GPT-5 family)
                    if model in ["openai/gpt-4o", "openai/gpt-4o-mini"]:
                        follow_up_params["temperature"] = 0.3
                    elif model not in ["openai/gpt-5", "openai/gpt-5-nano", "openai/gpt-5-mini", "openai/gpt-5-chat"]:
                        follow_up_params["temperature"] = 1
                    
                    response = await self.client.chat.completions.create(**follow_up_params)
            
            reply = response.choices[0].message.content
            
            # Add image URLs to assistant content if any were found
            has_images = len(image_urls) > 0
            content_with_images = []
            
            if has_images:
                # If we have image URLs, create a content array with text and images
                content_with_images.append({"type": "text", "text": reply})
                
                # Add each image URL to the content
                for img_url in image_urls:
                    content_with_images.append({
                        "type": "image_url",
                        "image_url": {"url": img_url},
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Store the response in history for models that support it
            if model in ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-5", "openai/gpt-5-nano", "openai/gpt-5-mini", "openai/gpt-5-chat", "openai/o1", "openai/o1-mini", "openai/o3-mini", "openai/gpt-4.1", "openai/gpt-4.1-nano", "openai/gpt-4.1-mini", "openai/o3", "openai/o4-mini", "openai/o1-preview"]:
                if model in ["openai/o1-mini", "openai/o1-preview"]:
                    # For models without system prompt support, keep track separately
                    if has_images:
                        history_without_system.append({"role": "assistant", "content": content_with_images})
                    else:
                        history_without_system.append({"role": "assistant", "content": reply})
                    
                    # Sync back to regular history format by preserving system message
                    new_history = []
                    if system_content:
                        new_history.append({"role": "system", "content": system_content})
                    new_history.extend(history_without_system[1:])  # Skip the first "Instructions" message
                    
                    # Only keep a reasonable amount of history
                    if len(new_history) > 20:
                        new_history = new_history[:1] + new_history[-19:]  # Keep system prompt + last 19 messages
                        
                    await self.db.save_history(user_id, new_history)
                else:
                    # For models with system prompt support, just append to regular history
                    if has_images:
                        history.append({"role": "assistant", "content": content_with_images})
                    else:
                        history.append({"role": "assistant", "content": reply})
                    
                    # Only keep a reasonable amount of history
                    if len(history) > 20:
                        history = history[:1] + history[-19:]  # Keep system prompt + last 19 messages
                        
                    await self.db.save_history(user_id, history)
            
            # Send the response text
            await send_response(message.channel, reply)
            
            # Handle charts from code interpreter if present
            if chart_id and chart_id in user_charts:
                try:
                    chart_data = user_charts[chart_id]["image"]
                    chart_filename = f"chart_{chart_id}.png"
                    
                    # Send the chart to Discord and get the URL
                    chart_url = await self._upload_and_get_chart_url(
                        chart_data, 
                        chart_filename,
                        message.channel
                    )
                    
                    if chart_url:
                        logging.info(f"Chart uploaded successfully: {chart_url}")
                        
                except Exception as e:
                    logging.error(f"Error handling chart: {str(e)}")
            
            # Log processing time for performance monitoring
            processing_time = time.time() - start_time
            logging.info(f"Message processed in {processing_time:.2f} seconds (User: {user_id}, Model: {model})")
            
        except asyncio.CancelledError:
            # Handle cancellation cleanly
            logging.info(f"Task for user {user_id} was cancelled")
            raise
        except Exception as e:
            error_message = f"Error: {str(e)}"
            logging.error(f"Error in message processing: {error_message}")
            await message.channel.send(error_message)

    # Tool implementation methods
    async def _google_search(self, args: Dict[str, Any]):
        """Perform a Google search"""
        try:
            from src.utils.web_utils import google_search
            result = await google_search(args)
            return result
        except Exception as e:
            logging.error(f"Error in Google search: {str(e)}")
            return json.dumps({"error": f"Google search failed: {str(e)}"})
    
    async def _scrape_webpage(self, args: Dict[str, Any]):
        """Scrape a webpage"""
        try:
            from src.utils.web_utils import scrape_webpage
            result = await scrape_webpage(args)
            return result
        except Exception as e:
            logging.error(f"Error in webpage scraping: {str(e)}")
            return json.dumps({"error": f"Webpage scraping failed: {str(e)}"})
    
    async def _generate_image(self, args: Dict[str, Any]):
        """Generate an image"""
        try:
            result = await self.image_generator.generate_image(args)
            return result
        except Exception as e:
            logging.error(f"Error in image generation: {str(e)}")
            return json.dumps({"error": f"Image generation failed: {str(e)}"})
    
    async def _edit_image(self, args: Dict[str, Any]):
        """Edit an image"""
        try:
            result = await self.image_generator.edit_image(args)
            return result
        except Exception as e:
            logging.error(f"Error in image editing: {str(e)}")
            return json.dumps({"error": f"Image editing failed: {str(e)}"})
    
    async def _set_reminder(self, args: Dict[str, Any]) -> str:
        """
        Set a reminder for a user.
        
        Args:
            args: Reminder information
            
        Returns:
            JSON string with reminder info
        """
        content = args.get("content", "")
        time_str = args.get("time", "")
        
        if not content:
            return json.dumps({"error": "No reminder content provided"})
            
        if not time_str:
            return json.dumps({"error": "No reminder time provided"})
            
        try:
            # Find user_id from current task
            user_id = self._find_user_id_from_current_task()
            
            if not user_id:
                return json.dumps({"error": "Could not identify user for reminder"})
                
            # Parse time using user's timezone if available
            remind_at = await self.reminder_manager.parse_time(time_str, user_id)
            
            if not remind_at:
                return json.dumps({"error": "Could not parse reminder time"})
                
            # Save reminder
            reminder = await self.reminder_manager.add_reminder(user_id, content, remind_at)
            
            # Get timezone info for the response
            user_tz = await self.reminder_manager.detect_user_timezone(user_id)
            
            return json.dumps({
                "success": True,
                "content": content,
                "time": remind_at.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "timezone": user_tz,
                "reminder_id": str(reminder["_id"])
            })
            
        except Exception as e:
            logging.error(f"Error setting reminder: {str(e)}")
            return json.dumps({"error": f"Error setting reminder: {str(e)}"})
    
    async def _get_reminders(self, args: Dict[str, Any]) -> str:
        """
        Get a user's reminders.
        
        Args:
            args: Not used
            
        Returns:
            JSON string with list of reminders
        """
        try:
            # Find user_id from current task
            user_id = self._find_user_id_from_current_task()
            
            if not user_id:
                return json.dumps({"error": "Could not identify user for reminders"})
                
            # Get reminders list
            reminders = await self.reminder_manager.get_user_reminders(user_id)
            
            # Format reminders
            formatted_reminders = []
            for reminder in reminders:
                formatted_reminders.append({
                    "id": str(reminder["_id"]),
                    "content": reminder["content"],
                    "time": reminder["remind_at"].strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "created": reminder.get("created_at", "Unknown")
                })
                
            return json.dumps({
                "success": True,
                "reminders": formatted_reminders,
                "count": len(formatted_reminders)
            })
            
        except Exception as e:
            logging.error(f"Error getting reminders: {str(e)}")
            return json.dumps({"error": f"Error retrieving reminders: {str(e)}"})
    
    async def _enhance_prompt(self, args: Dict[str, Any]):
        """Enhance a prompt"""
        try:
            result = await self.image_generator.enhance_prompt(args)
            return result
        except Exception as e:
            logging.error(f"Error in prompt enhancement: {str(e)}")
            return json.dumps({"error": f"Prompt enhancement failed: {str(e)}"})
    
    async def _image_to_text(self, args: Dict[str, Any]):
        """Convert image to text"""
        try:
            result = await self.image_generator.image_to_text(args)
            return result
        except Exception as e:
            logging.error(f"Error in image to text: {str(e)}")
            return json.dumps({"error": f"Image to text failed: {str(e)}"})
    
    async def _upscale_image(self, args: Dict[str, Any]):
        """Upscale an image"""
        try:
            result = await self.image_generator.upscale_image(args)
            return result
        except Exception as e:
            logging.error(f"Error in image upscaling: {str(e)}")
            return json.dumps({"error": f"Image upscaling failed: {str(e)}"})
    
    async def _photo_maker(self, args: Dict[str, Any]):
        """Create a photo"""
        try:
            result = await self.image_generator.photo_maker(args)
            return result
        except Exception as e:
            logging.error(f"Error in photo maker: {str(e)}")
            return json.dumps({"error": f"Photo maker failed: {str(e)}"})
    
    async def _generate_image_with_refiner(self, args: Dict[str, Any]):
        """Generate image with refiner"""
        try:
            result = await self.image_generator.generate_image_with_refiner(args)
            return result
        except Exception as e:
            logging.error(f"Error in image generation with refiner: {str(e)}")
            return json.dumps({"error": f"Image generation with refiner failed: {str(e)}"})
    
    # Helper method to download images with error handling
    async def _download_image(self, session, url):
        """Download an image from a URL with error handling"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logging.error(f"Failed to download image: HTTP {response.status}")
                    return None
        except Exception as e:
            logging.error(f"Error downloading image from {url}: {str(e)}")
            return None
    
    async def _run_chart_cleanup(self):
        """Run periodic chart cleanup"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                # Cleanup logic here
            except Exception as e:
                logging.error(f"Error in chart cleanup: {str(e)}")
    
    async def _run_file_cleanup(self):
        """Run periodic file cleanup"""
        while True:
            try:
                await asyncio.sleep(7200)  # Run every 2 hours
                # Cleanup logic here
            except Exception as e:
                logging.error(f"Error in file cleanup: {str(e)}")
    
    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count tokens in a list of messages using tiktoken o200k_base encoding.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            int: Total token count
        """
        try:
            total_tokens = 0
            
            for message in messages:
                # Count tokens for role
                if 'role' in message:
                    total_tokens += len(self.token_encoder.encode(message['role']))
                
                # Count tokens for content
                if 'content' in message:
                    content = message['content']
                    if isinstance(content, str):
                        # Simple string content
                        total_tokens += len(self.token_encoder.encode(content))
                    elif isinstance(content, list):
                        # Multi-modal content (text + images)
                        for item in content:
                            if isinstance(item, dict):
                                if item.get('type') == 'text' and 'text' in item:
                                    total_tokens += len(self.token_encoder.encode(item['text']))
                                elif item.get('type') == 'image_url':
                                    # Images use a fixed token cost (approximation)
                                    total_tokens += 765  # Standard cost for high-detail images
                
                # Add overhead for message formatting
                total_tokens += 4  # Overhead per message
            
            return total_tokens
            
        except Exception as e:
            logging.error(f"Error counting tokens: {str(e)}")
            # Return a conservative estimate if token counting fails
            return len(str(messages)) // 3  # Rough approximation
    
    def _trim_history_to_token_limit(self, history: List[Dict[str, Any]], model: str, target_tokens: int = None) -> List[Dict[str, Any]]:
        """
        Trim conversation history to fit within model token limits.
        
        Args:
            history: List of message dictionaries
            model: Model name to get token limit
            target_tokens: Optional custom target token count
            
        Returns:
            List[Dict[str, Any]]: Trimmed history that fits within token limits
        """
        try:
            # Get token limit for the model
            if target_tokens:
                token_limit = target_tokens
            else:
                token_limit = MODEL_TOKEN_LIMITS.get(model, DEFAULT_TOKEN_LIMIT)
            
            # Reserve 20% of tokens for the response and some buffer
            available_tokens = int(token_limit * 0.8)
            
            # Always keep the system message if present
            system_message = None
            conversation_messages = []
            
            for msg in history:
                if msg.get('role') == 'system':
                    system_message = msg
                else:
                    conversation_messages.append(msg)
            
            # Start with system message
            trimmed_history = []
            current_tokens = 0
            
            if system_message:
                system_tokens = self._count_tokens([system_message])
                if system_tokens < available_tokens:
                    trimmed_history.append(system_message)
                    current_tokens += system_tokens
                else:
                    # If system message is too large, truncate it
                    content = system_message.get('content', '')
                    if isinstance(content, str):
                        # Truncate system message to fit
                        words = content.split()
                        truncated_content = ''
                        for word in words:
                            test_content = truncated_content + ' ' + word if truncated_content else word
                            test_tokens = len(self.token_encoder.encode(test_content))
                            if test_tokens < available_tokens // 2:  # Use half available tokens for system
                                truncated_content = test_content
                            else:
                                break
                        
                        truncated_system = {
                            'role': 'system',
                            'content': truncated_content + '...[truncated]'
                        }
                        trimmed_history.append(truncated_system)
                        current_tokens += self._count_tokens([truncated_system])
            
            # Add conversation messages from most recent backwards
            available_for_conversation = available_tokens - current_tokens
            
            # Process messages in reverse order (most recent first)
            for msg in reversed(conversation_messages):
                msg_tokens = self._count_tokens([msg])
                
                if current_tokens + msg_tokens <= available_tokens:
                    if system_message:
                        # Insert after system message (position 1)
                        trimmed_history.insert(1, msg)
                    else:
                        # Insert at start if no system message
                        trimmed_history.insert(0, msg)
                    current_tokens += msg_tokens
                else:
                    # Stop adding more messages
                    break
            
            # Ensure we have at least the last user message if possible
            if len(conversation_messages) > 0 and len(trimmed_history) <= (1 if system_message else 0):
                last_msg = conversation_messages[-1]
                last_msg_tokens = self._count_tokens([last_msg])
                
                if last_msg_tokens < available_tokens:
                    if system_message:
                        trimmed_history.insert(-1, last_msg)
                    else:
                        trimmed_history.append(last_msg)
            
            logging.info(f"Trimmed history from {len(history)} to {len(trimmed_history)} messages "
                        f"({self._count_tokens(history)} to {self._count_tokens(trimmed_history)} tokens) "
                        f"for model {model}")
            
            return trimmed_history
            
        except Exception as e:
            logging.error(f"Error trimming history: {str(e)}")
            # Return a safe minimal history
            if history:
                # Keep system message + last user message if possible
                minimal_history = []
                for msg in history:
                    if msg.get('role') == 'system':
                        minimal_history.append(msg)
                        break
                
                # Add the last user message
                for msg in reversed(history):
                    if msg.get('role') == 'user':
                        minimal_history.append(msg)
                        break
                
                return minimal_history
            return history
