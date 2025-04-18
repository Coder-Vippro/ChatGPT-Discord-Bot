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
from datetime import datetime, timedelta
from src.utils.openai_utils import process_tool_calls, prepare_messages_for_api, get_tools_for_model
from src.utils.pdf_utils import process_pdf, send_response
from src.utils.code_utils import extract_code_blocks
from src.utils.reminder_utils import ReminderManager
from src.config.config import PDF_ALLOWED_MODELS

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

# Storage for user data files and charts
user_data_files = {}
user_charts = {}

# Try to import data analysis libraries early
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
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
            "code_interpreter": self._code_interpreter,
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
                timeout=aiohttp.ClientTimeout(total=30),
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
    
    async def _analyze_data(self, args: Dict[str, Any]) -> str:
        """
        [REMOVED] This function has been replaced by enhanced code_interpreter functionality
        """
        return json.dumps({"error": "This function has been deprecated. Please use code_interpreter instead."})
    
    async def _code_interpreter(self, args: Dict[str, Any]):
        """Handle code interpreter functionality"""
        try:
            # Import and call code interpreter
            from src.utils.code_interpreter import execute_code
            execute_result = await execute_code(args)
            
            # If there are visualizations, handle them
            if execute_result and execute_result.get("visualizations"):
                for i, viz_path in enumerate(execute_result["visualizations"]):
                    try:
                        with open(viz_path, 'rb') as f:
                            img_data = f.read()
                            
                        # Get the current message context from user_tasks
                        current_task = asyncio.current_task()
                        discord_message = None
                        for uid, tasks in user_tasks.items():
                            if current_task in tasks:
                                for task_info in tasks[current_task]:
                                    if 'message' in task_info:
                                        discord_message = task_info['message']
                                        break
                                break
                                
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
            error_msg = f"Error in code interpreter: {str(e)}"
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
            user_tasks[user_id] = []
        
        # Create and track a new task for this message
        task = asyncio.create_task(self._process_user_message(message))
        
        # Store the message object with the task for future reference
        task_info = {'message': message}
        user_tasks[user_id].append(task)
        user_tasks[user_id].append(task_info)  # This allows tools to find the original message
        
        # Use done callbacks to clean up and handle errors
        def task_done_callback(task):
            # Remove task from tracking list
            if user_id in user_tasks and task in user_tasks[user_id]:
                user_tasks[user_id].remove(task)
                # Also try to remove the task_info if it exists
                for item in list(user_tasks[user_id]):
                    if isinstance(item, dict) and 'message' in item:
                        user_tasks[user_id].remove(item)
                        break
            
            # Check for exceptions that weren't handled
            if task.done() and not task.cancelled():
                try:
                    task.result()
                except Exception as e:
                    logging.error(f"Unhandled task exception: {str(e)}")
        
        # Add the callback
        task.add_done_callback(task_done_callback)
    
    async def _handle_data_file(self, attachment, message, user_id, history, model, start_time):
        """
        Handle a data file attachment with analysis capabilities
        
        Args:
            attachment: The Discord file attachment
            message: The Discord message
            user_id: User ID for tracking
            history: Conversation history
            model: OpenAI model to use
            start_time: Timestamp when processing started
            
        Returns:
            Dict with analysis results
        """
        try:
            # Get file contents and determine file type
            file_extension = os.path.splitext(attachment.filename)[1].lower()
            file_bytes = await attachment.read()
            
            # Store the data file in user_data_files for future reference
            user_data_files[user_id] = {
                "bytes": file_bytes,
                "filename": attachment.filename,
                "timestamp": datetime.now()
            }
            
            # Extract query from message if any
            content = message.content.strip()
            query = content if content else "Analyze this data file and create relevant visualizations"
            
            # Determine analysis type based on query
            analysis_type = "comprehensive"  # Default
            if any(word in query.lower() for word in ['correlation', 'correlate', 'relationship']):
                analysis_type = "correlation"
            elif any(word in query.lower() for word in ['distribution', 'histogram', 'spread']):
                analysis_type = "distribution"
            elif any(word in query.lower() for word in ['bar', 'count', 'frequency']):
                analysis_type = "bar"
            elif any(word in query.lower() for word in ['scatter', 'relation']):
                analysis_type = "scatter"
            elif any(word in query.lower() for word in ['box', 'boxplot']):
                analysis_type = "box"
            
            # Save file to local storage
            from src.utils.code_utils import DATA_FILES_DIR
            temp_file_path = os.path.join(DATA_FILES_DIR, f"data_{user_id}_{int(time.time())}{file_extension}")
            
            # Save file
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
            with open(temp_file_path, "wb") as f:
                f.write(file_bytes)
                
            logging.info(f"Saved data file for analysis at: {temp_file_path}")
            
            # First send a message to show we're working
            await message.channel.send("📊 Analyzing data file and generating visualizations...")
            
            # Import analyze_data function from code_utils
            from src.utils.code_utils import analyze_data
            
            # Run analysis with the specific analysis type
            result = analyze_data(file_path=temp_file_path, user_id=user_id, analysis_type=analysis_type)
            
            # Handle error if any
            if "error" in result:
                await message.channel.send(f"❌ Error analyzing data: {result['error']}")
                return result
            
            # Send summary text
            summary = result.get("summary", {})
            summary_text = f"📋 **Data Analysis Summary**\n"
            summary_text += f"• Rows: {summary.get('rows', 'N/A')}\n"
            summary_text += f"• Columns: {summary.get('columns', 'N/A')}\n"
            summary_text += f"• Column names: {', '.join(summary.get('column_names', [])[:5])}{'...' if len(summary.get('column_names', [])) > 5 else ''}\n"
            
            # Add missing values info if available
            if "missing_values" in summary:
                missing_values = summary["missing_values"]
                missing_vals_text = "\n• Missing values: "
                missing_cols = [f"{col}({count})" for col, count in missing_values.items() if count > 0]
                summary_text += missing_vals_text + (", ".join(missing_cols[:3]) if missing_cols else "None") + "\n"
            
            await message.channel.send(summary_text)
            
            # Send each visualization as a separate file
            if "plots" in result and result["plots"]:
                for i, plot_path in enumerate(result["plots"]):
                    if os.path.exists(plot_path):
                        try:
                            # Read the image file
                            with open(plot_path, "rb") as f:
                                img_bytes = f.read()
                                
                            # Send the image directly to Discord
                            plot_filename = os.path.basename(plot_path)
                            sent_message = await message.channel.send(
                                f"📊 Visualization {i+1}:", 
                                file=discord.File(io.BytesIO(img_bytes), filename=plot_filename)
                            )
                            
                            # Store in user_charts for potential later reference
                            chart_id = f"{user_id}_{int(time.time())}_{i}"
                            user_charts[chart_id] = {
                                "image": img_bytes,
                                "timestamp": datetime.now(),
                                "user_id": user_id
                            }
                            
                            # Add image to conversation history - with fix for DMChannel
                            if len(history) > 0 and history[-1]["role"] == "assistant":
                                # Get the uploaded image URL safely - works in both guild channels and DMs
                                if sent_message.attachments and len(sent_message.attachments) > 0:
                                    plot_url = sent_message.attachments[0].url
                                    
                                    # If content is already a list, append to it
                                    if isinstance(history[-1]["content"], list):
                                        history[-1]["content"].append({
                                            "type": "image_url",
                                            "image_url": {"url": plot_url},
                                            "timestamp": datetime.now().isoformat()
                                        })
                                    else:
                                        # Convert text content to list with text and image
                                        text_content = history[-1]["content"]
                                        history[-1]["content"] = [
                                            {"type": "text", "text": text_content},
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": plot_url},
                                                "timestamp": datetime.now().isoformat()
                                            }
                                        ]
                                    
                                    # Save updated history
                                    await self.db.save_history(user_id, history)
                            
                        except Exception as e:
                            logging.error(f"Error sending visualization {i}: {str(e)}")
                            await message.channel.send(f"❌ Error displaying visualization {i+1}: {str(e)}")
            else:
                await message.channel.send("No visualizations were generated. Try a different data file or request.")
                
            return result
            
        except Exception as e:
            error_msg = f"Error handling data file: {str(e)}"
            logging.error(error_msg)
            traceback.print_exc()
            await message.channel.send(f"❌ {error_msg}")
            return {"error": error_msg}

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
                model = await self.db.get_user_model(user_id) or "openai/gpt-4.1-mini"  # Default to openai/gpt-4.1-mini if no model set
                
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
                                        logging.info(f"Decoded {attachment.filename} with UTF-8 (errors replaced)")
                                    except:
                                        # Fallback to latin-1 which can decode any byte sequence
                                        try:
                                            file_content = file_bytes.decode('latin-1')
                                            logging.info(f"Decoded {attachment.filename} with latin-1")
                                        except:
                                            # Final fallback
                                            file_content = file_bytes.decode('utf-8', errors='ignore')
                                            logging.warning(f"Decoded {attachment.filename} with UTF-8 (errors ignored)")
                                
                                # Add formatted text to extracted contents
                                extracted_text = f"\n\n--- Content of {attachment.filename} ---\n{file_content}\n--- End of {attachment.filename} ---\n\n"
                                extracted_text_contents.append(extracted_text)
                                
                                # Add a reference in the content
                                content.append({"type": "text", "text": f"[Attached file: {attachment.filename}"})
                                
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
            
            # Determine which models should have tools available
            # openai/openai/o1-mini and openai/openai/o1-preview do not support tools
            use_tools = model in ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/o1", "openai/o3-mini"]
            
            # Prepare API call parameters
            api_params = {
                "model": model,
                "messages": messages_for_api,
                "temperature": 0.3 if model in ["openai/gpt-4o", "openai/openai/gpt-4o-mini"] else 1,
                "top_p": 0.7 if model in ["openai/gpt-4o", "openai/gpt-4o-mini"] else 1,
                "timeout": 60  # Add an explicit timeout
            }
            
            # Add tools if using a supported model
            if use_tools:
                api_params["tools"] = get_tools_for_model()
            
            # Initialize variables to track tool responses
            image_generation_used = False
            chart_id = None
            image_urls = []  # Will store unique image URLs
            
            # Make the initial API call with retry logic
            response = await self._retry_api_call(lambda: self.client.chat.completions.create(**api_params))
            
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
                            # Check if operation is specified in arguments
                            try:
                                args = json.loads(tool_call.function.arguments)
                                operation = args.get("operation", "remove_background")
                                operation_name = operation.replace("_", " ").title()
                                await message.channel.send(f"Applying {operation_name}...")
                            except:
                                pass
                
                # Display appropriate messages based on which tools are being called
                if tool_messages.get("google_search") or tool_messages.get("scrape_webpage"):
                    await message.channel.send("🔍 Researching information...")
                
                if tool_messages.get("code_interpreter"):
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
                                content = msg.get('content', '')
                                if isinstance(content, str) and '{' in content:
                                    data = json.loads(content)
                                    if 'image_urls' in data and data['image_urls']:
                                        # Store the unique image URLs
                                        for url in data['image_urls']:
                                            if url not in image_urls:
                                                image_urls.append(url)
                                        
                                        logging.info(f"Found {len(data['image_urls'])} image URLs in tool response")
                            except Exception as e:
                                logging.error(f"Error parsing image URLs: {str(e)}")
                        
                        elif msg.get('role') == 'tool' and msg.get('name') == 'edit_image':
                            try:
                                content = msg.get('content', '')
                                if isinstance(content, str) and '{' in content:
                                    data = json.loads(content)
                                    # Check for Discord image URLs
                                    if 'discord_image_urls' in data and data['discord_image_urls']:
                                        for url in data['discord_image_urls']:
                                            if url not in image_urls:
                                                image_urls.append(url)
                                        logging.info(f"Found {len(data['discord_image_urls'])} edited image URLs in tool response")
                                    # Also check regular image URLs in case that's all we have
                                    elif 'image_urls' in data and data['image_urls']:
                                        for url in data['image_urls']:
                                            if url not in image_urls:
                                                image_urls.append(url)
                                        logging.info(f"Found {len(data['image_urls'])} edited image URLs in tool response")
                            except Exception as e:
                                logging.error(f"Error parsing edited image URLs: {str(e)}")
                        
                        elif msg.get('role') == 'tool' and msg.get('name') == 'code_interpreter':
                            try:
                                content = msg.get('content', '')
                                if isinstance(content, str) and '{' in content:
                                    data = json.loads(content)
                                    if data.get('has_chart') and 'chart_id' in data:
                                        chart_id = data.get('chart_id')
                                        logging.info(f"Found chart ID in tool response: {chart_id}")
                            except Exception as e:
                                logging.error(f"Error parsing chart data: {str(e)}")
                
                # If tool calls were processed, make another API call with the updated messages
                if tool_calls_processed and updated_messages:
                    response = await self._retry_api_call(lambda: self.client.chat.completions.create(
                        model=model,
                        messages=updated_messages,
                        temperature=0.3 if model in ["openai/gpt-4o", "openai/gpt-4o-mini"] else 1,
                        timeout=60
                    ))
            
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
            if model in ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/o1", "openai/o1-mini", "openai/o3-mini"]:
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
                    if len(new_history) > 20:  # Truncate to last 20 messages if too long
                        new_history = new_history[:1] + new_history[-19:]  # Keep system message + last 19 messages
                        
                    await self.db.save_history(user_id, new_history)
                else:
                    # For models with system prompt support, just append to regular history
                    if has_images:
                        history.append({"role": "assistant", "content": content_with_images})
                    else:
                        history.append({"role": "assistant", "content": reply})
                    
                    # Only keep a reasonable amount of history
                    if len(history) > 20:  # Truncate to last 20 messages if too long
                        history = history[:1] + history[-19:]  # Keep system message + last 19 messages
                        
                    await self.db.save_history(user_id, history)
            
            # Send the response text
            await send_response(message.channel, reply)
            
            # Handle charts from code interpreter if present
            if chart_id and chart_id in user_charts:
                try:
                    chart_data = user_charts[chart_id]["image"]
                    chart_filename = f"chart_{chart_id}.png"
                    
                    # Send the chart to Discord and get the URL
                    chart_message = await message.channel.send(
                        "📊 Chart generated:",
                        file=discord.File(io.BytesIO(chart_data), filename=chart_filename)
                    )
                    
                    # Get the chart URL from Discord attachment
                    if chart_message.attachments and len(chart_message.attachments) > 0:
                        chart_url = chart_message.attachments[0].url
                        # Add image URL to history with timestamp
                        if history[-1]["role"] == "assistant":
                            # If the last message was from the assistant, append the image to it
                            if isinstance(history[-1]["content"], list):
                                history[-1]["content"].append({
                                    "type": "image_url",
                                    "image_url": {"url": chart_url},
                                    "timestamp": datetime.now().isoformat()
                                })
                            else:
                                # Convert string content to list with text and image
                                text_content = history[-1]["content"]
                                history[-1]["content"] = [
                                    {"type": "text", "text": text_content},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": chart_url},
                                        "timestamp": datetime.now().isoformat()
                                    }
                                ]
                        
                        # Save updated history immediately after getting the URL
                        await self.db.save_history(user_id, history)
                        
                except Exception as e:
                    logging.error(f"Error sending chart: {str(e)}")
            
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
            current_task = asyncio.current_task()
            user_id = None
            
            for uid, tasks in user_tasks.items():
                if current_task in tasks:
                    user_id = uid
                    break
                    
            if not user_id:
                return json.dumps({"error": "Could not identify user"})
                
            # Parse time using user's timezone if available
            remind_at = await self.reminder_manager.parse_time(time_str, user_id)
            
            if not remind_at:
                return json.dumps({
                    "error": f"Could not parse time '{time_str}'. Please use formats like '30m', '2h', '1d', 'tomorrow', or '15:00'"
                })
                
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
            current_task = asyncio.current_task()
            user_id = None
            
            for uid, tasks in user_tasks.items():
                if current_task in tasks:
                    user_id = uid
                    break
                    
            if not user_id:
                return json.dumps({"error": "Could not identify user"})
                
            # Get reminders list
            reminders = await self.reminder_manager.get_user_reminders(user_id)
            
            # Format reminders
            formatted_reminders = []
            for reminder in reminders:
                formatted_reminders.append({
                    "id": str(reminder["_id"]),
                    "content": reminder["content"],
                    "time": reminder["remind_at"].strftime("%Y-%m-%d %H:%M:%S"),
                    "created_at": reminder["created_at"].strftime("%Y-%m-%d %H:%M:%S")
                })
                
            return json.dumps({
                "success": True,
                "reminders": formatted_reminders,
                "count": len(formatted_reminders)
            })
            
        except Exception as e:
            logging.error(f"Error getting reminders: {str(e)}")
            return json.dumps({"error": f"Error retrieving reminders: {str(e)}"})
    
    # Helper method to download images with error handling
    async def _download_image(self, session, url):
        """Download an image from a URL with error handling"""
        try:
            async with session.get(url) as resp:
                if (resp.status == 200):
                    return await resp.read()
                else:
                    logging.warning(f"Failed to download image, status: {resp.status}")
                    return None
        except Exception as e:
            logging.error(f"Error downloading image: {str(e)}")
            return None
    
    # API call with retry logic
    async def _retry_api_call(self, call_func, max_retries=3, base_delay=1):
        """Retry API calls with exponential backoff"""
        retries = 0
        last_error = None
        
        while retries < max_retries:
            try:
                return await call_func()
            except Exception as e:
                last_error = e
                
                # Check for 413 error (payload too large)
                error_str = str(e).lower()
                if '413' in error_str and ('entity too large' in error_str or 'exceeds the capacity limit' in error_str):
                    # Find the user_id from current task
                    current_task = asyncio.current_task()
                    user_id = None
                    discord_message = None
                    
                    for uid, tasks in user_tasks.items():
                        if current_task in tasks:
                            user_id = uid
                            # Look for message context in task info
                            for task_info in tasks:
                                if isinstance(task_info, dict) and 'message' in task_info:
                                    discord_message = task_info['message']
                                    break
                            break
                    
                    if user_id and discord_message:
                        logging.warning(f"413 Error detected for user {user_id}. Clearing history.")
                        try:
                            # Clear history but preserve system message
                            history = await self.db.get_history(user_id)
                            if history:
                                system_msg = None
                                # Extract system message if exists
                                for msg in history:
                                    if msg.get('role') == 'system':
                                        system_msg = msg
                                        break
                                
                                # Clear history but keep system message if found
                                new_history = []
                                if system_msg:
                                    new_history.append(system_msg)
                                
                                # Save the reset history
                                await self.db.save_history(user_id, new_history)
                                
                                # Send a helpful message to the user
                                await discord_message.channel.send(
                                    "⚠️ Your conversation history was too large for processing. "
                                    "I've cleared your previous messages to fix this issue. "
                                    "Please try your request again."
                                )
                                
                                # Re-raise to exit this request's processing
                                raise e
                        except Exception as clear_error:
                            logging.error(f"Error clearing history after 413 error: {str(clear_error)}")
                
                # Normal retry processing
                retries += 1
                if retries >= max_retries:
                    break
                    
                # Exponential backoff with jitter
                delay = base_delay * (2 ** (retries - 1)) * (0.5 + 0.5 * (asyncio.get_event_loop().time() % 1))
                logging.warning(f"API call failed: {str(e)}. Retrying in {delay:.2f}s ({retries}/{max_retries})")
                await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        logging.error(f"All API call retries failed: {str(last_error)}")
        raise last_error
            
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
            # Run in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                self.thread_pool,
                functools.partial(google_custom_search, query, num_results)
            )
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
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        content = await loop.run_in_executor(
            self.thread_pool,
            functools.partial(scrape_web_content, url)
        )
        return content
    
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
    
    async def _edit_image(self, args: Dict[str, Any]):
        """
        Execute image editing tool function (like background removal).
        
        Args:
            args: Arguments containing image_url and optional operation
            
        Returns:
            str: JSON string with edited image information
        """
        image_url = args.get("image_url", "")
        operation = args.get("operation", "remove_background")
        
        if not image_url:
            return json.dumps({"error": "No image URL provided"})
        
        try:
            # Find the current message context for displaying results
            current_task = asyncio.current_task()
            discord_message = None
            user_id = None
            
            for uid, tasks in user_tasks.items():
                if current_task in tasks:
                    user_id = uid
                    # Look for message context in task info
                    for task_info in tasks:
                        if isinstance(task_info, dict) and 'message' in task_info:
                            discord_message = task_info['message']
                            break
                    break
            
            # Process the image with the requested operation
            result = await self.image_generator.edit_image(image_url, operation)
            
            if result["success"]:
                # If we have a message context and binary images, send them to Discord
                edited_image_urls = []
                
                for i, img_data in enumerate(result["binary_images"]):
                    try:
                        # Send the edited image directly to Discord
                        operation_name = operation.replace("_", " ").title()
                        sent_message = await discord_message.channel.send(
                            f"🖼️ {operation_name} result:",
                            file=discord.File(io.BytesIO(img_data), filename=f"edited_image_{i+1}.png")
                        )
                        
                        # Store the URL if available
                        if sent_message.attachments and len(sent_message.attachments) > 0:
                            edited_image_urls.append(sent_message.attachments[0].url)
                    except Exception as e:
                        logging.error(f"Error sending edited image: {str(e)}")
                
                # Add any newly sent image URLs to the result
                if edited_image_urls:
                    result["discord_image_urls"] = edited_image_urls
                    
                    # Find the most recent assistant message in history and append the image URLs as text
                    if user_id is not None:
                        history = await self.db.get_history(user_id)
                        if history and len(history) > 0:
                            # Find the most recent assistant message
                            for i in range(len(history) - 1, -1, -1):
                                if history[i].get("role") == "assistant":
                                    # Append image URLs as plain text
                                    images_text = "\n\nProcessed images:"
                                    for j, url in enumerate(edited_image_urls):
                                        images_text += f"\n• Image {j+1}: {url}"
                                    
                                    # If content is already a list
                                    if isinstance(history[i]["content"], list):
                                        # Find the text item and append to it
                                        for item in history[i]["content"]:
                                            if item.get("type") == "text":
                                                item["text"] += images_text
                                                break
                                        else:
                                            # If no text item found, add a new one
                                            history[i]["content"].append({
                                                "type": "text",
                                                "text": images_text
                                            })
                                    else:
                                        # Convert string content to text with appended URLs
                                        history[i]["content"] += images_text
                                    
                                    # Save updated history
                                    await self.db.save_history(user_id, history)
                                    break
                
                return json.dumps({
                    "message": f"Successfully {operation.replace('_', ' ')}ed image",
                    "image_urls": result["image_urls"],
                    "operation": operation,
                    "discord_image_urls": result.get("discord_image_urls", [])
                })
            else:
                return json.dumps({
                    "error": result.get("error", f"Unknown error during {operation}"),
                    "operation": operation
                })
        except Exception as e:
            error_msg = f"Error in image editing: {str(e)}"
            logging.error(error_msg)
            return json.dumps({
                "error": error_msg,
                "operation": operation
            })
    
    async def _analyze_data_file(self, args: Dict[str, Any]) -> str:
        """
        Analyze a data file directly using the AI and data analysis tools.
        
        Args:
            args: Dictionary containing file_path and optional parameters
            
        Returns:
            JSON string with analysis results
        """
        file_path = args.get("file_path", "")
        user_id = args.get("user_id", "default")
        
        if not file_path:
            return json.dumps({"error": "No file path provided"})
            
        # Find real user_id from current task if not provided
        if user_id == "default":
            current_task = asyncio.current_task()
            for uid, tasks in user_tasks.items():
                if current_task in tasks:
                    user_id = uid
                    break
        
        try:
            # Try to find file in user data files if it's not a direct path
            if not os.path.exists(file_path) and user_id in user_data_files:
                # Create temporary file from bytes in memory
                from src.utils.code_utils import DATA_FILES_DIR
                file_bytes = user_data_files[user_id].get("bytes")
                filename = user_data_files[user_id].get("filename", "data.csv")
                file_extension = os.path.splitext(filename)[1]
                
                temp_file_path = os.path.join(DATA_FILES_DIR, f"data_{user_id}_{int(time.time())}{file_extension}")
                
                with open(temp_file_path, "wb") as f:
                    f.write(file_bytes)
                
                file_path = temp_file_path
                logging.info(f"Created temporary file for analysis: {file_path}")
                
            # Import analyze_data function from code_utils
            from src.utils.code_utils import analyze_data
            
            # Run analysis
            result = analyze_data(file_path, user_id)
            
            # Handle error if any
            if "error" in result:
                return json.dumps({"error": result["error"]})
                
            # Format plots for response
            plots_data = []
            if "plots" in result:
                for plot_path in result["plots"]:
                    if os.path.exists(plot_path):
                        try:
                            # Read the image file and encode
                            with open(plot_path, "rb") as f:
                                img_bytes = f.read()
                                # Store chart in user charts
                                chart_id = f"{user_id}_{int(time.time())}_{plots_data.__len__()}"
                                user_charts[chart_id] = {
                                    "image": img_bytes,
                                    "timestamp": datetime.now(),
                                    "user_id": user_id
                                }
                                
                                plots_data.append({
                                    "chart_id": chart_id,
                                    "path": plot_path
                                })
                        except Exception as e:
                            logging.error(f"Error processing visualization: {str(e)}")
                            
            # Format response
            response = {
                "success": True,
                "file_path": file_path,
                "summary": result.get("summary", {}),
                "visualizations": plots_data,
                "visualization_count": len(plots_data)
            }
            
            return json.dumps(response)
            
        except Exception as e:
            logging.error(f"Error in analyze_data_file: {str(e)}")
            return json.dumps({"error": f"Failed to analyze data file: {str(e)}"})
            
    async def _generate_data_analysis_code(self, args: Dict[str, Any]) -> str:
        """
        Generate Python code for analyzing data based on natural language request.
        
        Args:
            args: Dictionary containing file_path and analysis_request
            
        Returns:
            JSON string with generated code
        """
        file_path = args.get("file_path", "")
        analysis_request = args.get("analysis_request", "")
        
        if not file_path:
            return json.dumps({"error": "No file path provided"})
            
        if not analysis_request:
            return json.dumps({"error": "No analysis request provided"})
            
        # Find user_id from current task
        current_task = asyncio.current_task()
        user_id = None
        for uid, tasks in user_tasks.items():
            if current_task in tasks:
                user_id = uid
                break
                
        try:
            # Try to find file in user data files if it's not a direct path
            if not os.path.exists(file_path) and user_id in user_data_files:
                # Create temporary file from bytes in memory
                from src.utils.code_utils import DATA_FILES_DIR
                file_bytes = user_data_files[user_id].get("bytes")
                filename = user_data_files[user_id].get("filename", "data.csv")
                file_extension = os.path.splitext(filename)[1]
                
                temp_file_path = os.path.join(DATA_FILES_DIR, f"data_{user_id}_{int(time.time())}{file_extension}")
                
                with open(temp_file_path, "wb") as f:
                    f.write(file_bytes)
                
                file_path = temp_file_path
                logging.info(f"Created temporary file for code generation: {file_path}")
            
            # Import code generation function
            from src.utils.code_utils import generate_analysis_code
            
            # Generate code based on user request
            code = generate_analysis_code(file_path, analysis_request)
            
            # Format response
            response = {
                "success": True,
                "file_path": file_path,
                "analysis_request": analysis_request,
                "generated_code": code,
                "message": "You can run this code using the code_interpreter tool."
            }
            
            return json.dumps(response)
            
        except Exception as e:
            logging.error(f"Error in generate_data_analysis_code: {str(e)}")
            return json.dumps({"error": f"Failed to generate analysis code: {str(e)}"})
    
    @staticmethod
    async def stop_user_tasks(user_id: int):
        """
        Stop all tasks for a specific user.
        
        Args:
            user_id: The Discord user ID
        """
        logging.info(f"MessageHandler: Stopping all tasks for user {user_id}")
        
        # Cancel all tasks for the user
        if user_id in user_tasks:
            # Make a copy since we'll modify the list while iterating
            tasks_to_cancel = user_tasks[user_id].copy()
            for task in tasks_to_cancel:
                try:
                    if not task.done() and not task.cancelled():
                        task.cancel()
                        logging.info(f"Cancelled task for user {user_id}")
                except Exception as e:
                    logging.error(f"Error cancelling task: {str(e)}")
            
            # Clear the list
            user_tasks[user_id] = []
            logging.info(f"Cleared all tasks for user {user_id}")
            
        # Find any PDF processing tasks in progress
        # These might be running in asyncio tasks that aren't directly tracked in user_tasks
        import asyncio
        # Check all tasks in the event loop
        for task in asyncio.all_tasks():
            task_name = task.get_name()
            # Look for PDF processing tasks that match the user ID
            if f"pdf_processing_{user_id}" in task_name or f"process_pdf_{user_id}" in task_name:
                try:
                    if not task.done() and not task.cancelled():
                        task.cancel()
                        logging.info(f"Cancelled PDF processing task {task_name} for user {user_id}")
                except Exception as e:
                    logging.error(f"Error cancelling PDF task {task_name}: {str(e)}")
                    
        # Clear any data files associated with this user to prevent further processing
        from src.utils.code_utils import DATA_FILES_DIR
        try:
            if os.path.exists(DATA_FILES_DIR):
                for filename in os.listdir(DATA_FILES_DIR):
                    if f"{user_id}_" in filename:
                        file_path = os.path.join(DATA_FILES_DIR, filename)
                        try:
                            os.remove(file_path)
                            logging.info(f"Removed data file: {file_path}")
                        except Exception as e:
                            logging.error(f"Error removing data file {file_path}: {str(e)}")
        except Exception as e:
            logging.error(f"Error cleaning up data files: {str(e)}")
            
        # Also clear any cached data for this user
        if user_id in user_data_files:
            del user_data_files[user_id]
        
        # Remove any chart data for this user
        chart_ids_to_remove = []
        for chart_id in user_charts:
            if chart_id.startswith(f"{user_id}_"):
                chart_ids_to_remove.append(chart_id)
                
        for chart_id in chart_ids_to_remove:
            if chart_id in user_charts:
                del user_charts[chart_id]
                
        if chart_ids_to_remove:
            logging.info(f"Removed {len(chart_ids_to_remove)} charts for user {user_id}")
            
        logging.info(f"Completed stopping all tasks for user {user_id}")
            
    async def _run_chart_cleanup(self):
        """Run periodic chart cleanup to remove old chart data"""
        try:
            while True:
                # Clean up charts older than 23 hours
                await self._cleanup_old_charts(max_age_hours=23)
                await asyncio.sleep(3600)  # Check every hour
        except asyncio.CancelledError:
            logging.info("Chart cleanup task was cancelled")
            raise
        except Exception as e:
            logging.error(f"Error in chart cleanup task: {str(e)}")
    
    async def _cleanup_old_charts(self, max_age_hours=23):
        """
        Clean up chart data and temporary files older than the specified time
        
        Args:
            max_age_hours: Maximum age in hours before deleting charts
        """
        try:
            # Calculate expiration time
            expiration_time = datetime.now() - timedelta(hours=max_age_hours)
            expired_keys = []
            
            # Find expired charts
            for chart_id, chart_data in user_charts.items():
                if chart_data["timestamp"] < expiration_time:
                    expired_keys.append(chart_id)
            
            # Remove expired charts
            for key in expired_keys:
                del user_charts[key]
                
            if expired_keys:
                logging.info(f"Cleaned up {len(expired_keys)} expired charts")
                
            # Clean up expired data files from memory
            expired_users = []
            for user_id, data_info in user_data_files.items():
                if datetime.now().timestamp() - data_info.get("timestamp", 0) > max_age_hours * 3600:
                    expired_users.append(user_id)
            
            # Remove expired data files from memory
            for user_id in expired_users:
                del user_data_files[user_id]
                
            if expired_users:
                logging.info(f"Cleaned up {len(expired_users)} expired data files from memory")
                
            # Also clean up physical temporary data files
            from src.utils.code_utils import DATA_FILES_DIR
            if os.path.exists(DATA_FILES_DIR):
                now = time.time()
                for filename in os.listdir(DATA_FILES_DIR):
                    file_path = os.path.join(DATA_FILES_DIR, filename)
                    try:
                        # Check if the file is older than max_age_hours
                        if os.path.isfile(file_path) and os.path.getmtime(file_path) < now - (max_age_hours * 3600):
                            os.remove(file_path)
                            logging.info(f"Removed expired temporary data file: {file_path}")
                    except Exception as file_error:
                        logging.error(f"Error removing temporary file {file_path}: {str(file_error)}")
                
        except Exception as e:
            logging.error(f"Error cleaning up charts and data files: {str(e)}")
            logging.error(traceback.format_exc())
            
    async def _run_file_cleanup(self):
        """Run periodic cleanup of data files and visualization files"""
        try:
            while True:
                # Clean up files older than 23 hours
                try:
                    # Clean up temp_data_files directory
                    temp_data_dir = os.path.join(os.path.dirname(__file__), '..', 'temp_data_files')
                    if os.path.exists(temp_data_dir):
                        current_time = time.time()
                        max_age = 23 * 3600  # 23 hours in seconds
                        
                        # List all files in the directory
                        for filename in os.listdir(temp_data_dir):
                            file_path = os.path.join(temp_data_dir, filename)
                            try:
                                # Check if file is older than max_age
                                if os.path.isfile(file_path):
                                    file_age = current_time - os.path.getmtime(file_path)
                                    if file_age > max_age:
                                        os.remove(file_path)
                                        logging.info(f"Cleaned up old file: {filename}")
                            except Exception as e:
                                logging.error(f"Error cleaning up file {filename}: {str(e)}")
                                continue
                                
                    # Also clean up any other temp files from code_utils
                    from src.utils.code_utils import clean_old_files, DATA_FILES_DIR
                    clean_old_files(max_age_hours=23)
                    
                except Exception as e:
                    logging.error(f"Error in file cleanup task: {str(e)}")
                    
                await asyncio.sleep(3600)  # Check every hour
                
        except asyncio.CancelledError:
            logging.info("File cleanup task was cancelled")
            raise
        except Exception as e:
            logging.error(f"Error in file cleanup task: {str(e)}")
            logging.error(traceback.format_exc())
            
    async def close(self):
        """Clean up resources when closing the bot"""
        # Stop reminder manager
        await self.reminder_manager.stop()
        
        # Cancel chart cleanup task
        if hasattr(self, 'chart_cleanup_task'):
            self.chart_cleanup_task.cancel()
            try:
                await self.chart_cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close HTTP session
        if self.aiohttp_session:
            await self.aiohttp_session.close()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
    
    async def _enhance_prompt(self, args: Dict[str, Any]):
        """
        Execute prompt enhancement tool function.
        
        Args:
            args: Arguments containing the prompt and enhancement parameters
            
        Returns:
            str: JSON string with enhanced prompt information
        """
        prompt = args.get("prompt", "")
        num_versions = min(int(args.get("num_versions", 3)), 5)
        max_length = int(args.get("max_length", 100))
        
        if not prompt:
            return json.dumps({"error": "No prompt provided for enhancement"})
        
        try:
            # Find the current message context for displaying results
            current_task = asyncio.current_task()
            discord_message = None
            user_id = None
            
            for uid, tasks in user_tasks.items():
                if current_task in tasks:
                    user_id = uid
                    # Look for message context in task info
                    for task_info in tasks:
                        if isinstance(task_info, dict) and 'message' in task_info:
                            discord_message = task_info['message']
                            break
                    break
            
            # Process the prompt enhancement
            result = await self.image_generator.enhance_prompt(prompt, num_versions, max_length)
            
            if result["success"]:
                # Format enhanced prompts for response
                enhanced_prompts = result.get("enhanced_prompts", [])
                formatted_response = {
                    "message": f"Successfully enhanced prompt with {len(enhanced_prompts)} variations",
                    "original_prompt": prompt,
                    "enhanced_prompts": enhanced_prompts,
                    "prompt_count": len(enhanced_prompts)
                }
                
                # If we have a message context, send the enhanced prompts directly in Discord
                if discord_message and enhanced_prompts:
                    try:
                        prompt_message = "**Enhanced Prompt Variations:**\n\n"
                        for i, enhanced in enumerate(enhanced_prompts, 1):
                            prompt_message += f"**{i}.** {enhanced}\n\n"
                        
                        await discord_message.channel.send(prompt_message)
                    except Exception as e:
                        logging.error(f"Error sending enhanced prompts: {str(e)}")
                
                return json.dumps(formatted_response)
            else:
                return json.dumps({
                    "error": result.get("error", "Unknown error enhancing prompt"),
                    "original_prompt": prompt
                })
        except Exception as e:
            error_msg = f"Error in prompt enhancement: {str(e)}"
            logging.error(error_msg)
            return json.dumps({
                "error": error_msg,
                "original_prompt": prompt
            })

    async def _image_to_text(self, args: Dict[str, Any]):
        """
        Execute image-to-text conversion tool function.
        
        Args:
            args: Arguments containing the image URL
            
        Returns:
            str: JSON string with image caption information
        """
        image_url = args.get("image_url", "")
        
        if not image_url:
            return json.dumps({"error": "No image URL provided"})
        
        try:
            # Find the current message context for displaying results
            current_task = asyncio.current_task()
            discord_message = None
            user_id = None
            
            for uid, tasks in user_tasks.items():
                if current_task in tasks:
                    user_id = uid
                    # Look for message context in task info
                    for task_info in tasks:
                        if isinstance(task_info, dict) and 'message' in task_info:
                            discord_message = task_info['message']
                            break
                    break
            
            # Process the image captioning
            result = await self.image_generator.image_to_text(image_url)
            
            if result["success"]:
                caption = result.get("caption", "")
                formatted_response = {
                    "message": "Successfully generated caption for image",
                    "image_url": image_url,
                    "caption": caption
                }
                
                # If we have a message context and a caption, send it directly in Discord
                if discord_message and caption:
                    try:
                        await discord_message.channel.send(f"🖼️ **Image Caption:** {caption}")
                        
                        # Append the caption and image URL as plain text to the most recent assistant message
                        if user_id is not None:
                            history = await self.db.get_history(user_id)
                            if history and len(history) > 0:
                                # Find the most recent assistant message
                                for i in range(len(history) - 1, -1, -1):
                                    if history[i].get("role") == "assistant":
                                        caption_text = f"\n\nImage Caption for {image_url}:\n{caption}"
                                        
                                        # If content is already a list
                                        if isinstance(history[i]["content"], list):
                                            # Find the text item and append to it
                                            for item in history[i]["content"]:
                                                if item.get("type") == "text":
                                                    item["text"] += caption_text
                                                    break
                                            else:
                                                # If no text item found, add a new one
                                                history[i]["content"].append({
                                                    "type": "text",
                                                    "text": caption_text
                                                })
                                        else:
                                            # Append to string content
                                            history[i]["content"] += caption_text
                                        
                                        # Save updated history
                                        await self.db.save_history(user_id, history)
                                        break
                        
                    except Exception as e:
                        logging.error(f"Error sending image caption: {str(e)}")
                
                return json.dumps(formatted_response)
            else:
                return json.dumps({
                    "error": result.get("error", "Unknown error during image-to-text conversion"),
                    "image_url": image_url
                })
        except Exception as e:
            error_msg = f"Error in image-to-text conversion: {str(e)}"
            logging.error(error_msg)
            return json.dumps({
                "error": error_msg,
                "image_url": image_url
            })
    
    async def _upscale_image(self, args: Dict[str, Any]):
        """
        Execute image upscaling tool function.
        
        Args:
            args: Arguments containing the image URL and scale factor
            
        Returns:
            str: JSON string with upscaled image information
        """
        image_url = args.get("image_url", "")
        scale_factor = min(int(args.get("scale_factor", 4)), 4)
        
        if not image_url:
            return json.dumps({"error": "No image URL provided"})
        
        try:
            # Find the current message context for displaying results
            current_task = asyncio.current_task()
            discord_message = None
            user_id = None
            
            for uid, tasks in user_tasks.items():
                if current_task in tasks:
                    user_id = uid
                    # Look for message context in task info
                    for task_info in tasks:
                        if isinstance(task_info, dict) and 'message' in task_info:
                            discord_message = task_info['message']
                            break
                    break
            
            # Process the image upscaling
            result = await self.image_generator.upscale_image(image_url, scale_factor)
            
            if result["success"]:
                # If we have a message context and binary images, send them to Discord
                discord_image_urls = []
                if discord_message and result.get("binary_images"):
                    for i, img_data in enumerate(result["binary_images"]):
                        try:
                            # Send the upscaled image directly to Discord
                            sent_message = await discord_message.channel.send(
                                f"🖼️ Upscaled image (x{scale_factor}):",
                                file=discord.File(io.BytesIO(img_data), filename=f"upscaled_image_{i+1}.png")
                            )
                            
                            # Store the URL if available
                            if sent_message.attachments and len(sent_message.attachments) > 0:
                                discord_image_urls.append(sent_message.attachments[0].url)
                        except Exception as e:
                            logging.error(f"Error sending upscaled image: {str(e)}")
                
                # Add the image URLs to the formatted response
                formatted_response = {
                    "message": f"Successfully upscaled image (x{scale_factor})",
                    "original_url": image_url,
                    "scale_factor": scale_factor,
                    "image_urls": result["image_urls"],
                    "discord_image_urls": discord_image_urls
                }
                
                # Create explicit text links for the images
                image_links_text = ""
                if discord_image_urls:
                    image_links_text = "\n\nUpscaled images:"
                    for j, url in enumerate(discord_image_urls):
                        image_links_text += f"\n• Image {j+1}: {url}"
                    
                    # Append image URLs as plain text to the most recent assistant message
                    if user_id is not None:
                        history = await self.db.get_history(user_id)
                        if history and len(history) > 0:
                            # Find the most recent assistant message
                            for i in range(len(history) - 1, -1, -1):
                                if history[i].get("role") == "assistant":
                                    # If content is already a list
                                    if isinstance(history[i]["content"], list):
                                        # Find the text item and append to it
                                        for item in history[i]["content"]:
                                            if item.get("type") == "text":
                                                item["text"] += image_links_text
                                                break
                                        else:
                                            # If no text item found, add a new one
                                            history[i]["content"].append({
                                                "type": "text",
                                                "text": image_links_text
                                            })
                                    else:
                                        # Append to string content
                                        history[i]["content"] += image_links_text
                                    
                                    # Save updated history
                                    await self.db.save_history(user_id, history)
                                    break
                
                # Add image links to response
                formatted_response["image_links_text"] = image_links_text
                
                return json.dumps(formatted_response)
            else:
                return json.dumps({
                    "error": result.get("error", "Unknown error during image upscaling"),
                    "image_url": image_url,
                    "scale_factor": scale_factor
                })
        except Exception as e:
            error_msg = f"Error in image upscaling: {str(e)}"
            logging.error(error_msg)
            return json.dumps({
                "error": error_msg,
                "image_url": image_url,
                "scale_factor": scale_factor
            })
    
    async def _photo_maker(self, args: Dict[str, Any]):
        """
        Execute photo maker tool function.
        
        Args:
            args: Arguments containing prompt, input_images, and other parameters
            
        Returns:
            str: JSON string with generated image information
        """
        prompt = args.get("prompt", "")
        input_images = args.get("input_images", [])
        style = args.get("style", "No style")
        strength = min(max(int(args.get("strength", 40)), 1), 100)  # Clamp between 1-100
        num_images = min(int(args.get("num_images", 1)), 4)
        
        if not prompt:
            return json.dumps({"error": "No prompt provided for photo maker"})
            
        if not input_images or not isinstance(input_images, list) or len(input_images) == 0:
            return json.dumps({"error": "No input images provided for photo maker"})
        
        try:
            # Find the current message context for displaying results
            current_task = asyncio.current_task()
            discord_message = None
            user_id = None
            
            for uid, tasks in user_tasks.items():
                if current_task in tasks:
                    user_id = uid
                    # Look for message context in task info
                    for task_info in tasks:
                        if isinstance(task_info, dict) and 'message' in task_info:
                            discord_message = task_info['message']
                            break
                    break
            
            # Process the photo maker request
            result = await self.image_generator.photo_maker(
                prompt=prompt,
                input_images=input_images,
                style=style,
                strength=strength,
                num_images=num_images
            )
            
            if result["success"]:
                # If we have a message context and binary images, send them to Discord
                discord_image_urls = []
                if discord_message and result.get("binary_images"):
                    for i, img_data in enumerate(result["binary_images"]):
                        try:
                            # Send the generated image directly to Discord
                            sent_message = await discord_message.channel.send(
                                f"🖼️ Photo Maker result ({i+1}/{len(result['binary_images'])}):",
                                file=discord.File(io.BytesIO(img_data), filename=f"photo_maker_{i+1}.png")
                            )
                            
                            # Store the URL if available
                            if sent_message.attachments and len(sent_message.attachments) > 0:
                                discord_image_urls.append(sent_message.attachments[0].url)
                        except Exception as e:
                            logging.error(f"Error sending photo maker image: {str(e)}")
                
                # Add the image URLs to the formatted response
                formatted_response = {
                    "message": f"Successfully generated {result['image_count']} images with Photo Maker",
                    "prompt": prompt,
                    "style": style,
                    "strength": strength,
                    "image_urls": result["image_urls"],
                    "discord_image_urls": discord_image_urls
                }
                
                # Create explicit text links for the images
                image_links_text = ""
                if discord_image_urls:
                    image_links_text = "\n\nPhoto Maker results:"
                    for j, url in enumerate(discord_image_urls):
                        image_links_text += f"\n• Image {j+1}: {url}"
                    
                    # Append image URLs as plain text to the most recent assistant message
                    if user_id is not None:
                        history = await self.db.get_history(user_id)
                        if history and len(history) > 0:
                            # Find the most recent assistant message
                            for i in range(len(history) - 1, -1, -1):
                                if history[i].get("role") == "assistant":
                                    # If content is already a list
                                    if isinstance(history[i]["content"], list):
                                        # Find the text item and append to it
                                        for item in history[i]["content"]:
                                            if item.get("type") == "text":
                                                item["text"] += image_links_text
                                                break
                                        else:
                                            # If no text item found, add a new one
                                            history[i]["content"].append({
                                                "type": "text",
                                                "text": image_links_text
                                            })
                                    else:
                                        # Append to string content
                                        history[i]["content"] += image_links_text
                                    
                                    # Save updated history
                                    await self.db.save_history(user_id, history)
                                    break
                
                # Add image links to response
                formatted_response["image_links_text"] = image_links_text
                
                return json.dumps(formatted_response)
            else:
                return json.dumps({
                    "error": result.get("error", "Unknown error during photo maker generation"),
                    "prompt": prompt
                })
        except Exception as e:
            error_msg = f"Error in photo maker: {str(e)}"
            logging.error(error_msg)
            return json.dumps({
                "error": error_msg,
                "prompt": prompt
            })
    
    async def _generate_image_with_refiner(self, args: Dict[str, Any]):
        """
        Execute refined image generation tool function.
        
        Args:
            args: Arguments containing prompt and other parameters
            
        Returns:
            str: JSON string with generated image information
        """
        prompt = args.get("prompt", "")
        num_images = min(int(args.get("num_images", 1)), 4)
        negative_prompt = args.get("negative_prompt", "blurry, distorted, low quality, disfigured")
        
        if not prompt:
            return json.dumps({"error": "No prompt provided for image generation"})
        
        try:
            # Find the current message context for displaying results
            current_task = asyncio.current_task()
            discord_message = None
            user_id = None
            
            for uid, tasks in user_tasks.items():
                if current_task in tasks:
                    user_id = uid
                    # Look for message context in task info
                    for task_info in tasks:
                        if isinstance(task_info, dict) and 'message' in task_info:
                            discord_message = task_info['message']
                            break
                    break
            
            # Process the refined imagegeneration
            result = await self.image_generator.generate_image_with_refiner(
                prompt=prompt,
                num_images=num_images,
                negative_prompt=negative_prompt
            )
            
            if result["success"]:
                # If we have a message context and binary images, send them to Discord
                discord_image_urls = []
                if discord_message and result.get("binary_images"):
                    for i, img_data in enumerate(result["binary_images"]):
                        try:
                            # Send the generated image directly to Discord
                            sent_message = await discord_message.channel.send(
                                f"🖼️ High-quality image generation result ({i+1}/{len(result['binary_images'])}):",
                                file=discord.File(io.BytesIO(img_data), filename=f"refined_image_{i+1}.png")
                            )
                            
                            # Store the URL if available
                            if sent_message.attachments and len(sent_message.attachments) > 0:
                                discord_image_urls.append(sent_message.attachments[0].url)
                        except Exception as e:
                            logging.error(f"Error sending refined image: {str(e)}")
                
                # Format the response
                # Include image URLs in the final message text for history
                image_links = []
                for i, url in enumerate(discord_image_urls):
                    image_links.append(f"Image {i+1}: {url}")
                
                formatted_response = {
                    "message": f"Successfully generated {result['image_count']} high-quality images",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image_urls": result["image_urls"],
                    "discord_image_urls":discord_image_urls,
                    "image_links": "\n".join(image_links) if image_links else ""
                }
                
                return json.dumps(formatted_response)
            else:
                return json.dumps({
                    "error": result.get("error", "Unknown error during refined image generation"),
                    "prompt": prompt
                })
        except Exception as e:
            error_msg = f"Error in refined image generation: {str(e)}"
            logging.error(error_msg)
            return json.dumps({
                "error": error_msg,
                "prompt": prompt
            })



