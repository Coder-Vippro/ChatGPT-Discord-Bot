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
from datetime import datetime
from src.utils.openai_utils import process_tool_calls, prepare_messages_for_api, get_tools_for_model
from src.utils.pdf_utils import process_pdf, send_response
from src.utils.code_utils import extract_code_blocks, execute_code
from src.utils.data_utils import process_data_file
from src.utils.reminder_utils import ReminderManager

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
        
        # Tool functions mapping
        self.tool_functions = {
            "google_search": self._google_search,
            "scrape_webpage": self._scrape_webpage,
            "code_interpreter": self._code_interpreter,
            "generate_image": self._generate_image,
            "analyze_data": self._analyze_data,
            "set_reminder": self._set_reminder,
            "get_reminders": self._get_reminders
        }
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        
        # Temporary storage for data files
        self.user_data_files = {}
        
        # Create session for HTTP requests
        asyncio.create_task(self._setup_aiohttp_session())
        
        # Register message event handlers
        self._setup_event_handlers()
        
        # Start reminder manager
        self.reminder_manager.start()
        
        # Start chart cleanup task
        self.chart_cleanup_task = asyncio.create_task(self._run_chart_cleanup())
        
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
        Analyze data from recently uploaded CSV/Excel files.
        
        Args:
            args: Analysis parameters
            
        Returns:
            JSON string with analysis results
        """
        query = args.get("query", "")
        visualization_type = args.get("visualization_type", "auto")
        
        if not query:
            return json.dumps({"error": "No analysis request provided"})
        
        # Find user_id from current task to track chart generation
        current_task = asyncio.current_task()
        user_id = None
        
        for uid, tasks in user_tasks.items():
            if current_task in tasks:
                user_id = uid
                break
        
        if not user_id:
            logging.warning("Could not identify user_id for analyze_data call")
        
        # Check if there are any data files
        active_data_files = {}
        for file_user_id, data_info in self.user_data_files.items():
            # Only consider files uploaded in the last hour
            if datetime.now().timestamp() - data_info.get("timestamp", 0) < 3600:  # 1 hour
                active_data_files[file_user_id] = data_info
        
        if not active_data_files:
            return json.dumps({
                "error": "No data files found. Please upload a CSV or Excel file first."
            })
            
        # Use the most recent data file
        latest_user_id = max(active_data_files.keys(), key=lambda k: active_data_files[k]["timestamp"])
        data_info = active_data_files[latest_user_id]
        
        try:
            file_bytes = data_info["bytes"]
            filename = data_info["filename"]
            
            # Process data file with specified visualization type if provided
            modified_query = query
            if visualization_type and visualization_type != "auto":
                modified_query = f"{query} [Use {visualization_type} chart]"
            
            # Analyze data using data_utils - pass user_id to prevent duplicate chart creation
            summary, chart_image, metadata = await process_data_file(
                file_bytes, 
                filename, 
                modified_query, 
                str(user_id) if user_id else None
            )
            
            if chart_image and "chart_filename" in metadata:
                chart_path = metadata["chart_filename"]
                chart_basename = os.path.basename(chart_path)
                
                # Return results with chart file path
                return json.dumps({
                    "summary": summary,
                    "has_chart": True,
                    "chart_filename": chart_path,
                    "chart_display_name": chart_basename,
                    "metadata": {
                        "filename": metadata.get("filename", ""),
                        "rows": metadata.get("rows", 0),
                        "columns": metadata.get("columns", 0),
                        "chart_type": metadata.get("chart_type", ""),
                        "timestamp": metadata.get("timestamp", ""),
                        "request_id": metadata.get("request_id", "")
                    }
                })
            else:
                return json.dumps({
                    "summary": summary,
                    "has_chart": False,
                    "metadata": {
                        "filename": metadata.get("filename", ""),
                        "rows": metadata.get("rows", 0),
                        "columns": metadata.get("columns", 0),
                        "timestamp": metadata.get("timestamp", "")
                    }
                })
        except Exception as e:
            logging.error(f"Error analyzing data: {str(e)}")
            return json.dumps({"error": f"Error analyzing data: {str(e)}"})
    
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
        user_tasks[user_id].append(task)
        
        # Use done callbacks to clean up and handle errors
        def task_done_callback(task):
            # Remove task from tracking list
            if user_id in user_tasks and task in user_tasks[user_id]:
                user_tasks[user_id].remove(task)
            
            # Check for exceptions that weren't handled
            if task.done() and not task.cancelled():
                try:
                    task.result()
                except Exception as e:
                    logging.error(f"Unhandled task exception: {str(e)}")
        
        # Add the callback
        task.add_done_callback(task_done_callback)
    
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
                        
                        # Handle data files (CSV, Excel)
                        elif any(attachment.filename.lower().endswith(ext) for ext in DATA_FILE_EXTENSIONS):
                            try:
                                # Read data file
                                file_bytes = await attachment.read()
                                query = message.content if message.content else "Analyze this data file and provide a summary."
                                
                                # Save temporary file information
                                self.user_data_files[user_id] = {
                                    "filename": attachment.filename,
                                    "bytes": file_bytes,
                                    "timestamp": datetime.now().timestamp()
                                }
                                
                                # Analyze data for basic info ONLY - don't generate chart yet
                                # We'll add a flag to prevent chart generation here
                                summary, _, metadata = await process_data_file(
                                    file_bytes, 
                                    attachment.filename, 
                                    query + " [no_chart]",  # Special flag to skip chart creation
                                    str(user_id)
                                )
                                
                                # Send analysis text
                                await message.channel.send(summary[:2000])  # Discord message length limit
                                
                                # More detailed analysis with AI - let the analyze_data tool generate the chart
                                ai_prompt = f"I've uploaded a data file {attachment.filename}. {query}\n\nAnalyze this data and create a visualization that best represents the key insights. Be sure to consider the most appropriate chart type based on the data structure."
                                
                                # Call API for analysis - this will generate the chart through the tool
                                await self._process_text_message(message, user_id, ai_prompt, model, history, start_time)
                                return
                                
                            except Exception as e:
                                error_msg = f"Error processing data file: {str(e)}"
                                logging.error(error_msg)
                                await message.channel.send(error_msg)
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
                                    # Fallback to other common encodings
                                    try:
                                        file_content = file_bytes.decode('latin-1')
                                    except:
                                        file_content = file_bytes.decode('utf-8', errors='replace')
                                
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
                "top_p": 0.7 if model in ["gpt-4o", "gpt-4o-mini"] else 1,
                "timeout": 60  # Add an explicit timeout
            }
            
            # Add tools if using a supported model
            if use_tools:
                api_params["tools"] = get_tools_for_model()
            
            # Initialize variables to track tool responses
            image_generation_used = False
            chart_filename = None
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
                    
                if tool_messages.get("analyze_data"):
                    await message.channel.send("📊 Analyzing data...")
                    
                if tool_messages.get("set_reminder") or tool_messages.get("get_reminders"):
                    await message.channel.send("📅 Processing reminders...")
                
                if not tool_messages:                        
                    await message.channel.send("🤔 Processing...")
                
                # Process any tool calls and get the updated messages
                tool_calls_processed, updated_messages = await process_tool_calls(
                    self.client, 
                    response, 
                    messages_for_api, 
                    self.tool_functions
                )
                
                # Process tool responses to extract important data (images, charts)
                if updated_messages:
                    # Look for image generation and data analysis tool responses
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
                        
                        elif msg.get('role') == 'tool' and msg.get('name') == 'analyze_data':
                            try:
                                content = msg.get('content', '')
                                if isinstance(content, str) and '{' in content:
                                    data = json.loads(content)
                                    if data.get('has_chart') and 'chart_filename' in data:
                                        chart_filename = data.get('chart_filename')
                            except Exception as e:
                                logging.error(f"Error parsing chart data: {str(e)}")
                
                # If tool calls were processed, make another API call with the updated messages
                if tool_calls_processed and updated_messages:
                    response = await self._retry_api_call(lambda: self.client.chat.completions.create(
                        model=model,
                        messages=updated_messages,
                        temperature=0.3 if model in ["gpt-4o", "gpt-4o-mini"] else 1,
                        timeout=60
                    ))
            
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
                    
                    # Only keep a reasonable amount of history
                    if len(new_history) > 20:  # Truncate to last 20 messages if too long
                        new_history = new_history[:1] + new_history[-19:]  # Keep system message + last 19 messages
                        
                    await self.db.save_history(user_id, new_history)
                else:
                    # For models with system prompt support, just append to regular history
                    history.append({"role": "assistant", "content": reply})
                    
                    # Only keep a reasonable amount of history
                    if len(history) > 20:  # Truncate to last 20 messages if too long
                        history = history[:1] + history[-19:]  # Keep system message + last 19 messages
                        
                    await self.db.save_history(user_id, history)
            
            # Decide how to handle the response based on the tools that were used
            # Don't send any additional image links - just send the model's response
            await send_response(message.channel, reply)
            
            # Handle charts from data analysis if present
            if chart_filename:
                try:
                    with open(chart_filename, "rb") as f:
                        chart_data = f.read()
                    await message.channel.send(
                        "Chart from data analysis:",
                        file=discord.File(io.BytesIO(chart_data), filename=chart_filename)
                    )
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
                
            # Parse time
            remind_at = await self.reminder_manager.parse_time(time_str)
            
            if not remind_at:
                return json.dumps({
                    "error": f"Could not parse time '{time_str}'. Please use formats like '30m', '2h', '1d', 'tomorrow', or '15:00'"
                })
                
            # Save reminder
            reminder = await self.reminder_manager.add_reminder(user_id, content, remind_at)
            
            return json.dumps({
                "success": True,
                "content": content,
                "time": remind_at.strftime("%Y-%m-%d %H:%M:%S"),
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
                if resp.status == 200:
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
    
    async def _code_interpreter(self, args: Dict[str, Any]):
        """
        Execute code interpreter tool function.
        
        Args:
            args: Arguments containing code, language and optional input
        
        Returns:
            str: Code execution result along with the original code
        """
        code = args.get("code", "")
        language = args.get("language", "python")
        input_data = args.get("input", "")
        
        if not code:
            return "Error: No code provided"
        
        # Execute the code
        result = await execute_code(code, language, input_data=input_data)
        
        # Format the response to include both code and result
        formatted_response = {
            "code": code,
            "language": language,
            "result": result,
            "input_data": input_data if input_data else "None"
        }
        
        # Return a formatted JSON response with both code and result
        return json.dumps(formatted_response)
    
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
            
    async def _run_chart_cleanup(self):
        """Run periodic chart cleanup to remove old chart files"""
        from src.utils.data_utils import cleanup_old_charts
        
        try:
            while True:
                # Cleanup every 30 minutes but delete only charts older than 1 hour
                await cleanup_old_charts(max_age_hours=1)
                await asyncio.sleep(1800)  # 30 minutes
        except asyncio.CancelledError:
            logging.info("Chart cleanup task was cancelled")
            raise
        except Exception as e:
            logging.error(f"Error in chart cleanup task: {str(e)}")
            
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


