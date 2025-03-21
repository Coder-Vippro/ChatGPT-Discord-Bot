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
from datetime import datetime, timedelta
from src.utils.openai_utils import process_tool_calls, prepare_messages_for_api, get_tools_for_model
from src.utils.pdf_utils import process_pdf, send_response
from src.utils.code_utils import extract_code_blocks, execute_code
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

# Storage for user data files and charts
user_data_files = {}
user_charts = {}

# Get the Python executable to use for installations
# First check if using a virtual environment
PYTHON_EXECUTABLE = os.environ.get('VIRTUAL_ENV', None)
if PYTHON_EXECUTABLE:
    if os.name == 'nt':  # Windows
        PYTHON_EXECUTABLE = os.path.join(PYTHON_EXECUTABLE, 'Scripts', 'python.exe')
    else:  # Unix/Linux/Mac
        PYTHON_EXECUTABLE = os.path.join(PYTHON_EXECUTABLE, 'bin', 'python')
else:
    # Fallback to sys.executable if not in a virtual environment
    PYTHON_EXECUTABLE = sys.executable

logging.info(f"Using Python executable: {PYTHON_EXECUTABLE}")

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
        
        # Tool functions mapping
        self.tool_functions = {
            "google_search": self._google_search,
            "scrape_webpage": self._scrape_webpage,
            "code_interpreter": self._code_interpreter,
            "generate_image": self._generate_image,
            "set_reminder": self._set_reminder,
            "get_reminders": self._get_reminders
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

        # Install required packages if not available
        if not PANDAS_AVAILABLE:
            self._install_data_packages()
            
    def _install_data_packages(self):
        """Install required data analysis packages if not available"""
        try:
            logging.info("Attempting to install data analysis packages...")
            packages = ["pandas", "numpy", "matplotlib", "seaborn", "openpyxl"]
            
            # Log the Python executable being used
            logging.info(f"Using Python executable for package installation: {PYTHON_EXECUTABLE}")
            
            # First try using pip directly through the Python executable
            for package in packages:
                try:
                    # Use the appropriate Python executable with pip
                    result = subprocess.run(
                        [PYTHON_EXECUTABLE, "-m", "pip", "install", package],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    if result.returncode != 0:
                        logging.warning(f"Failed to install {package}: {result.stderr}")
                        # If we're in a virtualenv and that failed, try using pip directly
                        if os.environ.get('VIRTUAL_ENV'):
                            pip_path = os.path.join(os.environ.get('VIRTUAL_ENV'), 
                                                   'Scripts' if os.name == 'nt' else 'bin', 
                                                   'pip')
                            pip_result = subprocess.run(
                                [pip_path, "install", package],
                                capture_output=True,
                                text=True,
                                check=False
                            )
                            if pip_result.returncode == 0:
                                logging.info(f"Successfully installed {package} using virtualenv pip")
                            else:
                                logging.error(f"Failed to install {package} using virtualenv pip: {pip_result.stderr}")
                    else:
                        logging.info(f"Successfully installed {package}")
                except Exception as e:
                    logging.error(f"Error installing {package}: {str(e)}")
        except Exception as e:
            logging.error(f"Error in package installation process: {str(e)}")

        # Try to import the packages again
        try:
            import pandas as pd
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            logging.info(f"Successfully imported pandas {pd.__version__} after installation")
            global PANDAS_AVAILABLE
            PANDAS_AVAILABLE = True
        except ImportError as e:
            logging.error(f"Still unable to import data libraries after installation: {str(e)}")
            
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
        """
        Execute code interpreter tool function with enhanced data visualization capabilities.
        
        Args:
            args: Arguments containing code, language, input, and visualization flags
        
        Returns:
            str: Code execution result along with the original code
        """
        code = args.get("code", "")
        language = args.get("language", "python")
        input_data = args.get("input", "")
        include_visualization = args.get("include_visualization", False)
        file_path = args.get("file_path", "")
        
        if not code:
            return "Error: No code provided"
        
        # Find user_id from current task to track chart generation
        current_task = asyncio.current_task()
        user_id = None
        
        for uid, tasks in user_tasks.items():
            if current_task in tasks:
                user_id = uid
                break
        
        # Handle data file if specified in the code_interpreter call
        if file_path:
            file_bytes = None
            if user_id in user_data_files:
                file_bytes = user_data_files[user_id].get("bytes")
                
            if not file_bytes:
                return json.dumps({"error": "No data file found. Please upload a CSV or Excel file first."})
            
            # Create a temporary file for the code to use
            temp_file_path = f"temp_data_{user_id}_{int(time.time())}.csv"
            with open(temp_file_path, "wb") as f:
                f.write(file_bytes)
                
            # Include path to data file in input_data
            if input_data:
                input_data += f"\nDATA_FILE_PATH={temp_file_path}"
            else:
                input_data = f"DATA_FILE_PATH={temp_file_path}"
        
        # Check if visualization code is included
        has_visualization = False
        if language.lower() == "python":
            has_visualization = include_visualization or any(x in code for x in [
                "matplotlib", "plt.figure", "plt.plot", "sns.plot", 
                "plt.savefig", "plt.show", "BytesIO", 
                "plotly", "bokeh", "altair"
            ])
            
            # If we have visualization, make sure code produces image bytes
            if has_visualization and "BytesIO" not in code:
                # Modify the code to capture plot output in bytes
                visualization_wrapper = """
import io
from datetime import datetime

# Create BytesIO object to store the image
buffer = io.BytesIO()

# Execute the original code
{0}

# If using matplotlib, save the current figure to the buffer
try:
    import matplotlib.pyplot as plt
    if plt.get_fignums():  # Check if any figures exist
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close('all')
        buffer.seek(0)
        print("\\n[CHART_DATA_START]")
        import base64
        print(base64.b64encode(buffer.getvalue()).decode('utf-8'))
        print("[CHART_DATA_END]")
except Exception as e:
    print(f"Error saving visualization: {{e}}")
"""
                # Indent the original code to fit into our template
                indented_code = "\n".join("    " + line for line in code.split("\n"))
                code = visualization_wrapper.format(indented_code)
        
        # Execute the code
        result = await execute_code(code, language, input_data=input_data)
        
        # Clean up temporary data file if it was created
        if file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logging.error(f"Error removing temporary data file: {str(e)}")
        
        # Check for chart data in output
        chart_image = None
        if has_visualization and "[CHART_DATA_START]" in result and "[CHART_DATA_END]" in result:
            try:
                # Extract base64 encoded image data
                start_marker = "[CHART_DATA_START]"
                end_marker = "[CHART_DATA_END]"
                start_idx = result.find(start_marker) + len(start_marker)
                end_idx = result.find(end_marker)
                if start_idx > 0 and end_idx > start_idx:
                    base64_data = result[start_idx:end_idx].strip()
                    chart_image = base64.b64decode(base64_data)
                    
                    # Store chart in user history if we have a user_id
                    if user_id:
                        chart_id = f"{user_id}_{int(time.time())}"
                        user_charts[chart_id] = {
                            "image": chart_image,
                            "timestamp": datetime.now(),
                            "user_id": user_id
                        }
                        
                        # Remove chart data from result to avoid showing base64 code
                        result = result[:result.find(start_marker)] + "\n[Chart generated successfully]" + result[result.find(end_marker) + len(end_marker):]
            except Exception as e:
                logging.error(f"Error processing chart data: {str(e)}")
        
        # Format the response
        formatted_response = {
            "code": args.get("code", ""),  # Original code, not our modified version
            "language": language,
            "result": result,
            "input_data": input_data if input_data else "None",
            "has_chart": chart_image is not None
        }
        
        if chart_image is not None:
            formatted_response["chart_id"] = chart_id if 'chart_id' in locals() else None
        
        return json.dumps(formatted_response)
    
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
    
    async def _handle_data_file(self, attachment, message, user_id, history, model, start_time):
        """
        Handle data file analysis for CSV and Excel files
        
        Args:
            attachment: The file attachment
            message: The Discord message
            user_id: The user ID
            history: Message history
            model: AI model to use
            start_time: Time when processing started
        """
        try:
            # Read data file
            file_bytes = await attachment.read()
            query = message.content if message.content else "Analyze this data file and provide a summary."
            
            # Save temporary file information
            user_data_files[user_id] = {
                "filename": attachment.filename,
                "bytes": file_bytes,
                "timestamp": datetime.now().timestamp()
            }
            
            # Create a temporary file for analysis
            file_ext = os.path.splitext(attachment.filename)[1].lower()
            temp_file_path = f"temp_data_{user_id}_{int(time.time())}{file_ext}"
            
            # Write the file to disk
            with open(temp_file_path, "wb") as f:
                f.write(file_bytes)
                
            logging.info(f"Saved data file to: {temp_file_path}")
            
            # Try to import required packages
            try:
                import pandas as pd
                import io
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Read the data file directly from disk
                try:
                    if file_ext == '.csv':
                        df = pd.read_csv(temp_file_path)
                    elif file_ext in ['.xlsx', '.xls']:
                        df = pd.read_excel(temp_file_path)
                    
                    # Basic data summary
                    rows, cols = df.shape
                    summary = []
                    summary.append(f"Data File: {attachment.filename}")
                    summary.append(f"Rows: {rows}")
                    summary.append(f"Columns: {cols}")
                    summary.append(f"Column names: {', '.join(df.columns.tolist())}")

                    # Basic statistics for numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    summary.append(f"\nNumeric columns: {len(numeric_cols)}")
                    summary.append(f"Categorical columns: {len(categorical_cols)}")
                    
                    if numeric_cols:
                        summary.append("\nSample statistics:")
                        desc = df[numeric_cols].describe().round(2)
                        summary.append(desc.to_string())
                    
                    data_summary = '\n'.join(summary)
                    
                    # Send data summary
                    await message.channel.send(f"📊 Data File Analysis:\n```\n{data_summary[:1900]}```")
                    
                except Exception as analysis_error:
                    error_msg = f"Error analyzing data file: {str(analysis_error)}"
                    logging.error(error_msg)
                    await message.channel.send(f"📊 Data File Analysis:\nError: {error_msg}")
                    
            except ImportError as import_error:
                # If pandas import fails at this stage, reinstall packages
                logging.error(f"Import error handling data file: {str(import_error)}")
                await message.channel.send(f"📊 Data File Analysis:\nError: Required libraries not available. Installing required packages...")
                
                # Install required packages
                try:
                    # Log which Python executable we're using
                    await message.channel.send(f"Using Python executable: {PYTHON_EXECUTABLE}")
                    
                    for pkg in ["pandas", "numpy", "matplotlib", "seaborn", "openpyxl"]:
                        result = subprocess.run(
                            [PYTHON_EXECUTABLE, "-m", "pip", "install", pkg],
                            capture_output=True,
                            text=True,
                            check=False
                        )
                        
                        if result.returncode != 0:
                            error_msg = f"Failed to install {pkg}: {result.stderr}"
                            logging.warning(error_msg)
                            await message.channel.send(f"⚠️ Warning: {error_msg}")
                        else:
                            logging.info(f"Successfully installed {pkg}")
                            await message.channel.send(f"✅ Installed {pkg}")
                    
                    await message.channel.send(f"📊 Data File Analysis:\nPackages installed. Please try again.")
                    
                    # Clean up temp file
                    try:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                    except Exception as cleanup_error:
                        logging.error(f"Error removing temp file: {str(cleanup_error)}")
                        
                    return
                except Exception as e:
                    logging.error(f"Error installing data analysis packages: {str(e)}")
                    await message.channel.send(f"📊 Data File Analysis:\nError: Failed to install required packages. {str(e)}")
                    
                    # Clean up temp file
                    try:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                    except Exception as cleanup_error:
                        logging.error(f"Error removing temp file: {str(cleanup_error)}")
                        
                    return
            
            # Now use the AI to generate a response with chart creation hint
            ai_prompt = f"I've uploaded a data file {attachment.filename}. {query}\n\nAnalyze this data and create a visualization that best represents the key insights."
            
            # Call API for analysis
            await self._process_text_message(message, user_id, ai_prompt, model, history, start_time)
            
            # Clean up temp file
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            except Exception as cleanup_error:
                logging.error(f"Error removing temp file: {str(cleanup_error)}")
            
        except Exception as e:
            error_msg = f"Error processing data file: {str(e)}"
            logging.error(error_msg)
            await message.channel.send(error_msg)
            return

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
            
            # Send the response text
            await send_response(message.channel, reply)
            
            # Handle charts from code interpreter if present
            if chart_id and chart_id in user_charts:
                try:
                    chart_data = user_charts[chart_id]["image"]
                    chart_filename = f"chart_{chart_id}.png"
                    
                    # Send the chart
                    await message.channel.send(
                        "📊 Chart generated:",
                        file=discord.File(io.BytesIO(chart_data), filename=chart_filename)
                    )
                    
                    # Add the chart to message history with timestamp
                    chart_url = await self._upload_and_get_chart_url(chart_data, chart_filename, message.channel)
                    if chart_url:
                        # Add image to history with timestamp
                        if history[-1]["role"] == "assistant":
                            # If the last message was from the assistant, append the image to it
                            if isinstance(history[-1]["content"], list):
                                history[-1]["content"].append({
                                    "type": "image_url",
                                    "image_url": {"url": chart_url},
                                    "timestamp": datetime.now().isoformat()  # Add timestamp for expiration tracking
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
                        
                        # Save updated history
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
    
    async def _upload_and_get_chart_url(self, chart_data, filename, channel):
        """
        Upload a chart image to Discord and get its URL for history.
        
        Args:
            chart_data: The binary chart data
            filename: The filename to use
            channel: Discord channel to send to
            
        Returns:
            str: URL of the uploaded image or None if failed
        """
        try:
            message = await channel.send(
                "Saving chart to history...",
                file=discord.File(io.BytesIO(chart_data), filename=filename)
            )
            
            # Get the attachment URL from the message
            if message.attachments and len(message.attachments) > 0:
                # Delete the message since we only needed it to get the URL
                await message.delete()
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
                
            # Clean up expired data files
            expired_users = []
            for user_id, data_info in user_data_files.items():
                if datetime.now().timestamp() - data_info.get("timestamp", 0) > max_age_hours * 3600:
                    expired_users.append(user_id)
            
            # Remove expired data files
            for user_id in expired_users:
                del user_data_files[user_id]
                
            if expired_users:
                logging.info(f"Cleaned up {len(expired_users)} expired data files")
                
        except Exception as e:
            logging.error(f"Error cleaning up charts: {str(e)}")
            
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


