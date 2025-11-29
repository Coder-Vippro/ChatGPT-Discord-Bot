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
from src.config.config import PDF_ALLOWED_MODELS, MODEL_TOKEN_LIMITS, DEFAULT_TOKEN_LIMIT, DEFAULT_MODEL

# Global task and rate limiting tracking
user_tasks = {}
user_last_request = {}
RATE_LIMIT_WINDOW = 5  # seconds
MAX_REQUESTS = 3  # max requests per window

# Model pricing per 1M tokens (in USD)
MODEL_PRICING = {
    "openai/gpt-4o": {"input": 5.00, "output": 20.00},
    "openai/gpt-4o-mini": {"input": 0.60, "output": 2.40},
    "openai/gpt-4.1": {"input": 2.00, "output": 8.00},
    "openai/gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "openai/gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "openai/gpt-5": {"input": 1.25, "output": 10.00},
    "openai/gpt-5-mini": {"input": 0.25, "output": 2.00},
    "openai/gpt-5-nano": {"input": 0.05, "output": 0.40},
    "openai/gpt-5-chat": {"input": 1.25, "output": 10.00},
    "openai/o1-preview": {"input": 15.00, "output": 60.00},
    "openai/o1-mini": {"input": 1.10, "output": 4.40},
    "openai/o1": {"input": 15.00, "output": 60.00},
    "openai/o3-mini": {"input": 1.10, "output": 4.40},
    "openai/o3": {"input": 2.00, "output": 8.00},
    "openai/o4-mini": {"input": 2.00, "output": 8.00}
}

# File extensions that should be treated as text files
TEXT_FILE_EXTENSIONS = [
    '.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm', '.css', 
    '.js', '.py', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.php',
    '.rb', '.pl', '.sh', '.bat', '.ps1', '.sql', '.yaml', '.yml',
    '.ini', '.cfg', '.conf', '.log', '.ts', '.jsx', '.tsx', '.vue', 
    '.go', '.rs', '.swift', '.kt', '.kts', '.dart', '.lua'
]

# File extensions for data files (ALL types - Python can handle almost anything!)
# With code_interpreter, we support 200+ file types
DATA_FILE_EXTENSIONS = [
    # Tabular data
    '.csv', '.tsv', '.tab', '.xlsx', '.xls', '.xlsm', '.xlsb', '.ods', '.numbers',
    # Structured data
    '.json', '.jsonl', '.ndjson', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.properties', '.env',
    # Database
    '.db', '.sqlite', '.sqlite3', '.sql', '.mdb', '.accdb',
    # Scientific/Binary
    '.parquet', '.feather', '.arrow', '.hdf', '.hdf5', '.h5', '.pickle', '.pkl',
    '.joblib', '.npy', '.npz', '.mat', '.sav', '.dta', '.sas7bdat', '.xpt', '.rda', '.rds',
    # Text/Code
    '.txt', '.text', '.log', '.out', '.err', '.md', '.markdown', '.rst', '.tex', '.adoc', '.org',
    '.py', '.pyw', '.ipynb', '.r', '.R', '.rmd', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp',
    '.h', '.hpp', '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.m', '.pl', '.sh',
    '.bash', '.zsh', '.ps1', '.lua', '.jl', '.nim', '.asm', '.html', '.htm', '.css', '.scss', '.sass',
    '.vue', '.svelte',
    # Geospatial
    '.geojson', '.shp', '.shx', '.dbf', '.kml', '.kmz', '.gpx', '.gml',
    # Scientific
    '.fits', '.fts', '.dicom', '.dcm', '.nii', '.vtk', '.stl', '.obj', '.ply',
    # Other data
    '.avro', '.orc', '.protobuf', '.pb', '.msgpack', '.bson', '.cbor', '.pcap', '.pcapng',
    # Documents (for text extraction)
    '.pdf', '.doc', '.docx', '.odt', '.rtf', '.epub', '.mobi',
    # Audio/Video (for metadata analysis)
    '.mp3', '.wav', '.flac', '.ogg', '.aac', '.m4a', '.wma', '.opus', '.aiff',
    '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg',
    # Archives (Python can extract these)
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar', '.tgz', '.tbz', '.lz', '.lzma', '.zst',
    # Binary (generic - Python can read as bytes)
    '.bin', '.dat'
]

# File extensions for image files (should never be processed as data)
IMAGE_FILE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg', '.tiff', '.ico']

# Note: Removed global user_data_files and user_charts dictionaries for memory optimization
# Data files are now processed immediately and cleaned up

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
        
        # Memory-optimized user data tracking (with TTL)
        self.user_data_files = {}  # Will be cleaned up periodically
        self.user_charts = {}      # Will be cleaned up periodically
        self.max_user_files = 20   # Limit concurrent user files
        
        # Store latest image URL per user (in-memory, refreshed from attachments)
        self.user_latest_image_url = {}
        
        # Tool mapping for API integration
        self.tool_mapping = {
            "google_search": self._google_search,
            "scrape_webpage": self._scrape_webpage,
            "execute_python_code": self._execute_python_code,
            "generate_image": self._generate_image,
            "edit_image": self._edit_image,
            "remove_background": self._remove_background,
            "set_reminder": self._set_reminder,
            "get_reminders": self._get_reminders,
            "enhance_prompt": self._enhance_prompt,
            "image_to_text": self._image_to_text,
            "upscale_image": self._upscale_image,
            "photo_maker": self._photo_maker,
            "generate_image_with_refiner": self._generate_image_with_refiner
        }
        
        # Thread pool for CPU-bound tasks (balanced for performance)
        import multiprocessing
        max_workers = min(4, multiprocessing.cpu_count())  # Increased to 4 for better concurrency
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
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
        
        # Initialize tiktoken encoder for internal operations (trimming, estimation)
        try:
            import tiktoken
            self.token_encoder = tiktoken.get_encoding("o200k_base")
            logging.info("Tiktoken encoder initialized for internal operations")
        except Exception as e:
            logging.warning(f"Failed to initialize tiktoken encoder: {e}")
            self.token_encoder = None
    
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
    
    async def _get_latest_image_url_from_db(self, user_id: int) -> str:
        """Get the latest valid image URL from user's history in database"""
        try:
            # Get history from database (already filtered for expired images)
            history = await self.db.get_history(user_id)
            
            # Find the latest image URL by iterating in reverse
            for msg in reversed(history):
                content = msg.get('content')
                if isinstance(content, list):
                    for item in reversed(content):
                        if item.get('type') == 'image_url':
                            image_url_data = item.get('image_url', {})
                            url = image_url_data.get('url') if isinstance(image_url_data, dict) else None
                            if url:
                                logging.info(f"Found latest image URL from database: {url[:80]}...")
                                return url
            return None
        except Exception as e:
            logging.error(f"Error getting latest image URL from database: {e}")
            return None
    
    def _count_tokens_with_tiktoken(self, text: str) -> int:
        """Count tokens using tiktoken encoder for internal operations."""
        if self.token_encoder is None:
            # Fallback estimation if tiktoken is not available
            return len(text) // 4
        
        try:
            return len(self.token_encoder.encode(text))
        except Exception as e:
            logging.warning(f"Error counting tokens with tiktoken: {e}")
            return len(text) // 4
    
    def _get_system_prompt_with_time(self) -> str:
        """
        Get the system prompt with current time and timezone information.
        
        Returns:
            str: The system prompt with current datetime
        """
        from src.config.config import NORMAL_CHAT_PROMPT, TIMEZONE
        
        try:
            # Try using zoneinfo (Python 3.9+)
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(TIMEZONE)
            current_time = datetime.now(tz)
            time_str = current_time.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
        except ImportError:
            # Fallback: try pytz if zoneinfo is not available
            try:
                import pytz
                tz = pytz.timezone(TIMEZONE)
                current_time = datetime.now(tz)
                time_str = current_time.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
            except Exception as e:
                logging.warning(f"Error getting timezone with pytz: {e}, falling back to UTC")
                current_time = datetime.utcnow()
                time_str = current_time.strftime("%A, %B %d, %Y at %I:%M:%S %p UTC")
        except Exception as e:
            # Final fallback to UTC
            logging.warning(f"Error getting timezone info: {e}, falling back to UTC")
            current_time = datetime.utcnow()
            time_str = current_time.strftime("%A, %B %d, %Y at %I:%M:%S %p UTC")
        
        # Prepend current time to the system prompt
        time_prefix = f"Current date and time: {time_str}\n\n"
        return time_prefix + NORMAL_CHAT_PROMPT
    
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
        """Create a memory-optimized aiohttp session"""
        if self.aiohttp_session is None or self.aiohttp_session.closed:
            self.aiohttp_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120),  # Reduced timeout
                connector=aiohttp.TCPConnector(
                    limit=8,  # Reduced from 20 to 8
                    ttl_dns_cache=600,  # Increased DNS cache for efficiency
                    enable_cleanup_closed=True,  # Enable connection cleanup
                    keepalive_timeout=30  # Shorter keepalive
                )
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
        """
        Handle Python code execution through code_interpreter
        All user files are automatically accessible via load_file(file_id)
        """
        try:
            # Find user_id from current task context
            user_id = args.get("user_id")
            if not user_id:
                user_id = self._find_user_id_from_current_task()
            
            # Get the Discord message to send code execution display
            discord_message = self._get_discord_message_from_current_task()
            
            # Get ALL user files from database (not just in-memory cache)
            user_files = []
            if user_id:
                try:
                    db_files = await self.db.get_user_files(user_id)
                    logging.info(f"[DEBUG] Fetched {len(db_files) if db_files else 0} files from DB for user {user_id}")
                    if db_files:
                        for f in db_files:
                            logging.info(f"[DEBUG] DB file: {f.get('file_id', 'NO_ID')} - {f.get('filename', 'NO_NAME')}")
                    user_files = [f['file_id'] for f in db_files if 'file_id' in f]
                    if user_files:
                        logging.info(f"Code execution will have access to {len(user_files)} file(s) for user {user_id}: {user_files}")
                    else:
                        logging.warning(f"[DEBUG] No files found in database for user {user_id}")
                except Exception as e:
                    logging.warning(f"Could not fetch user files: {e}")
            
            # Extract code and packages for display
            code_to_execute = args.get("code", "")
            install_packages = args.get("install_packages", [])
            packages_to_install = install_packages  # For display purposes
            input_data = args.get("input_data", "")  # For display purposes
            
            # Import and call unified code interpreter
            from src.utils.code_interpreter import execute_code
            
            # Execute code with file access
            execute_result = await execute_code(
                code=code_to_execute,
                user_id=user_id,
                user_files=user_files,  # Pass all file_ids - code_interpreter handles load_file()
                install_packages=install_packages,
                db_handler=self.db
            )
            
            # Display the executed code information in Discord
            if discord_message and code_to_execute:
                # Check user's tool display preference
                show_execution_details = await self.db.get_user_tool_display(user_id) if user_id else False
                
                if show_execution_details:
                    try:
                        # Clean up the code display (remove file context comments)
                        code_lines = code_to_execute.split('\n')
                        clean_code_lines = []
                        for line in code_lines:
                            if not (line.strip().startswith('# Data file available:') or 
                                   line.strip().startswith('# File path:') or 
                                   line.strip().startswith('# You can access this file using:')):
                                clean_code_lines.append(line)
                        
                        clean_code = '\n'.join(clean_code_lines).strip()
                        
                        # Check if code is too long for Discord message (3000 chars limit)
                        if len(clean_code) > 3000:
                            # Send code as file attachment
                            code_file = discord.File(
                                io.StringIO(clean_code), 
                                filename="executed_code.py"
                            )
                            
                            # Create display text without code
                            execution_display = "**🐍 Python Code Execution**\n\n"
                            
                            # Show packages to install if any
                            if packages_to_install:
                                execution_display += f"**📦 Installing packages:** {', '.join(packages_to_install)}\n\n"
                            
                            # Show input data if any
                            if input_data:
                                execution_display += "**📥 Input:**\n```\n"
                                execution_display += input_data[:500]  # Limit input length
                                if len(input_data) > 500:
                                    execution_display += "\n... (input truncated)"
                                execution_display += "\n```\n\n"
                            
                            execution_display += "**💻 Code:** *Attached as file (too long to display)*\n\n"
                            
                            # Show the output
                            if execute_result and execute_result.get("success"):
                                output = execute_result.get("output", "")
                                # Remove package installation info from output if it exists
                                if output and "Installed packages:" in output:
                                    lines = output.split('\n')
                                    output = '\n'.join(lines[2:]) if len(lines) > 2 else ""
                                
                                if output and output.strip():
                                    execution_display += "**📤 Output:**\n```\n"
                                    # Calculate remaining space (2000 - current length - markdown)
                                    remaining = 1900 - len(execution_display)
                                    if remaining > 100:
                                        execution_display += output[:remaining]
                                        if len(output) > remaining:
                                            execution_display += "\n... (output truncated)"
                                    else:
                                        execution_display += "(output too long)"
                                    execution_display += "\n```"
                                else:
                                    execution_display += "**📤 Output:** *(No output)*"
                            else:
                                error_msg = execute_result.get("error", "Unknown error") if execute_result else "Execution failed"
                                # Calculate remaining space
                                remaining = 1900 - len(execution_display)
                                if remaining > 100:
                                    execution_display += f"**❌ Error:**\n```\n{error_msg[:remaining]}\n```"
                                    if len(error_msg) > remaining:
                                        execution_display += "*(Error message truncated)*"
                                else:
                                    execution_display += "**❌ Error:** *(Error too long - see logs)*"
                            
                            # Final safety check: ensure total length < 2000
                            if len(execution_display) > 1990:
                                execution_display = execution_display[:1980] + "\n...(truncated)"
                            
                            # Send with file attachment
                            await discord_message.channel.send(execution_display, file=code_file)
                        else:
                            # Use normal display for shorter code
                            execution_display = "**🐍 Python Code Execution**\n\n"
                            
                            # Show packages to install if any
                            if packages_to_install:
                                execution_display += f"**📦 Installing packages:** {', '.join(packages_to_install)}\n\n"
                            
                            # Show input data if any
                            if input_data:
                                execution_display += "**📥 Input:**\n```\n"
                                execution_display += input_data[:500]  # Limit input length
                                if len(input_data) > 500:
                                    execution_display += "\n... (input truncated)"
                                execution_display += "\n```\n\n"
                            
                            # Show the actual code
                            execution_display += "**💻 Code:**\n```python\n"
                            execution_display += clean_code
                            execution_display += "\n```\n\n"
                            
                            # Show the output
                            if execute_result and execute_result.get("success"):
                                output = execute_result.get("output", "")
                                # Remove package installation info from output if it exists
                                if output and "Installed packages:" in output:
                                    lines = output.split('\n')
                                    output = '\n'.join(lines[2:]) if len(lines) > 2 else ""
                                
                                if output and output.strip():
                                    execution_display += "**📤 Output:**\n```\n"
                                    # Calculate remaining space (2000 - current length - markdown)
                                    remaining = 1900 - len(execution_display)
                                    if remaining > 100:
                                        execution_display += output[:remaining]
                                        if len(output) > remaining:
                                            execution_display += "\n... (output truncated)"
                                    else:
                                        execution_display += "(output too long)"
                                    execution_display += "\n```"
                                else:
                                    execution_display += "**📤 Output:** *(No output)*"
                            else:
                                error_msg = execute_result.get("error", "Unknown error") if execute_result else "Execution failed"
                                # Calculate remaining space
                                remaining = 1900 - len(execution_display)
                                if remaining > 100:
                                    execution_display += f"**❌ Error:**\n```\n{error_msg[:remaining]}\n```"
                                    if len(error_msg) > remaining:
                                        execution_display += "*(Error message truncated)*"
                                else:
                                    execution_display += "**❌ Error:** *(Error too long - see logs)*"
                            
                            # Final safety check: ensure total length < 2000
                            if len(execution_display) > 1990:
                                execution_display = execution_display[:1980] + "\n...(truncated)"
                            
                            # Send the execution display to Discord as a separate message
                            await discord_message.channel.send(execution_display)
                        
                    except Exception as e:
                        logging.error(f"Error displaying code execution: {str(e)}")
            
            # Handle generated files (NEW unified approach)
            if execute_result and execute_result.get("generated_files"):
                generated_files = execute_result["generated_files"]
                
                # Send summary if multiple files
                if len(generated_files) > 1 and discord_message:
                    summary = f"📎 **Generated {len(generated_files)} file(s):**\n"
                    for gf in generated_files:
                        size_kb = gf.get('size', 0) / 1024
                        file_type = gf.get('type', 'file')
                        summary += f"• `{gf['filename']}` ({file_type}, {size_kb:.1f} KB)\n"
                    await discord_message.channel.send(summary)
                
                # Send each generated file
                for gf in generated_files:
                    try:
                        file_data = gf.get("data")
                        filename = gf.get("filename", "output.txt")
                        file_type = gf.get("type", "file")
                        file_id = gf.get("file_id", "")
                        
                        if file_data and discord_message:
                            # File type emoji mapping
                            emoji_map = {
                                "image": "🖼️",
                                "data": "📊",
                                "text": "📝",
                                "structured": "📋",
                                "html": "🌐",
                                "pdf": "📄",
                                "code": "💻",
                                "archive": "📦",
                                "file": "📎"
                            }
                            emoji = emoji_map.get(file_type, "📎")
                            
                            # Create Discord file and send
                            file_bytes = io.BytesIO(file_data)
                            discord_file = discord.File(file_bytes, filename=filename)
                            
                            caption = f"{emoji} `{filename}`"
                            if file_id:
                                caption += f" (ID: `{file_id}`)"
                            
                            # Send the file
                            msg = await discord_message.channel.send(caption, file=discord_file)
                            
                            # For images, extract URL from the sent message for history
                            if file_type == "image" and msg.attachments:
                                chart_url = msg.attachments[0].url
                                execute_result.setdefault("chart_urls", []).append(chart_url)
                            
                    except Exception as e:
                        logging.error(f"Error sending generated file {gf.get('filename', 'unknown')}: {str(e)}")
                        traceback.print_exc()
            
            # Legacy: Handle old visualizations format (for backward compatibility)
            elif execute_result and execute_result.get("visualizations"):
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
            
            # Get the Discord message to send code execution display
            discord_message = self._get_discord_message_from_current_task()
            
            # Import and call unified code interpreter for data analysis
            from src.utils.code_interpreter import execute_code, upload_discord_attachment
            
            # Get file_path from args first
            file_path = args.get("file_path", "")
            analysis_type = args.get("analysis_type", "")
            custom_analysis = args.get("custom_analysis", "")
            
            # Check if this is a Discord attachment - upload it to code interpreter
            if file_path and not file_path.startswith('/tmp/bot_code_interpreter'):
                # This is an old-style file path, try to upload to new system
                try:
                    # Read the file
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    
                    # Upload to new system
                    filename = os.path.basename(file_path)
                    from src.utils.code_interpreter import upload_file
                    upload_result = await upload_file(
                        user_id=user_id,
                        file_data=file_data,
                        filename=filename,
                        file_type='csv' if file_path.endswith('.csv') else 'excel',
                        db_handler=self.db
                    )
                    
                    if upload_result['success']:
                        # Get file_id for new load_file() system
                        file_id = upload_result['file_id']
                        file_path = upload_result['file_path']
                        logging.info(f"Migrated file to code interpreter: {file_path} (ID: {file_id})")
                except Exception as e:
                    logging.warning(f"Could not migrate file to code interpreter: {e}")
                    file_id = None
            else:
                # File is already in new system, get file_id from args
                file_id = args.get("file_id")
            
            # Generate analysis code based on the request
            # Detect file type
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Use load_file() if we have a file_id, otherwise use direct path
            if file_id:
                if file_ext in ['.xlsx', '.xls']:
                    load_statement = f"df = pd.read_excel(load_file('{file_id}'))"
                elif file_ext == '.json':
                    load_statement = f"df = pd.read_json(load_file('{file_id}'))"
                elif file_ext == '.parquet':
                    load_statement = f"df = pd.read_parquet(load_file('{file_id}'))"
                else:  # Default to CSV
                    load_statement = f"df = pd.read_csv(load_file('{file_id}'))"
            else:
                # Fallback to direct path for legacy support
                if file_ext in ['.xlsx', '.xls']:
                    load_statement = f"df = pd.read_excel('{file_path}')"
                elif file_ext == '.json':
                    load_statement = f"df = pd.read_json('{file_path}')"
                elif file_ext == '.parquet':
                    load_statement = f"df = pd.read_parquet('{file_path}')"
                else:  # Default to CSV
                    load_statement = f"df = pd.read_csv('{file_path}')"
            
            analysis_code = f"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data file
{load_statement}

# Display basic info
print("=== Data Overview ===")
print(f"Shape: {{df.shape}}")
print(f"\\nColumns: {{df.columns.tolist()}}")
print(f"\\nData Types:\\n{{df.dtypes}}")
print(f"\\nMissing Values:\\n{{df.isnull().sum()}}")

# Display statistical summary
print("\\n=== Statistical Summary ===")
print(df.describe())

# Custom analysis based on type
"""
            if analysis_type == "summary":
                analysis_code += """
print("\\n=== First Few Rows ===")
print(df.head(10))
"""
            elif analysis_type == "correlation" and custom_analysis:
                analysis_code += f"""
# Correlation analysis
print("\\n=== Correlation Analysis ===")
{custom_analysis}
"""
            elif custom_analysis:
                analysis_code += f"""
# Custom analysis
{custom_analysis}
"""
            
            # Execute the analysis code
            # Pass file_id as user_files if available
            user_files_for_analysis = [file_id] if file_id else []
            
            result = await execute_code(
                code=analysis_code,
                user_id=user_id,
                user_files=user_files_for_analysis,
                db_handler=self.db
            )
            
            # Display the generated code if available
            if discord_message and analysis_code:
                try:
                    generated_code = analysis_code
                    
                    # Check if code is too long for Discord message (3000 chars limit)
                    if len(generated_code) > 3000:
                        # Send code as file attachment
                        code_file = discord.File(
                            io.StringIO(generated_code), 
                            filename="data_analysis_code.py"
                        )
                        
                        # Create display text without code
                        execution_display = "**📊 Data Analysis Execution**\n\n"
                        
                        # Show the file being analyzed
                        file_path = args.get("file_path", "")
                        if file_path:
                            filename = os.path.basename(file_path)
                            execution_display += f"**📁 Analyzing file:** `{filename}`\n\n"
                        
                        # Show the analysis type if specified
                        analysis_type = args.get("analysis_type", "")
                        custom_analysis = args.get("custom_analysis", "")
                        if analysis_type:
                            execution_display += f"**🔍 Analysis type:** {analysis_type}\n\n"
                        if custom_analysis:
                            execution_display += f"**📝 Custom request:** {custom_analysis}\n\n"
                        
                        execution_display += "**💻 Generated Code:** *Attached as file (too long to display)*\n\n"
                        
                        # Show the output
                        if result.get("success"):
                            output = result.get("output", "")
                            if output and output.strip():
                                execution_display += "**� Analysis Results:**\n```\n"
                                execution_display += output[:2000]  # More space for output when code is attached
                                if len(output) > 2000:
                                    execution_display += "\n... (output truncated)"
                                execution_display += "\n```"
                            else:
                                execution_display += "**📊 Analysis Results:** *(No text output - check for visualizations below)*"
                        else:
                            error_msg = result.get("error", "Unknown error")
                            execution_display += f"**❌ Error:**\n```\n{error_msg[:1000]}\n```"
                            if len(error_msg) > 1000:
                                execution_display += "*(Error message truncated)*"
                        
                        # Send with file attachment
                        await discord_message.channel.send(execution_display, file=code_file)
                    else:
                        # Use normal display for shorter code
                        execution_display = "**📊 Data Analysis Execution**\n\n"
                        
                        # Show the file being analyzed
                        file_path = args.get("file_path", "")
                        if file_path:
                            filename = os.path.basename(file_path)
                            execution_display += f"**📁 Analyzing file:** `{filename}`\n\n"
                        
                        # Show the analysis type if specified
                        analysis_type = args.get("analysis_type", "")
                        custom_analysis = args.get("custom_analysis", "")
                        if analysis_type:
                            execution_display += f"**🔍 Analysis type:** {analysis_type}\n\n"
                        if custom_analysis:
                            execution_display += f"**📝 Custom request:** {custom_analysis}\n\n"
                        
                        # Show the generated code
                        execution_display += "**💻 Generated Code:**\n```python\n"
                        execution_display += generated_code
                        execution_display += "\n```\n\n"
                        
                        # Show the output
                        if result.get("success"):
                            output = result.get("output", "")
                            if output and output.strip():
                                execution_display += "**📊 Analysis Results:**\n```\n"
                                execution_display += output[:1000]  # Limit output length for Discord
                                if len(output) > 1000:
                                    execution_display += "\n... (output truncated)"
                                execution_display += "\n```"
                            else:
                                execution_display += "**📊 Analysis Results:** *(No text output - check for visualizations below)*"
                        else:
                            error_msg = result.get("error", "Unknown error")
                            execution_display += f"**❌ Error:**\n```\n{error_msg[:800]}\n```"
                            if len(error_msg) > 800:
                                execution_display += "*(Error message truncated)*"
                        
                        # Send the execution display to Discord as a separate message
                        await discord_message.channel.send(execution_display)
                    
                except Exception as e:
                    logging.error(f"Error displaying data analysis code: {str(e)}")
            
            # If there are visualizations, handle them for Discord
            if result and result.get("visualizations"):
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
        Download and save file to code_interpreter system with automatic cleanup
        Respects FILE_EXPIRATION_HOURS and MAX_FILES_PER_USER from .env
        
        Args:
            attachment: The Discord file attachment
            user_id: User ID for tracking
            
        Returns:
            Dict with file info including file_id for code_interpreter access
        """
        try:
            # Import code_interpreter's upload function
            from src.utils.code_interpreter import upload_discord_attachment
            from src.config.config import MAX_FILES_PER_USER
            
            # Check user's current file count (enforce limit)
            user_files = await self.db.get_user_files(user_id)
            if len(user_files) >= MAX_FILES_PER_USER:
                # Delete oldest file to make room
                oldest_file = min(user_files, key=lambda f: f.get('uploaded_at', datetime.min))
                from src.utils.code_interpreter import delete_file
                await delete_file(oldest_file['file_id'], user_id, self.db)
                logging.info(f"Deleted oldest file {oldest_file['file_id']} for user {user_id} (limit: {MAX_FILES_PER_USER})")
            
            # Upload to code_interpreter (handles expiration automatically)
            result = await upload_discord_attachment(
                attachment=attachment,
                user_id=user_id,
                db_handler=self.db
            )
            
            if not result['success']:
                raise Exception(result.get('error', 'Upload failed'))
            
            # Extract file info from result
            metadata = result.get('metadata', {})
            file_info = {
                "file_id": result['file_id'],
                "filename": metadata.get('filename', attachment.filename),
                "file_type": metadata.get('file_type', 'unknown'),
                "file_size": metadata.get('file_size', 0),
                "file_path": metadata.get('file_path', ''),
                "expires_at": metadata.get('expires_at'),
                "timestamp": datetime.now()
            }
            
            logging.info(
                f"Uploaded file for user {user_id}: {file_info['filename']} "
                f"(ID: {file_info['file_id']}, Type: {file_info['file_type']}, "
                f"Size: {file_info['file_size']} bytes, Expires: {file_info['expires_at']})"
            )
            
            return {"success": True, "file_info": file_info}
            
            # Store in memory for quick access (optional)
            self._cleanup_old_user_files()
            self.user_data_files[user_id] = file_info
            
            logging.info(f"Uploaded file to code_interpreter: {attachment.filename} -> {save_result['file_id']}")
            return {"success": True, "file_info": file_info}
            
        except Exception as e:
            error_msg = f"Error uploading data file: {str(e)}"
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
        Handle ANY data file by uploading to code_interpreter and adding context
        All file types supported - AI will decide how to process via execute_python_code
        
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
            # Upload file to code_interpreter system
            download_result = await self._download_and_save_data_file(attachment, user_id)
            
            if not download_result["success"]:
                await message.channel.send(f"❌ {download_result['error']}")
                return download_result
            
            file_info = download_result["file_info"]
            file_id = file_info["file_id"]
            filename = file_info["filename"]
            file_type = file_info.get("file_type", "unknown")
            file_size = file_info.get("file_size", 0)
            expires_at = file_info.get("expires_at", "Unknown")
            
            # Safety check: Ensure this is not an image file
            if file_type == "image" or os.path.splitext(filename)[1].lower() in IMAGE_FILE_EXTENSIONS:
                await message.channel.send(
                    f"🖼️ **Image File**: `{filename}`\n"
                    f"Your image has been sent to the AI for visual analysis."
                )
                return {"success": True, "message": "Image processed by AI"}
            
            # Format file size for display
            size_kb = file_size / 1024
            size_mb = size_kb / 1024
            if size_mb >= 1:
                size_str = f"{size_mb:.2f} MB"
            else:
                size_str = f"{size_kb:.1f} KB"
            
            # Emoji based on file type
            emoji_map = {
                "csv": "📊", "excel": "📊", "tabular": "📊",
                "json": "📋", "xml": "📋", "yaml": "📋", "structured": "📋",
                "text": "📝", "markdown": "📝",
                "database": "🗄️", "sql": "🗄️",
                "parquet": "📦", "hdf5": "📦", "binary": "📦",
                "python": "🐍", "code": "💻",
                "geojson": "🌍", "shapefile": "🌍", "geospatial": "🌍"
            }
            emoji = emoji_map.get(file_type, "📎")
            
            # Inform user with detailed info
            from src.config.config import MAX_FILES_PER_USER, FILE_EXPIRATION_HOURS
            
            user_files = await self.db.get_user_files(user_id)
            files_count = len(user_files)
            
            expiration_info = f"{FILE_EXPIRATION_HOURS} hours" if FILE_EXPIRATION_HOURS > 0 else "Never (permanent storage)"
            
            await message.channel.send(
                f"{emoji} **File Uploaded Successfully!**\n\n"
                f"📁 **Name**: `{filename}`\n"
                f"� **Type**: {file_type.upper()}\n"
                f"💾 **Size**: {size_str}\n"
                f"🆔 **File ID**: `{file_id}`\n"
                f"⏰ **Expires**: {expires_at}\n"
                f"� **Your Files**: {files_count}/{MAX_FILES_PER_USER}\n\n"
                f"✅ **Ready for processing!** You can now:\n"
                f"• Ask me to analyze this data\n"
                f"• Request visualizations or insights\n"
                f"• Write Python code to process it\n"
                f"• The file is automatically accessible in code execution\n\n"
                f"💡 **Examples:**\n"
                f"```\n"
                f"Analyze this data and show key statistics\n"
                f"Create visualizations from this file\n"
                f"Show me the first 10 rows\n"
                f"Plot correlations between all numeric columns\n"
                f"```"
            )
            
            # Add file context to conversation history for AI
            user_message = message.content.strip() if message.content else ""
            
            file_context = (
                f"\n\n[User uploaded file: {filename}]\n"
                f"[File ID: {file_id}]\n"
                f"[File Type: {file_type}]\n"
                f"[Size: {size_str}]\n"
                f"[Available in code_interpreter via: load_file('{file_id}')]\n"
            )
            
            if user_message:
                file_context += f"[User's request: {user_message}]\n"
            
            # Append to the last user message in history
            if len(history) > 0 and history[-1]["role"] == "user":
                if isinstance(history[-1]["content"], list):
                    history[-1]["content"].append({
                        "type": "text", 
                        "text": file_context
                    })
                else:
                    history[-1]["content"] += file_context
            else:
                # Create new user message with file context
                history.append({
                    "role": "user",
                    "content": file_context
                })
            
            # Save updated history
            await self.db.save_history(user_id, history)
            
            return {
                "success": True, 
                "file_id": file_id, 
                "filename": filename,
                "file_type": file_type
            }
                
        except Exception as e:
            error_msg = f"Error handling file: {str(e)}"
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
                            # Store latest image URL for this user
                            self.user_latest_image_url[user_id] = attachment.url
                            logging.info(f"Stored latest image URL for user {user_id}")
                            
                            content.append({
                                "type": "image_url", 
                                "image_url": {
                                    "url": attachment.url,
                                    "detail": "high"
                                }
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
                # Get fresh system prompt with current time
                system_prompt = self._get_system_prompt_with_time()
                
                # Convert system messages to user instructions
                history_without_system = []
                
                # Remove old system messages and keep conversation messages
                for msg in history:
                    if msg.get('role') != 'system':
                        history_without_system.append(msg)
                
                # Add the fresh system content as a special user message at the beginning
                history_without_system.insert(0, {"role": "user", "content": f"Instructions: {system_prompt}"})
                
                # Add current message and prepare for API
                history_without_system.append(current_message)
                messages_for_api = prepare_messages_for_api(history_without_system)
            else:
                # For models that support system prompts
                # Always update system prompt with current time
                system_prompt = self._get_system_prompt_with_time()
                
                # Remove old system message if present
                history = [msg for msg in history if msg.get('role') != 'system']
                
                # Add updated system prompt with current time
                history.insert(0, {"role": "system", "content": system_prompt})
                    
                history.append(current_message)
                messages_for_api = prepare_messages_for_api(history)
            
            # Proactively trim history to avoid context overload while preserving system prompt
            # Simplified: just check message count instead of tokens
            max_messages = 20
            
            if len(messages_for_api) > max_messages:
                logging.info(f"Proactively trimming history: {len(messages_for_api)} messages > {max_messages} limit for {model}")
                
                if model in ["openai/o1-mini", "openai/o1-preview"]:
                    # For o1 models, trim the history without system prompt
                    trimmed_history_without_system = self._trim_history_to_token_limit(history_without_system, model)
                    messages_for_api = prepare_messages_for_api(trimmed_history_without_system)
                    
                    # Update the history tracking
                    history_without_system = trimmed_history_without_system
                else:
                    # For regular models, trim the full history (preserving system prompt)
                    trimmed_history = self._trim_history_to_token_limit(history, model)
                    messages_for_api = prepare_messages_for_api(trimmed_history)
                    
                    # Update the history tracking
                    history = trimmed_history
                
                # Save the trimmed history immediately to keep it in sync
                if model in ["openai/o1-mini", "openai/o1-preview"]:
                    new_history = []
                    # Save with fresh system prompt for consistency
                    new_history.append({"role": "system", "content": system_prompt})
                    new_history.extend(history_without_system[1:])  # Skip the "Instructions" message
                    await self.db.save_history(user_id, new_history)
                else:
                    await self.db.save_history(user_id, history)
                
                logging.info(f"History trimmed from multiple messages to {len(messages_for_api)} messages")
            
            # Determine which models should have tools available
            # openai/o1-mini and openai/o1-preview do not support tools
            use_tools = model in ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-5", "openai/gpt-5-nano", "openai/gpt-5-mini", "openai/gpt-5-chat", "openai/o1", "openai/o3-mini", "openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4.1-nano", "openai/o3", "openai/o4-mini"]
            
            # Count tokens being sent to API
            total_content_length = 0
            for msg in messages_for_api:
                content = msg.get('content', '')
                if isinstance(content, list):
                    # Handle list content (mixed text/images)
                    for item in content:
                        if item.get('type') == 'text':
                            total_content_length += len(str(item.get('text', '')))
                elif isinstance(content, str):
                    total_content_length += len(content)
            
            estimated_tokens = self._count_tokens_with_tiktoken(' '.join([
                str(msg.get('content', '')) for msg in messages_for_api
            ]))
            
            logging.info(f"API Request Debug - Model: {model}, Messages: {len(messages_for_api)}, "
                        f"Est. tokens: {estimated_tokens}, Content length: {total_content_length} chars")
            
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
                tools = get_tools_for_model()
                api_params["tools"] = tools
            
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
            
            # Extract token usage and calculate cost
            input_tokens = 0
            output_tokens = 0
            total_cost = 0.0
            
            if hasattr(response, 'usage') and response.usage:
                input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                output_tokens = getattr(response.usage, 'completion_tokens', 0)
                
                # Calculate cost based on model pricing
                if model in MODEL_PRICING:
                    pricing = MODEL_PRICING[model]
                    input_cost = (input_tokens / 1_000_000) * pricing["input"]
                    output_cost = (output_tokens / 1_000_000) * pricing["output"]
                    total_cost = input_cost + output_cost
                    
                    logging.info(f"API call - Model: {model}, Input tokens: {input_tokens}, Output tokens: {output_tokens}, Cost: ${total_cost:.6f}")
                    
                    # Save token usage and cost to database
                    await self.db.save_token_usage(user_id, model, input_tokens, output_tokens, total_cost)
            
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
                    
                    # Extract token usage and calculate cost for follow-up call
                    if hasattr(response, 'usage') and response.usage:
                        follow_up_input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                        follow_up_output_tokens = getattr(response.usage, 'completion_tokens', 0)
                        
                        input_tokens += follow_up_input_tokens
                        output_tokens += follow_up_output_tokens
                        
                        # Calculate additional cost
                        if model in MODEL_PRICING:
                            pricing = MODEL_PRICING[model]
                            additional_input_cost = (follow_up_input_tokens / 1_000_000) * pricing["input"]
                            additional_output_cost = (follow_up_output_tokens / 1_000_000) * pricing["output"]
                            additional_cost = additional_input_cost + additional_output_cost
                            total_cost += additional_cost
                            
                            logging.info(f"Follow-up API call - Model: {model}, Input tokens: {follow_up_input_tokens}, Output tokens: {follow_up_output_tokens}, Additional cost: ${additional_cost:.6f}")
                            
                            # Save additional token usage and cost to database
                            await self.db.save_token_usage(user_id, model, follow_up_input_tokens, follow_up_output_tokens, additional_cost)
            
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
                    # Save with fresh system prompt (will be updated with current time on next request)
                    new_history.append({"role": "system", "content": system_prompt})
                    new_history.extend(history_without_system[1:])  # Skip the first "Instructions" message
                    
                    # Only keep a reasonable amount of history (reduced for memory)
                    if len(new_history) > 15:  # Reduced from 20 to 15
                        new_history = new_history[:1] + new_history[-14:]  # Keep system prompt + last 14 messages
                        
                    await self.db.save_history(user_id, new_history)
                else:
                    # For models with system prompt support, just append to regular history
                    if has_images:
                        history.append({"role": "assistant", "content": content_with_images})
                    else:
                        history.append({"role": "assistant", "content": reply})
                    
                    # Only keep a reasonable amount of history (reduced for memory)
                    if len(history) > 15:  # Reduced from 20 to 15
                        history = history[:1] + history[-14:]  # Keep system prompt + last 14 messages
                        
                    await self.db.save_history(user_id, history)
            
            # Send the response text
            await send_response(message.channel, reply)
            
            # Handle charts from code interpreter if present
            if chart_id and chart_id in self.user_charts:
                try:
                    chart_data = self.user_charts[chart_id]["image"]
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
            
            # Log processing time and cost for performance monitoring
            processing_time = time.time() - start_time
            logging.info(f"Message processed in {processing_time:.2f} seconds (User: {user_id}, Model: {model}, Cost: ${total_cost:.6f})")
            
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
        """Perform a Google search with Discord display"""
        try:
            # Find user_id from current task context
            user_id = args.get("user_id")
            if not user_id:
                user_id = self._find_user_id_from_current_task()
            
            # Get the Discord message to display search activity
            discord_message = self._get_discord_message_from_current_task()
            
            # Extract search parameters
            query = args.get('query', '')
            num_results = args.get('num_results', 3)
            
            # Import and call Google search
            from src.utils.web_utils import google_search
            result = await google_search(args)
            
            # Display the search activity in Discord (only if user has enabled tool display)
            if discord_message and query:
                # Check user's tool display preference
                show_search_details = await self.db.get_user_tool_display(user_id) if user_id else False
                
                if show_search_details:
                    try:
                        # Parse the result to get structured data
                        import json
                        search_data = json.loads(result) if isinstance(result, str) else result
                        
                        # Get the combined content
                        combined_content = search_data.get('combined_content', '')
                        
                        # Check if content is too long for Discord message (3000 chars limit)
                        if len(combined_content) > 3000:
                            # Send content as file attachment
                            content_file = discord.File(
                                io.StringIO(combined_content), 
                                filename="search_results.txt"
                            )
                            
                            # Create display text without full content
                            search_display = "**🔍 Google Search**\n\n"
                            search_display += f"**📝 Query:** `{query}`\n"
                            search_display += f"**📊 Results:** {num_results} requested\n\n"
                            
                            # Show search results with links
                            if 'results' in search_data and search_data['results']:
                                search_display += "**🔗 Found Links:**\n"
                                for i, item in enumerate(search_data['results'][:5], 1):
                                    title = item.get('title', 'No title')[:80]
                                    link = item.get('link', '')
                                    used_mark = "✅" if item.get('used_for_content', False) else "📄"
                                    search_display += f"{i}. {used_mark} [{title}]({link})\n"
                                search_display += "\n"
                            
                            search_display += "**📄 Content:** *Attached as file (too long to display)*"
                            
                            if 'error' in search_data:
                                search_display += f"\n**❌ Error:** {search_data['error'][:300]}"
                            
                            # Send with file attachment
                            await discord_message.channel.send(search_display, file=content_file)
                        else:
                            # Use normal display for shorter content
                            search_display = "**🔍 Google Search**\n\n"
                            search_display += f"**📝 Query:** `{query}`\n"
                            search_display += f"**📊 Results:** {num_results} requested\n\n"
                            
                            # Show search results with links
                            if 'results' in search_data and search_data['results']:
                                search_display += "**🔗 Found Links:**\n"
                                for i, item in enumerate(search_data['results'][:5], 1):
                                    title = item.get('title', 'No title')[:80]
                                    link = item.get('link', '')
                                    used_mark = "✅" if item.get('used_for_content', False) else "📄"
                                    search_display += f"{i}. {used_mark} [{title}]({link})\n"
                                search_display += "\n"
                            
                            # Show content preview
                            if combined_content.strip():
                                search_display += "**📄 Content:**\n```\n"
                                search_display += combined_content
                                search_display += "\n```"
                            else:
                                search_display += "**📄 Content:** *(No content retrieved)*"
                            
                            if 'error' in search_data:
                                search_display += f"\n**❌ Error:** {search_data['error']}"
                            
                            # Send the search display to Discord
                            await discord_message.channel.send(search_display)
                            
                    except Exception as e:
                        logging.error(f"Error displaying Google search: {str(e)}")
                        # Fallback: just send a simple message to prevent bot from getting stuck
                        try:
                            await discord_message.channel.send(f"🔍 Google search completed for: `{query}`")
                        except:
                            pass
            
            return result
        except Exception as e:
            logging.error(f"Error in Google search: {str(e)}")
            return json.dumps({"error": f"Google search failed: {str(e)}"})
    
    async def _scrape_webpage(self, args: Dict[str, Any]):
        """Scrape a webpage with Discord display"""
        try:
            # Find user_id from current task context
            user_id = args.get("user_id")
            if not user_id:
                user_id = self._find_user_id_from_current_task()
            
            # Get the Discord message to display scraping activity
            discord_message = self._get_discord_message_from_current_task()
            
            # Extract scraping parameters
            url = args.get('url', '')
            max_tokens = args.get('max_tokens', 4000)
            
            # Import and call webpage scraper
            from src.utils.web_utils import scrape_webpage
            result = await scrape_webpage(args)
            
            # Display the scraping activity in Discord (only if user has enabled tool display)
            if discord_message and url:
                # Check user's tool display preference
                show_scrape_details = await self.db.get_user_tool_display(user_id) if user_id else False
                
                if show_scrape_details:
                    try:
                        # Parse the result to get structured data
                        import json
                        scrape_data = json.loads(result) if isinstance(result, str) else result
                        
                        # Get the scraped content
                        content = scrape_data.get('content', '') if scrape_data.get('success') else ''
                        
                        # Check if content is too long for Discord message (3000 chars limit)
                        if len(content) > 3000:
                            # Send content as file attachment
                            content_file = discord.File(
                                io.StringIO(content), 
                                filename="scraped_content.txt"
                            )
                            
                            # Create display text without full content
                            scrape_display = "**🌐 Webpage Scraping**\n\n"
                            scrape_display += f"**🔗 URL:** {url}\n"
                            scrape_display += f"**⚙️ Max Tokens:** {max_tokens}\n\n"
                            scrape_display += "**📄 Content:** *Attached as file (too long to display)*"
                            
                            if 'error' in scrape_data:
                                scrape_display += f"\n**❌ Error:** {scrape_data['error'][:300]}"
                            elif content:
                                scrape_display += f"\n**✅ Success:** Scraped {len(content)} characters"
                            
                            # Send with file attachment
                            await discord_message.channel.send(scrape_display, file=content_file)
                        else:
                            # Use normal display for shorter content
                            scrape_display = "**🌐 Webpage Scraping**\n\n"
                            scrape_display += f"**🔗 URL:** {url}\n"
                            scrape_display += f"**⚙️ Max Tokens:** {max_tokens}\n\n"
                            
                            # Show content
                            if content.strip():
                                scrape_display += "**📄 Content:**\n```\n"
                                scrape_display += content
                                scrape_display += "\n```"
                            else:
                                scrape_display += "**📄 Content:** *(No content retrieved)*"
                            
                            if 'error' in scrape_data:
                                scrape_display += f"\n**❌ Error:** {scrape_data['error']}"
                            
                            # Send the scraping display to Discord
                            await discord_message.channel.send(scrape_display)
                            
                    except Exception as e:
                        logging.error(f"Error displaying webpage scraping: {str(e)}")
                        # Fallback: just send a simple message to prevent bot from getting stuck
                        try:
                            await discord_message.channel.send(f"🌐 Webpage scraping completed for: {url}")
                        except:
                            pass
            
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
            # Check if model passed "latest_image" - use stored URL
            image_url = args.get("image_url", "")
            if image_url == "latest_image" or not image_url:
                user_id = self._find_user_id_from_current_task()
                if user_id:
                    # Try in-memory first (from current session), then database
                    if user_id in self.user_latest_image_url:
                        args["image_url"] = self.user_latest_image_url[user_id]
                        logging.info(f"Using in-memory image URL for image_to_text")
                    else:
                        db_url = await self._get_latest_image_url_from_db(user_id)
                        if db_url:
                            args["image_url"] = db_url
                            logging.info(f"Using database image URL for image_to_text")
                        else:
                            return json.dumps({"error": "No image found. Please upload an image first."})
                else:
                    return json.dumps({"error": "No image found. Please upload an image first."})
            
            result = await self.image_generator.image_to_text(args)
            return result
        except Exception as e:
            logging.error(f"Error in image to text: {str(e)}")
            return json.dumps({"error": f"Image to text failed: {str(e)}"})
    
    async def _upscale_image(self, args: Dict[str, Any]):
        """Upscale an image"""
        try:
            # Check if model passed "latest_image" - use stored URL
            image_url = args.get("image_url", "")
            if image_url == "latest_image" or not image_url:
                user_id = self._find_user_id_from_current_task()
                if user_id:
                    # Try in-memory first (from current session), then database
                    if user_id in self.user_latest_image_url:
                        args["image_url"] = self.user_latest_image_url[user_id]
                        logging.info(f"Using in-memory image URL for upscale")
                    else:
                        db_url = await self._get_latest_image_url_from_db(user_id)
                        if db_url:
                            args["image_url"] = db_url
                            logging.info(f"Using database image URL for upscale")
                        else:
                            return json.dumps({"error": "No image found. Please upload an image first."})
                else:
                    return json.dumps({"error": "No image found. Please upload an image first."})
            
            result = await self.image_generator.upscale_image(args)
            return result
        except Exception as e:
            logging.error(f"Error in image upscaling: {str(e)}")
            return json.dumps({"error": f"Image upscaling failed: {str(e)}"})
    
    async def _remove_background(self, args: Dict[str, Any]):
        """Remove background from an image"""
        try:
            # Check if model passed "latest_image" - use stored URL
            image_url = args.get("image_url", "")
            if image_url == "latest_image" or not image_url:
                user_id = self._find_user_id_from_current_task()
                if user_id:
                    # Try in-memory first (from current session), then database
                    if user_id in self.user_latest_image_url:
                        args["image_url"] = self.user_latest_image_url[user_id]
                        logging.info(f"Using in-memory image URL for background removal")
                    else:
                        db_url = await self._get_latest_image_url_from_db(user_id)
                        if db_url:
                            args["image_url"] = db_url
                            logging.info(f"Using database image URL for background removal")
                        else:
                            return json.dumps({"error": "No image found. Please upload an image first."})
                else:
                    return json.dumps({"error": "No image found. Please upload an image first."})
            
            result = await self.image_generator.remove_background(args)
            return result
        except Exception as e:
            logging.error(f"Error in background removal: {str(e)}")
            return json.dumps({"error": f"Background removal failed: {str(e)}"})
    
    async def _photo_maker(self, args: Dict[str, Any]):
        """Create a photo"""
        try:
            # Check if model passed "latest_image" in input_images - use stored URL
            input_images = args.get("input_images", [])
            if input_images and "latest_image" in input_images:
                user_id = self._find_user_id_from_current_task()
                if user_id:
                    # Try in-memory first (from current session), then database
                    if user_id in self.user_latest_image_url:
                        url = self.user_latest_image_url[user_id]
                        args["input_images"] = [url if img == "latest_image" else img for img in input_images]
                        logging.info(f"Using in-memory image URL for photo_maker")
                    else:
                        db_url = await self._get_latest_image_url_from_db(user_id)
                        if db_url:
                            args["input_images"] = [db_url if img == "latest_image" else img for img in input_images]
                            logging.info(f"Using database image URL for photo_maker")
                        else:
                            return json.dumps({"error": "No image found. Please upload an image first."})
                else:
                    return json.dumps({"error": "No image found. Please upload an image first."})
            
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
        """Run aggressive chart cleanup for memory optimization"""
        while True:
            try:
                await asyncio.sleep(600)  # Every 10 minutes (reduced from 1 hour)
                current_time = datetime.now()
                
                # Clean charts older than 30 minutes
                expired_charts = [
                    chart_id for chart_id, data in self.user_charts.items()
                    if current_time - data.get('timestamp', current_time) > timedelta(minutes=30)
                ]
                
                for chart_id in expired_charts:
                    self.user_charts.pop(chart_id, None)
                
                if expired_charts:
                    logging.info(f"Cleaned up {len(expired_charts)} expired charts")
                    
            except Exception as e:
                logging.error(f"Error in chart cleanup: {str(e)}")
    
    async def _run_file_cleanup(self):
        """Run aggressive file cleanup for memory optimization"""
        while True:
            try:
                await asyncio.sleep(900)  # Every 15 minutes (reduced from 2 hours)
                self._cleanup_old_user_files()
            except Exception as e:
                logging.error(f"Error in file cleanup: {str(e)}")
    
    def _cleanup_old_user_files(self):
        """Clean up old user data files to prevent memory bloat"""
        current_time = datetime.now()
        
        # Remove files older than 1 hour
        expired_users = [
            user_id for user_id, file_info in self.user_data_files.items()
            if current_time - file_info['timestamp'] > timedelta(hours=1)
        ]
        
        for user_id in expired_users:
            file_info = self.user_data_files.pop(user_id, None)
            if file_info and os.path.exists(file_info['file_path']):
                try:
                    os.remove(file_info['file_path'])
                except Exception as e:
                    logging.error(f"Error removing file: {e}")
        
        # Limit total number of cached files
        if len(self.user_data_files) > self.max_user_files:
            # Remove oldest files
            sorted_files = sorted(
                self.user_data_files.items(), 
                key=lambda x: x[1]['timestamp']
            )
            
            files_to_remove = len(self.user_data_files) - self.max_user_files
            for user_id, file_info in sorted_files[:files_to_remove]:
                self.user_data_files.pop(user_id, None)
                if os.path.exists(file_info['file_path']):
                    try:
                        os.remove(file_info['file_path'])
                    except Exception as e:
                        logging.error(f"Error removing file: {e}")
        
        if expired_users:
            logging.info(f"Cleaned up {len(expired_users)} expired user files")
    
    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        DEPRECATED: Token counting is now handled by API response.
        This method is kept for backward compatibility but returns 0.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            int: Always returns 0 (use API response for actual counts)
        """
        logging.warning("_count_tokens is deprecated. Use API response usage field instead.")
        return 0
    
    def _trim_history_to_token_limit(self, history: List[Dict[str, Any]], model: str, target_tokens: int = None) -> List[Dict[str, Any]]:
        """
        Trim conversation history using sliding window approach (like ChatGPT).
        No summarization - just keep most recent messages that fit within limit.
        Uses MODEL_TOKEN_LIMITS from config for each model.
        
        Args:
            history: List of message dictionaries
            model: Model name
            target_tokens: Override token limit (optional)
            
        Returns:
            List[Dict[str, Any]]: Trimmed history within token limits
        """
        try:
            from src.config.config import MODEL_TOKEN_LIMITS, DEFAULT_TOKEN_LIMIT
            
            # Get token limit for this model (use configured limits)
            if target_tokens is None:
                target_tokens = MODEL_TOKEN_LIMITS.get(model, DEFAULT_TOKEN_LIMIT)
            
            # Always preserve system messages
            system_messages = [msg for msg in history if msg.get('role') == 'system']
            conversation_messages = [msg for msg in history if msg.get('role') != 'system']
            
            # Count tokens for system messages (always keep)
            system_tokens = sum(
                self._count_tokens_with_tiktoken(str(msg.get('content', '')))
                for msg in system_messages
            )
            
            # Available tokens for conversation (reserve 20% for response)
            available_tokens = int((target_tokens - system_tokens) * 0.8)
            
            if available_tokens <= 0:
                logging.warning(f"System messages exceed token limit! System: {system_tokens}, Limit: {target_tokens}")
                return system_messages + conversation_messages[-1:]  # Keep at least last message
            
            # Sliding window: Keep most recent messages that fit
            # Group user+assistant pairs together for better context
            message_pairs = []
            i = len(conversation_messages) - 1
            
            while i >= 0:
                msg = conversation_messages[i]
                
                # If assistant message, try to include the user message before it
                if msg.get('role') == 'assistant' and i > 0 and conversation_messages[i-1].get('role') == 'user':
                    pair = [conversation_messages[i-1], msg]
                    i -= 2
                else:
                    pair = [msg]
                    i -= 1
                
                message_pairs.insert(0, pair)
            
            # Now select pairs from most recent until we hit token limit
            selected_messages = []
            current_tokens = 0
            
            for pair in reversed(message_pairs):
                pair_tokens = sum(
                    self._count_tokens_with_tiktoken(str(msg.get('content', '')))
                    for msg in pair
                )
                
                if current_tokens + pair_tokens <= available_tokens:
                    selected_messages = pair + selected_messages
                    current_tokens += pair_tokens
                else:
                    # Stop if we can't fit this pair
                    break
            
            # Always keep at least the last user message if nothing fits
            if not selected_messages and conversation_messages:
                selected_messages = [conversation_messages[-1]]
                current_tokens = self._count_tokens_with_tiktoken(str(conversation_messages[-1].get('content', '')))
            
            result = system_messages + selected_messages
            
            messages_removed = len(conversation_messages) - len(selected_messages)
            if messages_removed > 0:
                logging.info(
                    f"Sliding window trim: {len(history)} → {len(result)} messages "
                    f"({messages_removed} removed, ~{current_tokens + system_tokens}/{target_tokens} tokens, {model})"
                )
            
            return result
            
        except Exception as e:
            logging.error(f"Error trimming history: {e}")
            traceback.print_exc()
            # Fallback: simple message count limit
            max_messages = 20
            if len(history) > max_messages:
                system_msgs = [msg for msg in history if msg.get('role') == 'system']
                other_msgs = [msg for msg in history if msg.get('role') != 'system']
                return system_msgs + other_msgs[-max_messages:]
            return history
