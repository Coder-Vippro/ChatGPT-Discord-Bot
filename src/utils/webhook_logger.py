import asyncio
import json
import logging
import threading
import time
import traceback
import queue
import requests
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, TextIO

class WebhookLogHandler(logging.Handler):
    """
    A logging handler that sends log records to a Discord webhook.
    Implements batching and asynchronous sending to avoid performance impact.
    """
    
    def __init__(self, webhook_url: str, app_name: str, level=logging.INFO, 
                 batch_size: int = 10, flush_interval: int = 60):
        """
        Initialize the webhook log handler.
        
        Args:
            webhook_url (str): Discord webhook URL to send logs to
            app_name (str): Name of the application for identifying the source
            level: Log level (default: INFO)
            batch_size (int): Number of logs to batch before sending
            flush_interval (int): Maximum seconds to wait before sending logs
        """
        super().__init__(level)
        self.webhook_url = webhook_url
        self.app_name = app_name
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Queue for log records
        self.log_queue = queue.Queue()
        
        # Background thread for processing logs
        self.should_stop = threading.Event()
        self.thread = threading.Thread(target=self._process_logs, daemon=True)
        self.thread.start()
        
        # Track last flush time
        self.last_flush = time.time()
    
    def emit(self, record):
        """Process a log record by adding it to the queue for batching."""
        try:
            if self.should_stop.is_set():
                return
                
            # Format and enqueue the log record
            log_entry = self.format_log_entry(record)
            self.log_queue.put(log_entry)
        except Exception:
            self.handleError(record)
    
    def format_log_entry(self, record):
        """Format a log record into a dictionary for the webhook."""
        try:
            # Get the formatted exception info if available
            if record.exc_info:
                exc_text = self.formatter.formatException(record.exc_info)
            else:
                exc_text = None
                
            # Get the log message
            try:
                message = self.format(record)
            except Exception:
                message = str(record.msg)
                
            # Create a color based on log level
            colors = {
                logging.DEBUG: 0x7F8C8D,     # Gray
                logging.INFO: 0x3498DB,      # Blue
                logging.WARNING: 0xF1C40F,   # Yellow
                logging.ERROR: 0xE74C3C,     # Red
                logging.CRITICAL: 0x9B59B6   # Purple
            }
            color = colors.get(record.levelno, 0xFFFFFF)  # Default to white
            
            # Create a timestamp in ISO format
            timestamp = datetime.fromtimestamp(record.created).isoformat()
            
            # Structure the log entry as a Discord embed
            log_entry = {
                "embeds": [{
                    "title": f"{self.app_name} - {record.levelname}",
                    "description": f"```{message[:2000]}```",  # Discord limits to 2000 chars
                    "color": color,
                    "fields": [
                        {
                            "name": "Module",
                            "value": record.name,
                            "inline": True
                        },
                        {
                            "name": "Function",
                            "value": record.funcName,
                            "inline": True
                        }
                    ],
                    "footer": {
                        "text": f"{record.filename}:{record.lineno}"
                    },
                    "timestamp": timestamp
                }]
            }
            
            # Add exception information if present
            if exc_text:
                # Truncate if too long
                if len(exc_text) > 1000:
                    exc_text = exc_text[:997] + "..."
                    
                log_entry["embeds"][0]["fields"].append({
                    "name": "Exception",
                    "value": f"```{exc_text}```",
                    "inline": False
                })
                
            return log_entry
            
        except Exception as e:
            # Fallback in case of formatting error
            return {
                "content": f"**{self.app_name} - LOG ERROR**: Could not format log record. Error: {str(e)}"
            }
    
    def _process_logs(self):
        """Background thread to process and send logs in batches."""
        batch = []
        
        while not self.should_stop.is_set():
            try:
                # Try to get a log entry with timeout
                try:
                    log_entry = self.log_queue.get(timeout=1.0)
                    batch.append(log_entry)
                    self.log_queue.task_done()
                except queue.Empty:
                    # No new logs in the last second
                    pass
                
                current_time = time.time()
                should_flush = (
                    len(batch) >= self.batch_size or
                    (len(batch) > 0 and current_time - self.last_flush >= self.flush_interval)
                )
                
                # Send the batch if it's full or it's time to flush
                if should_flush:
                    self._send_batch(batch)
                    batch = []
                    self.last_flush = current_time
                    
            except Exception as e:
                # Log errors to standard error since we can't use the logging system
                print(f"Error in webhook logger thread: {str(e)}", file=sys.stderr)
                time.sleep(5)  # Avoid tight error loops
    
    def _send_batch(self, batch: List[Dict]):
        """Send a batch of log entries to the webhook."""
        if not batch:
            return
            
        try:
            # For multiple logs, combine them into a single webhook call if possible
            if len(batch) == 1:
                # Single log entry - send as is
                payload = batch[0]
            else:
                # Multiple logs - combine embeds up to Discord's limit (10 embeds per message)
                all_embeds = []
                for entry in batch:
                    if "embeds" in entry:
                        all_embeds.extend(entry["embeds"][:10 - len(all_embeds)])
                        if len(all_embeds) >= 10:
                            break
                            
                payload = {"embeds": all_embeds[:10]}
            
            # Send to webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                # Get retry_after from response
                retry_after = response.json().get('retry_after', 5) / 1000.0  # Convert to seconds
                time.sleep(retry_after + 0.5)  # Add a small buffer
                
                # Retry the request
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
            if response.status_code not in (200, 204):
                print(f"Error sending logs to webhook. Status: {response.status_code}", file=sys.stderr)
                
        except Exception as e:
            print(f"Failed to send logs to webhook: {str(e)}", file=sys.stderr)
            traceback.print_exc()
    
    def flush(self):
        """Force flushing of logs."""
        # Process all remaining logs in the queue
        batch = []
        while not self.log_queue.empty():
            try:
                log_entry = self.log_queue.get_nowait()
                batch.append(log_entry)
                self.log_queue.task_done()
            except queue.Empty:
                break
        
        if batch:
            self._send_batch(batch)
            self.last_flush = time.time()
    
    def close(self):
        """Close the handler and stop the background thread."""
        self.should_stop.set()
        self.flush()
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)
        super().close()


class ConsoleToWebhookRedirector(TextIO):
    """
    A class that redirects stdout/stderr to both the original stream and a logger.
    This allows capturing console output and sending it to a Discord webhook.
    """
    
    def __init__(self, original_stream, logger_name, level=logging.INFO):
        """
        Initialize the redirector.
        
        Args:
            original_stream: The original stream (sys.stdout or sys.stderr)
            logger_name: Name of the logger to use
            level: Logging level for the messages
        """
        self.original_stream = original_stream
        self.logger = logging.getLogger(logger_name)
        self.level = level
        self.line_buffer = ""
        
    def write(self, message):
        """Write to both original stream and logger."""
        # Always write to the original stream
        self.original_stream.write(message)
        
        # Accumulate message parts until we get a newline
        self.line_buffer += message
        if '\n' in self.line_buffer:
            # Split by newlines, preserving any trailing partial line
            lines = self.line_buffer.split('\n')
            
            # The last element might be a partial line or empty string
            self.line_buffer = lines.pop()
            
            # Log each complete line
            for line in lines:
                if line.strip():  # Skip empty lines
                    self.logger.log(self.level, line)
        
    def flush(self):
        """Flush the original stream."""
        self.original_stream.flush()
        if self.line_buffer.strip():
            self.logger.log(self.level, self.line_buffer)
            self.line_buffer = ""
            
    def close(self):
        """Close is a no-op for compatibility."""
        # Don't close the original stream
        pass
    
    # Implement other TextIO methods for compatibility
    def readable(self): return False
    def writable(self): return True
    def seekable(self): return False
    def isatty(self): return self.original_stream.isatty()
    def fileno(self): return self.original_stream.fileno()
    
    # Support context manager interface
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.flush()

class WebhookLogManager:
    """
    Manager class for webhook logging setup and cleanup.
    """
    def __init__(self):
        self.active_handlers = []
        self.console_redirectors = []
    
    def setup_webhook_logging(
        self, 
        webhook_url: str,
        app_name: str = "Discord Bot",
        level: int = logging.INFO,
        loggers: Optional[List[str]] = None,
        formatter: Optional[logging.Formatter] = None,
        batch_size: int = 10, 
        flush_interval: int = 60
    ):
        """
        Set up webhook logging for the specified loggers.
        
        Args:
            webhook_url: Discord webhook URL for sending logs
            app_name: Name of the application
            level: Minimum log level to send
            loggers: List of logger names to set up. If None, uses the root logger.
            formatter: Custom formatter. If None, a default formatter is used.
            batch_size: Number of logs to batch before sending
            flush_interval: Maximum seconds to wait before sending logs
        """
        if not formatter:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            
        # Create the handler
        handler = WebhookLogHandler(
            webhook_url=webhook_url,
            app_name=app_name,
            level=level,
            batch_size=batch_size,
            flush_interval=flush_interval
        )
        
        # Set the formatter
        handler.setFormatter(formatter)
        
        # Add to the specified loggers, or the root logger if none specified
        if not loggers:
            logging.getLogger().addHandler(handler)
        else:
            for logger_name in loggers:
                logging.getLogger(logger_name).addHandler(handler)
                
        # Keep track of the handler
        self.active_handlers.append(handler)
        
        return handler
    
    def capture_console_to_webhook(self, logger_name="console", stdout_level=logging.INFO, stderr_level=logging.ERROR):
        """
        Redirect stdout and stderr to both console and webhook logger.
        
        Args:
            logger_name: Name for the console capture logger
            stdout_level: Log level for stdout messages
            stderr_level: Log level for stderr messages
        """
        # Create stdout redirector
        stdout_redirector = ConsoleToWebhookRedirector(
            original_stream=sys.stdout,
            logger_name=logger_name,
            level=stdout_level
        )
        sys.stdout = stdout_redirector
        self.console_redirectors.append(stdout_redirector)
        
        # Create stderr redirector
        stderr_redirector = ConsoleToWebhookRedirector(
            original_stream=sys.stderr,
            logger_name=logger_name,
            level=stderr_level
        )
        sys.stderr = stderr_redirector
        self.console_redirectors.append(stderr_redirector)
        
        return (stdout_redirector, stderr_redirector)
    
    def capture_module_logs_to_webhook(self, module_name, webhook_url=None, app_name=None):
        """
        Utility function to quickly capture logs from a specific module to webhook.
        
        Args:
            module_name: Name of the module to capture logs from
            webhook_url: Optional webhook URL (uses existing if None)
            app_name: Optional app name for the logs
        """
        # Get the logger for the module
        logger = logging.getLogger(module_name)
        
        # Set up handler if webhook URL is provided
        if webhook_url:
            self.setup_webhook_logging(
                webhook_url=webhook_url,
                app_name=app_name or f"{module_name} Module",
                loggers=[module_name]
            )
        
        return logger
    
    def cleanup(self):
        """Close and remove all active webhook handlers and console redirectors."""
        # First restore console streams if they were redirected
        for redirector in self.console_redirectors:
            if redirector is sys.stdout:
                sys.stdout = redirector.original_stream
            elif redirector is sys.stderr:
                sys.stderr = redirector.original_stream
        
        self.console_redirectors.clear()
        
        for handler in self.active_handlers:
            try:
                # Flush any remaining logs
                handler.flush()
                
                # Find loggers using this handler
                for logger in [logging.getLogger()] + list(logging.Logger.manager.loggerDict.values()):
                    if hasattr(logger, 'handlers'):
                        if handler in logger.handlers:
                            logger.removeHandler(handler)
                
                # Close the handler
                handler.close()
            except Exception as e:
                print(f"Error cleaning up webhook handler: {str(e)}", file=sys.stderr)
        
        # Clear the list of active handlers
        self.active_handlers.clear()

# Export a singleton instance for easy access
webhook_log_manager = WebhookLogManager()

# Export a convenient function to get a webhook logger
def webhook_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger configured to send to webhook if one is set up."""
    return logging.getLogger(name)

# Add configuration option to disable HTTPS verification in requests
# (useful for development environments with self-signed certificates)
VERIFY_HTTPS = True

def configure_requests_session(verify: bool = True):
    """Configure requests session for webhook logging."""
    global VERIFY_HTTPS
    VERIFY_HTTPS = verify
    
    if not verify:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)