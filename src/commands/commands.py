import discord
from discord import app_commands
from discord.ext import commands
import logging
import io
import asyncio
from typing import Optional, Dict, List, Any, Callable

from src.config.config import MODEL_OPTIONS, PDF_ALLOWED_MODELS
from src.utils.image_utils import ImageGenerator
from src.utils.web_utils import google_custom_search, scrape_web_content
from src.utils.pdf_utils import process_pdf, send_response
from src.utils.openai_utils import prepare_file_from_path
from src.utils.model_selector import model_selector
from src.utils.user_preferences import UserPreferences
from src.utils.conversation_manager import ConversationSummarizer
from src.utils.enhanced_file_processor import enhanced_file_processor

# Dictionary to keep track of user requests and their cooldowns
user_requests = {}
# Dictionary to store user tasks
user_tasks = {}

def setup_commands(bot: commands.Bot, db_handler, openai_client, image_generator: ImageGenerator):
    """
    Set up all slash commands for the bot.
    
    Args:
        bot: Discord bot instance
        db_handler: Database handler instance
        openai_client: OpenAI client instance
        image_generator: Image generator instance
    """
    tree = bot.tree
    
    # Initialize enhancement utilities
    user_prefs_manager = UserPreferences(db_handler)
    conversation_summarizer = ConversationSummarizer(openai_client, db_handler)
    
    def check_blacklist():
        """Decorator to check if a user is blacklisted before executing a command."""
        async def predicate(interaction: discord.Interaction):
            if await db_handler.is_admin(interaction.user.id):
                return True
            if await db_handler.is_user_blacklisted(interaction.user.id):
                await interaction.response.send_message("You have been blacklisted from using this bot. Please contact the admin if you think this is a mistake.", ephemeral=True)
                return False
            return True
        return app_commands.check(predicate)

    # Processes a command request with rate limiting and queuing.
    async def process_request(interaction, command_func, *args):
        user_id = interaction.user.id
        now = discord.utils.utcnow().timestamp()
        
        if user_id not in user_requests:
            user_requests[user_id] = {'last_request': 0, 'queue': asyncio.Queue()}
            
        last_request = user_requests[user_id]['last_request']
        
        if now - last_request < 5:
            await interaction.followup.send("You are sending requests too quickly. Please wait a moment.", ephemeral=True)
            return

        # Update last request time
        user_requests[user_id]['last_request'] = now

        # Add request to queue
        queue = user_requests[user_id]['queue']
        await queue.put((command_func, args))

        # Start processing if it's the only request in the queue
        if queue.qsize() == 1:
            await process_queue(interaction)

    # Processes requests in the user's queue sequentially.
    async def process_queue(interaction):
        user_id = interaction.user.id
        queue = user_requests[user_id]['queue']
        
        while not queue.empty():
            command_func, args = await queue.get()
            try:
                await command_func(interaction, *args)
            except Exception as e:
                logging.error(f"Error processing command: {str(e)}")
                await interaction.followup.send(f"An error occurred: {str(e)}", ephemeral=True)
            await asyncio.sleep(1)  # Optional delay between processing

    async def send_response_with_image(interaction: discord.Interaction, response_text: str, image_path: str):
        """Send a response with an image file."""
        try:
            file = await prepare_file_from_path(image_path)
            await interaction.followup.send(content=response_text, file=file)
        except Exception as e:
            logging.error(f"Error sending image: {str(e)}")
            await interaction.followup.send(f"Error sending image: {str(e)}")
    
    @tree.command(name="choose_model", description="Select the AI model to use for responses.")
    @check_blacklist()
    async def choose_model(interaction: discord.Interaction):
        """Lets users choose an AI model and saves it to the database."""
        options = [discord.SelectOption(label=model, value=model) for model in MODEL_OPTIONS]
        select_menu = discord.ui.Select(placeholder="Choose a model", options=options)

        async def select_callback(interaction: discord.Interaction):
            selected_model = select_menu.values[0]
            user_id = interaction.user.id
            
            # Save the model selection to the database
            await db_handler.save_user_model(user_id, selected_model)
            await interaction.response.send_message(
                f"Model set to `{selected_model}` for your responses.", ephemeral=True
            )

        select_menu.callback = select_callback
        view = discord.ui.View()
        view.add_item(select_menu)
        await interaction.response.send_message("Choose a model:", view=view, ephemeral=True)

    @tree.command(name="search", description="Search on Google and send results to AI model.")
    @app_commands.describe(query="The search query")
    @check_blacklist()
    async def search(interaction: discord.Interaction, query: str):
        """Searches Google and sends results to the AI model."""
        await interaction.response.defer(thinking=True)
        
        async def process_search(interaction: discord.Interaction, query: str):
            user_id = interaction.user.id
            model = await db_handler.get_user_model(user_id) or "openai/gpt-4.1-mini"
            history = await db_handler.get_history(user_id)

            try:
                # Perform Google search
                search_results = google_custom_search(query)
                
                if not search_results or not search_results.get('results'):
                    await interaction.followup.send("No search results found.")
                    return

                # Format search results for the AI model
                from src.config.config import SEARCH_PROMPT
                formatted_results = f"Search results for: {query}\n\n"
                
                for i, result in enumerate(search_results.get('results', [])):
                    formatted_results += f"{i+1}. {result.get('title')}\n"
                    formatted_results += f"URL: {result.get('link')}\n"
                    formatted_results += f"Snippet: {result.get('snippet')}\n"
                    if 'scraped_content' in result:
                        content_preview = result['scraped_content'][:300] + "..." if len(result['scraped_content']) > 300 else result['scraped_content']
                        formatted_results += f"Content: {content_preview}\n"
                    formatted_results += "\n"

                # Prepare messages for the AI model, handling system prompts appropriately
                messages = []
                if model in ["openai/o1-mini", "openai/o1-preview"]:
                    messages = [
                        {"role": "user", "content": f"Instructions: {SEARCH_PROMPT}\n\n{formatted_results}\n\nUser query: {query}"}
                    ]
                else:
                    messages = [
                        {"role": "system", "content": SEARCH_PROMPT},
                        {"role": "user", "content": f"{formatted_results}\n\nUser query: {query}"}
                    ]

                # Send to the AI model
                response = await openai_client.chat.completions.create(
                    model=model if model in ["openai/gpt-4o", "openai/openai/gpt-4o-mini"] else "openai/gpt-4o",
                    messages=messages,
                    temperature=0.5
                )

                reply = response.choices[0].message.content
                
                # Add the interaction to history
                history.append({"role": "user", "content": f"Search query: {query}"})
                history.append({"role": "assistant", "content": reply})
                await db_handler.save_history(user_id, history)

                # Check if the reply exceeds Discord's character limit (2000)
                if len(reply) > 2000:
                    # Create a text file with the full response
                    file_bytes = io.BytesIO(reply.encode('utf-8'))
                    file = discord.File(file_bytes, filename="search_response.txt")
                    
                    # Send a short message with the file attachment
                    await interaction.followup.send(
                        f"The search response for '{query}' is too long for Discord (>{len(reply)} characters). Here's the full response as a text file:", 
                        file=file
                    )
                else:
                    # Send as normal message if within limits
                    await interaction.followup.send(reply)

            except Exception as e:
                error_message = f"Search error: {str(e)}"
                logging.error(error_message)
                await interaction.followup.send(f"An error occurred while searching: {str(e)}")
        
        await process_request(interaction, process_search, query)

    @tree.command(name="web", description="Scrape a webpage and send data to AI model.")
    @app_commands.describe(url="The webpage URL to scrape")
    @check_blacklist()
    async def web(interaction: discord.Interaction, url: str):
        """Scrapes a webpage and sends data to the AI model."""
        await interaction.response.defer(thinking=True)
        
        async def process_web(interaction: discord.Interaction, url: str):
            user_id = interaction.user.id
            model = await db_handler.get_user_model(user_id) or "openai/gpt-4.1-mini"
            history = await db_handler.get_history(user_id)

            try:
                content = scrape_web_content(url)
                if content.startswith("Failed"):
                    await interaction.followup.send(content)
                    return
                
                from src.config.config import WEB_SCRAPING_PROMPT
                
                if model in ["openai/o1-mini", "openai/o1-preview"]:
                    messages = [
                        {"role": "user", "content": f"Instructions: {WEB_SCRAPING_PROMPT}\n\nContent from {url}:\n{content}"}
                    ]
                else:
                    messages = [
                        {"role": "system", "content": WEB_SCRAPING_PROMPT},
                        {"role": "user", "content": f"Content from {url}:\n{content}"}
                    ]

                response = await openai_client.chat.completions.create(
                    model=model if model in ["openai/gpt-4o", "openai/gpt-4o-mini"] else "openai/gpt-4o",
                    messages=messages,
                    temperature=0.3,
                    top_p=0.7
                )

                reply = response.choices[0].message.content
                
                # Add the interaction to history
                history.append({"role": "user", "content": f"Scraped content from {url}"})
                history.append({"role": "assistant", "content": reply})
                await db_handler.save_history(user_id, history)

                # Check if the reply exceeds Discord's character limit (2000)
                if len(reply) > 2000:
                    # Create a text file with the full response
                    file_bytes = io.BytesIO(reply.encode('utf-8'))
                    file = discord.File(file_bytes, filename="web_response.txt")
                    
                    # Send a short message with the file attachment
                    await interaction.followup.send(
                        f"The response from analyzing {url} is too long for Discord (>{len(reply)} characters). Here's the full response as a text file:", 
                        file=file
                    )
                else:
                    # Send as normal message if within limits
                    await interaction.followup.send(reply)

            except Exception as e:
                await interaction.followup.send(f"Error: {str(e)}", ephemeral=True)
        
        await process_request(interaction, process_web, url)

    @tree.command(name='generate', description='Generates an image from a text prompt.')
    @app_commands.describe(prompt='The prompt for image generation')
    @check_blacklist()
    async def generate_image_command(interaction: discord.Interaction, prompt: str):
        """Generates an image from a text prompt."""
        await interaction.response.defer(thinking=True)  # Indicate that the bot is processing
        
        async def process_image_generation(interaction: discord.Interaction, prompt: str):
            try:
                # Generate images
                result = await image_generator.generate_image(prompt, 4)  # Generate 4 images
                
                if not result['success']:
                    await interaction.followup.send(f"Error: {result.get('error', 'Unknown error')}")
                    return
                
                # Send images as attachments
                if result["binary_images"]:
                    await interaction.followup.send(
                        f"Generated {len(result['binary_images'])} images for prompt: \"{prompt}\"",
                        files=[discord.File(io.BytesIO(img), filename=f"image_{i}.png") 
                               for i, img in enumerate(result["binary_images"])]
                    )
                else:
                    await interaction.followup.send("No images were generated.")
                    
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                logging.error(f"Error in generate_image_command: {error_message}")
                await interaction.followup.send(error_message)
        
        await process_request(interaction, process_image_generation, prompt)

    @tree.command(name="reset", description="Reset the bot by clearing user data.")
    @check_blacklist()
    async def reset(interaction: discord.Interaction):
        """Resets the bot by clearing user data."""
        user_id = interaction.user.id
        await db_handler.save_history(user_id, [])
        await interaction.response.send_message("Your conversation history has been cleared and reset!", ephemeral=True)

    @tree.command(name="user_stat", description="Get your current input token, output token, and model.")
    @check_blacklist()
    async def user_stat(interaction: discord.Interaction):
        """Fetches and displays the current input token, output token, and model for the user."""
        await interaction.response.defer(thinking=True, ephemeral=True)
        
        async def process_user_stat(interaction: discord.Interaction):
            import tiktoken
            
            user_id = interaction.user.id
            history = await db_handler.get_history(user_id)
            model = await db_handler.get_user_model(user_id) or "openai/gpt-4.1-mini"  # Default model

            # Adjust model for encoding purposes
            if model in ["openai/gpt-4o", "openai/o1", "openai/o1-preview", "openai/o1-mini", "openai/o3-mini"]:
                encoding_model = "openai/gpt-4o"
            else:
                encoding_model = model

            # Retrieve the appropriate encoding for the selected model
            encoding = tiktoken.get_encoding("o200k_base")

            # Initialize token counts
            input_tokens = 0
            output_tokens = 0

            # Calculate input and output tokens
            if history:
                for item in history:
                    content = item.get('content', '')

                    # Handle case where content is a list or other type
                    if isinstance(content, list):
                        content_str = ""
                        for part in content:
                            if isinstance(part, dict) and 'text' in part:
                                content_str += part['text'] + " "
                        content = content_str

                    # Ensure content is a string before processing
                    if isinstance(content, str):
                        tokens = len(encoding.encode(content))
                        if item.get('role') == 'user':
                            input_tokens += tokens
                        elif item.get('role') == 'assistant':
                            output_tokens += tokens

            # Create the statistics message
            stat_message = (
                f"**User Statistics:**\n"
                f"Model: `{model}`\n"
                f"Input Tokens: `{input_tokens}`\n"
                f"Output Tokens: `{output_tokens}`\n"
            )

            # Send the response
            await interaction.followup.send(stat_message, ephemeral=True)
        
        await process_request(interaction, process_user_stat)

    @tree.command(name="help", description="Display a list of available commands.")
    @check_blacklist()
    async def help_command(interaction: discord.Interaction):
        """Sends a list of available commands to the user."""
        help_message = (
            "**Available commands:**\n"
            "/choose_model - Select which AI model to use for responses (openai/gpt-4o, openai/gpt-4o-mini, openai/o1-preview, openai/o1-mini).\n"
            "/search `<query>` - Search Google and send results to the AI model.\n"
            "/web `<url>` - Scrape a webpage and send the data to the AI model.\n"
            "/generate `<prompt>` - Generate an image from a text prompt.\n"
            "/reset - Reset your chat history.\n"
            "/user_stat - Get information about your input tokens, output tokens, and current model.\n"
            "/help - Display this help message.\n"
            "\n"
            "**🆕 New Enhanced Features:**\n"
            "/smart_model `<task>` - Get AI model recommendations for your task.\n"
            "/preferences `<action>` - Manage your personal settings and preferences.\n"
            "/conversation_stats - View your conversation statistics and health.\n"
            "/process_file `<file>` - Process various file types (Word, Excel, code, etc.).\n"
            "/help_enhanced - Detailed help with feature discovery.\n"
            "\n"
            "💡 **Try `/help_enhanced` for detailed guides and tips!**"
        )
        await interaction.response.send_message(help_message, ephemeral=True)

    @tree.command(name="stop", description="Stop any process or queue of the user. Admins can stop other users' tasks by providing their ID.")
    @app_commands.describe(user_id="The Discord user ID to stop tasks for (admin only)")
    @check_blacklist()
    async def stop(interaction: discord.Interaction, user_id: str = None):
        """Stops any process or queue of the user. Admins can stop other users' tasks by providing their ID."""
        # Defer the interaction first
        await interaction.response.defer(ephemeral=True)
        
        if user_id and not await db_handler.is_admin(interaction.user.id):
            await interaction.followup.send("You don't have permission to stop other users' tasks.", ephemeral=True)
            return
        
        target_user_id = int(user_id) if user_id else interaction.user.id
        await stop_user_tasks(target_user_id)
        await interaction.followup.send(f"Stopped all tasks for user {target_user_id}.", ephemeral=True)

    # Admin commands
    @tree.command(name="whitelist_add", description="Add a user to the PDF processing whitelist")
    @app_commands.describe(user_id="The Discord user ID to whitelist")
    async def whitelist_add(interaction: discord.Interaction, user_id: str):
        """Adds a user to the PDF processing whitelist."""
        if not await db_handler.is_admin(interaction.user.id):
            await interaction.response.send_message("You don't have permission to use this command. Only admin can use whitelist commands.", ephemeral=True)
            return
        
        try:
            user_id = int(user_id)
            if await db_handler.is_admin(user_id):
                await interaction.response.send_message("Admins are automatically whitelisted and don't need to be added.", ephemeral=True)
                return
            await db_handler.add_user_to_whitelist(user_id)
            await interaction.response.send_message(f"User {user_id} has been added to the PDF processing whitelist.", ephemeral=True)
        except ValueError:
            await interaction.response.send_message("Invalid user ID. Please provide a valid Discord user ID.", ephemeral=True)

    @tree.command(name="whitelist_remove", description="Remove a user from the PDF processing whitelist")
    @app_commands.describe(user_id="The Discord user ID to remove from whitelist")
    async def whitelist_remove(interaction: discord.Interaction, user_id: str):
        """Removes a user from the PDF processing whitelist."""
        if not await db_handler.is_admin(interaction.user.id):
            await interaction.response.send_message("You don't have permission to use this command. Only admin can use whitelist commands.", ephemeral=True)
            return
        
        try:
            user_id = int(user_id)
            if await db_handler.remove_user_from_whitelist(user_id):
                await interaction.response.send_message(f"User {user_id} has been removed from the PDF processing whitelist.", ephemeral=True)
            else:
                await interaction.response.send_message(f"User {user_id} was not found in the whitelist.", ephemeral=True)
        except ValueError:
            await interaction.response.send_message("Invalid user ID. Please provide a valid Discord user ID.", ephemeral=True)

    @tree.command(name="blacklist_add", description="Add a user to the bot blacklist")
    @app_commands.describe(user_id="The Discord user ID to blacklist")
    async def blacklist_add(interaction: discord.Interaction, user_id: str):
        """Adds a user to the bot blacklist."""
        if not await db_handler.is_admin(interaction.user.id):
            await interaction.response.send_message("You don't have permission to use this command. Only admin can use blacklist commands.", ephemeral=True)
            return
        
        try:
            user_id = int(user_id)
            if await db_handler.is_admin(user_id):
                await interaction.response.send_message("Cannot blacklist an admin.", ephemeral=True)
                return
            await db_handler.add_user_to_blacklist(user_id)
            await interaction.response.send_message(f"User {user_id} has been added to the bot blacklist. They can no longer use any bot features.", ephemeral=True)
        except ValueError:
            await interaction.response.send_message("Invalid user ID. Please provide a valid Discord user ID.", ephemeral=True)

    @tree.command(name="blacklist_remove", description="Remove a user from the bot blacklist")
    @app_commands.describe(user_id="The Discord user ID to remove from blacklist")
    async def blacklist_remove(interaction: discord.Interaction, user_id: str):
        """Removes a user from the bot blacklist."""
        if not await db_handler.is_admin(interaction.user.id):
            await interaction.response.send_message("You don't have permission to use this command. Only admin can use blacklist commands.", ephemeral=True)
            return
        
        try:
            user_id = int(user_id)
            if await db_handler.remove_user_from_blacklist(user_id):
                await interaction.response.send_message(f"User {user_id} has been removed from the bot blacklist. They can now use bot features again.", ephemeral=True)
            else:
                await interaction.response.send_message(f"User {user_id} was not found in the blacklist.", ephemeral=True)
        except ValueError:
            await interaction.response.send_message("Invalid user ID. Please provide a valid Discord user ID.", ephemeral=True)

    # Helper function to stop user tasks
    async def stop_user_tasks(user_id: int):
        """Stop all tasks for a specific user."""
        logging.info(f"Stopping all tasks for user {user_id}")
        
        # Cancel all active tasks in user_tasks
        if user_id in user_tasks:
            for task in user_tasks[user_id]:
                try:
                    task.cancel()
                    logging.info(f"Cancelled task for user {user_id}")
                except Exception as e:
                    logging.error(f"Error cancelling task: {str(e)}")
            user_tasks[user_id] = []
        
        # Clear any queued requests
        if user_id in user_requests:
            queue_size = user_requests[user_id]['queue'].qsize()
            while not user_requests[user_id]['queue'].empty():
                try:
                    user_requests[user_id]['queue'].get_nowait()
                    user_requests[user_id]['queue'].task_done()
                except Exception as e:
                    logging.error(f"Error clearing queue: {str(e)}")
            logging.info(f"Cleared {queue_size} queued requests for user {user_id}")
        
        # Also notify the message handler to stop any running PDF processes
        # This is important for PDF batch processing which might be running in separate tasks
        try:
            # Import here to avoid circular imports
            from src.module.message_handler import MessageHandler
            if hasattr(MessageHandler, 'stop_user_tasks'):
                await MessageHandler.stop_user_tasks(user_id)
                logging.info(f"Called MessageHandler.stop_user_tasks for user {user_id}")
        except Exception as e:
            logging.error(f"Error stopping message handler tasks: {str(e)}")

    # ==================== NEW ENHANCED COMMANDS ====================
    
    @tree.command(name="smart_model", description="Get AI model suggestions based on your task type.")
    @app_commands.describe(task="Describe what you want to do")
    @check_blacklist()
    async def smart_model(interaction: discord.Interaction, task: str):
        """Suggest the best AI model for a specific task."""
        await interaction.response.defer(thinking=True, ephemeral=True)
        
        try:
            user_id = interaction.user.id
            user_prefs = await user_prefs_manager.get_user_preferences(user_id)
            
            # Get model suggestion
            suggestion = model_selector.suggest_model_with_alternatives(
                task, 
                user_prefs.get('preferred_model')
            )
            
            # Format response
            response_lines = [
                f"**🎯 Smart Model Suggestion for:** `{task[:100]}{'...' if len(task) > 100 else ''}`",
                "",
                f"**📋 Task Type Detected:** `{suggestion['task_type'].title()}`",
                "",
                f"**🤖 Recommended Model:** `{suggestion['suggested_model']}`",
                f"**💡 Why:** {suggestion['reason']}",
                f"**📝 Details:** {suggestion['explanation']}",
                ""
            ]
            
            if suggestion['alternatives']:
                response_lines.append("**🔄 Alternative Models:**")
                for alt in suggestion['alternatives']:
                    response_lines.append(f"  • `{alt['model']}` - {alt['explanation']}")
                response_lines.append("")
            
            response_lines.extend([
                "*💡 Tip: Use `/preferences set preferred_model` to set a default model*",
                "*🔧 Use `/choose_model` to select a model for your conversations*"
            ])
            
            await interaction.followup.send("\n".join(response_lines), ephemeral=True)
            
        except Exception as e:
            await interaction.followup.send(f"Error analyzing task: {str(e)}", ephemeral=True)

    @tree.command(name="preferences", description="Manage your personal bot preferences and settings.")
    @app_commands.describe(
        action="Action to perform",
        setting="Setting to modify",
        value="New value for the setting"
    )
    @app_commands.choices(action=[
        app_commands.Choice(name="view", value="view"),
        app_commands.Choice(name="set", value="set"),
        app_commands.Choice(name="reset", value="reset")
    ])
    @check_blacklist()
    async def preferences(
        interaction: discord.Interaction, 
        action: str, 
        setting: str = None, 
        value: str = None
    ):
        """Manage user preferences."""
        await interaction.response.defer(thinking=True, ephemeral=True)
        
        try:
            user_id = interaction.user.id
            
            if action == "view":
                prefs = await user_prefs_manager.get_user_preferences(user_id)
                formatted_prefs = user_prefs_manager.format_preferences_display(prefs)
                await interaction.followup.send(formatted_prefs, ephemeral=True)
                
            elif action == "set":
                if not setting or not value:
                    await interaction.followup.send(
                        "❌ Please provide both setting and value.\n"
                        "Example: `/preferences set response_style detailed`\n\n"
                        "Available settings:\n"
                        "• `preferred_model` - Your default AI model\n"
                        "• `response_style` - balanced, concise, detailed\n"
                        "• `auto_model_selection` - true, false\n"
                        "• `show_model_suggestions` - true, false\n"
                        "• `enable_conversation_summary` - true, false\n"
                        "• `max_response_length` - short, medium, long\n"
                        "• `language` - auto, en, es, fr, de, etc.\n"
                        "• `timezone` - Your timezone (e.g., America/New_York)",
                        ephemeral=True
                    )
                    return
                
                # Convert string values to appropriate types
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                
                success = await user_prefs_manager.set_preference(user_id, setting, value)
                
                if success:
                    await interaction.followup.send(
                        f"✅ Successfully updated `{setting}` to `{value}`", 
                        ephemeral=True
                    )
                else:
                    await interaction.followup.send(
                        f"❌ Failed to update setting. Please check the setting name and value.", 
                        ephemeral=True
                    )
                    
            elif action == "reset":
                success = await user_prefs_manager.reset_preferences(user_id)
                
                if success:
                    await interaction.followup.send(
                        "✅ Your preferences have been reset to defaults.", 
                        ephemeral=True
                    )
                else:
                    await interaction.followup.send(
                        "❌ Failed to reset preferences.", 
                        ephemeral=True
                    )
                    
        except Exception as e:
            logging.error(f"Error in preferences command: {str(e)}")
            await interaction.followup.send(f"❌ Error managing preferences: {str(e)}", ephemeral=True)

    @tree.command(name="conversation_stats", description="Get statistics about your current conversation.")
    @check_blacklist()
    async def conversation_stats(interaction: discord.Interaction):
        """Show conversation statistics."""
        await interaction.response.defer(thinking=True, ephemeral=True)
        
        try:
            user_id = interaction.user.id
            history = await db_handler.get_history(user_id)
            
            if not history:
                await interaction.followup.send(
                    "📊 **Conversation Statistics**\n\n"
                    "No conversation history found. Start chatting to see statistics!",
                    ephemeral=True
                )
                return
            
            stats = await conversation_summarizer.get_conversation_stats(history)
            
            response_lines = [
                "📊 **Your Conversation Statistics**",
                "",
                f"💬 **Total Messages:** {stats['total_messages']}",
                f"👤 **Your Messages:** {stats['user_messages']}",
                f"🤖 **Bot Responses:** {stats['assistant_messages']}",
                f"📝 **Summaries:** {stats['summary_messages']}",
                "",
                f"🔤 **Token Usage:** {stats['total_tokens']:,} tokens",
                f"📏 **Context Limit:** {stats['token_limit']:,} tokens",
                "",
                f"📊 **Status:** {'⚠️ Needs summarization' if stats['needs_summary'] else '✅ Within limits'}",
                "",
                "*💡 Long conversations are automatically summarized to maintain context quality*"
            ]
            
            await interaction.followup.send("\n".join(response_lines), ephemeral=True)
            
        except Exception as e:
            logging.error(f"Error getting conversation stats: {str(e)}")
            await interaction.followup.send(f"❌ Error getting statistics: {str(e)}", ephemeral=True)

    @tree.command(name="process_file", description="Process and analyze various file types (documents, data, code).")
    @app_commands.describe(file="Upload a file to process and analyze")
    @check_blacklist()
    async def process_file(interaction: discord.Interaction, file: discord.Attachment):
        """Process various file types with enhanced capabilities."""
        await interaction.response.defer(thinking=True)
        
        async def process_uploaded_file(interaction: discord.Interaction, file: discord.Attachment):
            try:
                # Check if file type is supported
                if not enhanced_file_processor.is_supported(file.filename):
                    supported_types = ", ".join(enhanced_file_processor.get_supported_extensions())
                    await interaction.followup.send(
                        f"❌ **Unsupported file type:** `{file.filename}`\n\n"
                        f"**Supported types:** {supported_types}\n\n"
                        "*💡 Tip: For PDF files, use the regular file upload feature*"
                    )
                    return
                
                # Check file size (limit to 10MB)
                if file.size > 10 * 1024 * 1024:  # 10MB
                    await interaction.followup.send(
                        f"❌ **File too large:** {file.size / (1024*1024):.1f}MB\n"
                        "Maximum supported size: 10MB"
                    )
                    return
                
                # Download and process the file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                    await file.save(tmp_file.name)
                    
                    # Process the file
                    result = await enhanced_file_processor.process_file(tmp_file.name, file.filename)
                    
                    # Clean up
                    os.unlink(tmp_file.name)
                
                if not result['success']:
                    await interaction.followup.send(f"❌ **Processing failed:** {result['error']}")
                    return
                
                # Format the response
                metadata = result['metadata']
                content = result['content']
                
                response_lines = [
                    f"📄 **File Analysis: {metadata['filename']}**",
                    "",
                    f"📊 **File Info:**",
                    f"  • Type: `{metadata['type'].upper()}`",
                    f"  • Size: `{metadata['size_human']}`",
                    f"  • Processor: `{metadata['processor']}`",
                    ""
                ]
                
                # Add type-specific metadata
                if metadata['type'] == 'csv':
                    response_lines.extend([
                        f"📈 **Data Info:**",
                        f"  • Rows: `{metadata['rows']}`",
                        f"  • Columns: `{metadata['columns']}`",
                        f"  • Has null values: `{'Yes' if metadata['has_null_values'] else 'No'}`",
                        ""
                    ])
                elif metadata['type'] == 'excel':
                    response_lines.extend([
                        f"📊 **Excel Info:**",
                        f"  • Sheets: `{metadata['sheet_count']}`",
                        f"  • Sheet names: `{', '.join(metadata['sheets'])}`",
                        ""
                    ])
                elif metadata['type'] == 'code':
                    response_lines.extend([
                        f"💻 **Code Info:**",
                        f"  • Language: `{metadata['language']}`",
                        f"  • Total lines: `{metadata['total_lines']}`",
                        f"  • Code lines: `{metadata['code_lines']}`",
                        f"  • Comment lines: `{metadata['comment_lines']}`",
                        ""
                    ])
                
                response_text = "\n".join(response_lines)
                
                # Send response with file content
                if len(content) <= 1500:  # Show content directly if short
                    response_text += f"**📝 Content Preview:**\n```\n{content[:1500]}\n```"
                    await interaction.followup.send(response_text)
                else:
                    # Send as file attachment if too long
                    content_file = io.BytesIO(content.encode('utf-8'))
                    discord_file = discord.File(content_file, filename=f"processed_{file.filename}.txt")
                    
                    response_text += "*📎 Full content attached as file*"
                    await interaction.followup.send(response_text, file=discord_file)
                
            except Exception as e:
                logging.error(f"Error processing file {file.filename}: {str(e)}")
                await interaction.followup.send(f"❌ **Error processing file:** {str(e)}")
        
        await process_request(interaction, process_uploaded_file, file)

    @tree.command(name="help_enhanced", description="Discover advanced features and get detailed help.")
    @app_commands.describe(category="Help category to explore")
    @app_commands.choices(category=[
        app_commands.Choice(name="🆕 New Features", value="new"),
        app_commands.Choice(name="🤖 AI Models", value="models"),
        app_commands.Choice(name="⚙️ Preferences", value="preferences"),
        app_commands.Choice(name="📁 File Processing", value="files"),
        app_commands.Choice(name="🔧 All Commands", value="all"),
        app_commands.Choice(name="💡 Tips & Tricks", value="tips")
    ])
    @check_blacklist()
    async def help_enhanced(interaction: discord.Interaction, category: str = "all"):
        """Enhanced help system with feature discovery."""
        await interaction.response.defer(thinking=True, ephemeral=True)
        
        try:
            if category == "new":
                help_text = """
🆕 **New Enhanced Features**

**🧠 Smart Model Selection**
• `/smart_model` - Get AI model recommendations based on your task
• Automatically suggests the best model for coding, analysis, creative tasks, etc.

**⚙️ Personalization**
• `/preferences` - Customize your bot experience
• Set preferred models, response styles, and behavior
• Auto-conversation summarization to maintain context

**📊 Analytics**
• `/conversation_stats` - See your conversation statistics
• Track token usage and conversation health

**📁 Enhanced File Processing**
• `/process_file` - Support for Word docs, Excel, PowerPoint, code files, etc.
• Better analysis and content extraction

**💡 Smarter Conversations**
• Automatic conversation summarization for long chats
• Better context management and memory
"""

            elif category == "models":
                help_text = """
🤖 **AI Model Guide**

**🧠 Reasoning Models (Best for complex problems):**
• `openai/o1-preview` - Advanced reasoning and step-by-step problem solving
• `openai/o1` - Enhanced reasoning for analytical tasks
• `openai/o1-mini` - Fast reasoning for structured problems

**🎯 Balanced Models (Great for everything):**
• `openai/gpt-4o` - Excellent for coding, analysis, creativity
• `openai/gpt-4o-mini` - Fast and efficient for general tasks

**⚡ Speed Models:**
• `openai/gpt-4.1-mini` - Compact with great performance
• `openai/gpt-4.1-nano` - Ultra-fast for simple tasks

**🔧 How to Choose:**
• Use `/smart_model` to get suggestions based on your task
• Set a default with `/preferences set preferred_model <model>`
• Use `/choose_model` to select for current conversation
"""

            elif category == "preferences":
                help_text = """
⚙️ **Preferences System**

**📋 Available Settings:**
• `preferred_model` - Your default AI model
• `auto_model_selection` - Enable smart model suggestions
• `response_style` - balanced, concise, detailed
• `show_model_suggestions` - Show why a model was chosen
• `enable_conversation_summary` - Auto-summarize long chats
• `max_response_length` - short, medium, long
• `language` - Response language (auto-detect or specific)
• `timezone` - For reminders and timestamps

**🔧 Commands:**
• `/preferences view` - See all your settings
• `/preferences set <setting> <value>` - Change a setting
• `/preferences reset` - Reset to defaults

**💡 Examples:**
• `/preferences set response_style detailed`
• `/preferences set preferred_model openai/gpt-4o`
• `/preferences set auto_model_selection true`
"""

            elif category == "files":
                help_text = f"""
📁 **Enhanced File Processing**

**📄 Supported File Types:**
{', '.join([f'`{ext}`' for ext in enhanced_file_processor.get_supported_extensions()])}

**🔧 Features:**
• **Documents:** Word (.docx), PowerPoint (.pptx), PDF
• **Data:** CSV, Excel (.xlsx), JSON, YAML
• **Code:** Python, JavaScript, HTML, CSS, and more
• **Text:** Markdown, plain text, logs

**💡 How to Use:**
• `/process_file` - Upload and analyze any supported file
• Regular file upload - For PDF analysis (existing feature)
• Drag & drop files in chat for automatic processing

**📊 What You Get:**
• Content extraction and analysis
• Metadata and statistics
• Structure analysis for data files
• Code metrics for programming files
"""

            elif category == "tips":
                help_text = """
💡 **Tips & Tricks**

**🎯 Getting Better Results:**
• Be specific about your task type for better model suggestions
• Use `/smart_model` before complex tasks to get the right model
• Set preferences once to customize your experience

**⚡ Efficiency Tips:**
• Enable auto-model selection for optimal performance
• Use conversation summaries for long discussions
• Check `/conversation_stats` to monitor context usage

**🔧 Power User Features:**
• Combine multiple file types in analysis
• Use preferences to match your workflow
• Try different response styles for different tasks

**📊 Monitoring:**
• Use `/user_stat` for token usage tracking
• Check `/conversation_stats` for conversation health
• Monitor your preferences with `/preferences view`

**🎨 Creative Workflows:**
• Use detailed response style for creative writing
• Try different models for different creative tasks
• Experiment with image generation prompts
"""

            else:  # "all"
                help_text = """
🔧 **Complete Command Reference**

**💬 Core Chat Commands:**
• `/choose_model` - Select AI model for responses
• `/search <query>` - Google search with AI analysis
• `/web <url>` - Scrape and analyze web content
• `/generate <prompt>` - Generate images from text
• `/reset` - Clear conversation history

**🆕 Enhanced Features:**
• `/smart_model <task>` - Get model recommendations
• `/preferences <action>` - Manage personal settings
• `/conversation_stats` - View conversation analytics
• `/process_file <file>` - Analyze various file types
• `/help_enhanced` - This detailed help system

**📊 Statistics & Info:**
• `/user_stat` - Your token usage and current model
• `/help` - Basic command list

**👑 Admin Commands:**
• `/whitelist_add/remove` - Manage PDF whitelist
• `/blacklist_add/remove` - Manage user access
• `/stop` - Stop user processes

**💡 Pro Tips:**
• Upload files directly in chat for automatic processing
• Use @ mentions to get the bot's attention
• Try different models for different types of tasks
• Set your preferences for a personalized experience
"""

            await interaction.followup.send(help_text, ephemeral=True)
            
        except Exception as e:
            logging.error(f"Error in help_enhanced: {str(e)}")
            await interaction.followup.send("❌ Error loading help content.", ephemeral=True)