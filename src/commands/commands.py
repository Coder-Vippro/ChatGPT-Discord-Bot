import discord
from discord import app_commands
from discord.ext import commands
import logging
import io
import asyncio
from typing import Optional, Dict, List, Any, Callable

from src.config.config import MODEL_OPTIONS, PDF_ALLOWED_MODELS, DEFAULT_MODEL
from src.config.pricing import MODEL_PRICING, calculate_cost, format_cost
from src.utils.image_utils import ImageGenerator
from src.utils.web_utils import google_custom_search, scrape_web_content
from src.utils.pdf_utils import process_pdf, send_response
from src.utils.openai_utils import prepare_file_from_path
from src.utils.token_counter import token_counter
from src.utils.code_interpreter import delete_all_user_files
from src.utils.discord_utils import create_info_embed, create_error_embed, create_success_embed
from src.utils.claude_utils import is_claude_model, call_claude_api

# Dictionary to keep track of user requests and their cooldowns
user_requests: Dict[int, Dict[str, Any]] = {}
# Dictionary to store user tasks
user_tasks: Dict[int, List] = {}


# ============================================================
# Autocomplete Functions
# ============================================================

async def model_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> List[app_commands.Choice[str]]:
    """
    Autocomplete function for model selection.
    Provides filtered model suggestions based on user input.
    """
    # Filter models based on current input
    matches = [
        model for model in MODEL_OPTIONS
        if current.lower() in model.lower()
    ]
    
    # If no matches, show all models
    if not matches:
        matches = MODEL_OPTIONS
    
    # Return up to 25 choices (Discord limit)
    return [
        app_commands.Choice(name=model, value=model)
        for model in matches[:25]
    ]


async def image_model_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> List[app_commands.Choice[str]]:
    """
    Autocomplete function for image generation model selection.
    """
    image_models = ["flux", "flux-dev", "sdxl", "realistic", "anime", "dreamshaper"]
    matches = [m for m in image_models if current.lower() in m.lower()]
    
    if not matches:
        matches = image_models
    
    return [
        app_commands.Choice(name=model, value=model)
        for model in matches[:25]
    ]

def setup_commands(bot: commands.Bot, db_handler, openai_client, image_generator: ImageGenerator, anthropic_client=None):
    """
    Set up all slash commands for the bot.
    
    Args:
        bot: Discord bot instance
        db_handler: Database handler instance
        openai_client: OpenAI client instance
        image_generator: Image generator instance
        anthropic_client: Anthropic client instance (optional, for Claude models)
    """
    tree = bot.tree
    
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
        """Lets users choose an AI model using a dropdown menu."""
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

    @tree.command(name="set_model", description="Set AI model directly with autocomplete suggestions.")
    @app_commands.describe(model="The AI model to use (type to search)")
    @app_commands.autocomplete(model=model_autocomplete)
    @check_blacklist()
    async def set_model(interaction: discord.Interaction, model: str):
        """Sets the AI model directly using autocomplete."""
        user_id = interaction.user.id
        
        # Validate the model is in the allowed list
        if model not in MODEL_OPTIONS:
            # Find close matches for suggestions
            close_matches = [m for m in MODEL_OPTIONS if model.lower() in m.lower()]
            if close_matches:
                suggestions = ", ".join(f"`{m}`" for m in close_matches[:5])
                await interaction.response.send_message(
                    f"❌ Invalid model `{model}`. Did you mean: {suggestions}?",
                    ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    f"❌ Invalid model `{model}`. Use `/choose_model` to see available options.",
                    ephemeral=True
                )
            return
        
        # Save the model selection
        await db_handler.save_user_model(user_id, model)
        
        # Get pricing info for the selected model
        pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
        
        await interaction.response.send_message(
            f"✅ Model set to `{model}`\n"
            f"💰 Pricing: ${pricing['input']:.2f}/1M input, ${pricing['output']:.2f}/1M output",
            ephemeral=True
        )

    @tree.command(name="search", description="Search on Google and send results to AI model.")
    @app_commands.describe(query="The search query")
    @check_blacklist()
    async def search(interaction: discord.Interaction, query: str):
        """Searches Google and sends results to the AI model."""
        await interaction.response.defer(thinking=True)
        
        async def process_search(interaction: discord.Interaction, query: str):
            user_id = interaction.user.id
            model = await db_handler.get_user_model(user_id) or DEFAULT_MODEL
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

                # Check context limit before sending
                context_check = await token_counter.check_context_limit(messages, model)
                
                if not context_check["within_limit"]:
                    await interaction.followup.send(
                        f"⚠️ Search results are too large ({context_check['input_tokens']:,} tokens). "
                        f"Maximum context is {context_check['max_tokens']:,} tokens. "
                        "Please try a more specific search query.",
                        ephemeral=True
                    )
                    return
                
                # Count input tokens before API call
                input_token_count = await token_counter.count_message_tokens(messages, model)
                
                logging.info(
                    f"Search request - User: {user_id}, Model: {model}, "
                    f"Input tokens: {input_token_count['total_tokens']} "
                    f"(text: {input_token_count['text_tokens']}, images: {input_token_count['image_tokens']})"
                )

                # Check if using Claude model
                if is_claude_model(model):
                    if anthropic_client is None:
                        await interaction.followup.send(
                            "❌ Claude model not available. ANTHROPIC_API_KEY is not configured.",
                            ephemeral=True
                        )
                        return
                    
                    # Use Claude API
                    claude_response = await call_claude_api(
                        anthropic_client,
                        messages,
                        model,
                        max_tokens=4096,
                        use_tools=False
                    )
                    
                    reply = claude_response.get("content", "")
                    actual_input_tokens = claude_response.get("input_tokens", 0)
                    actual_output_tokens = claude_response.get("output_tokens", 0)
                else:
                    # Send to the OpenAI model
                    api_params = {
                        "model": model if model in ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-5", "openai/gpt-5-nano", "openai/gpt-5-mini", "openai/gpt-5-chat"] else "openai/gpt-4o",
                        "messages": messages
                    }
                    
                    # Add temperature only for models that support it (exclude GPT-5 family)
                    if model not in ["openai/gpt-5", "openai/gpt-5-nano", "openai/gpt-5-mini", "openai/gpt-5-chat"]:
                        api_params["temperature"] = 0.5
                    
                    response = await openai_client.chat.completions.create(**api_params)

                    reply = response.choices[0].message.content
                    
                    # Get actual token usage from API response
                    usage = response.usage
                    actual_input_tokens = usage.prompt_tokens if usage else input_token_count['total_tokens']
                    actual_output_tokens = usage.completion_tokens if usage else token_counter.count_text_tokens(reply, model)
                
                # Calculate cost
                cost = token_counter.estimate_cost(actual_input_tokens, actual_output_tokens, model)
                
                # Update database with detailed token info
                await db_handler.save_token_usage(
                    user_id=user_id,
                    model=model,
                    input_tokens=actual_input_tokens,
                    output_tokens=actual_output_tokens,
                    cost=cost,
                    text_tokens=input_token_count['text_tokens'],
                    image_tokens=input_token_count['image_tokens']
                )
                
                logging.info(
                    f"Search completed - User: {user_id}, "
                    f"Input: {actual_input_tokens}, Output: {actual_output_tokens}, "
                    f"Cost: ${cost:.6f}"
                )
                
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
                        f"The search response for '{query}' is too long ({len(reply):,} characters). "
                        f"Full response attached.\n💰 Cost: ${cost:.6f}", 
                        file=file
                    )
                else:
                    # Send as normal message if within limits
                    await interaction.followup.send(f"{reply}\n\n💰 Cost: ${cost:.6f}")

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
            model = await db_handler.get_user_model(user_id) or DEFAULT_MODEL
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

                # Check if using Claude model
                if is_claude_model(model):
                    if anthropic_client is None:
                        await interaction.followup.send(
                            "❌ Claude model not available. ANTHROPIC_API_KEY is not configured.",
                            ephemeral=True
                        )
                        return
                    
                    # Use Claude API
                    claude_response = await call_claude_api(
                        anthropic_client,
                        messages,
                        model,
                        max_tokens=4096,
                        use_tools=False
                    )
                    reply = claude_response.get("content", "")
                else:
                    api_params = {
                        "model": model if model in ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-5", "openai/gpt-5-nano", "openai/gpt-5-mini", "openai/gpt-5-chat"] else "openai/gpt-4o",
                        "messages": messages
                    }
                    
                    # Add temperature and top_p only for models that support them (exclude GPT-5 family)
                    if model not in ["openai/gpt-5", "openai/gpt-5-nano", "openai/gpt-5-mini", "openai/gpt-5-chat"]:
                        api_params["temperature"] = 0.3
                        api_params["top_p"] = 0.7
                    
                    response = await openai_client.chat.completions.create(**api_params)

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

    @tree.command(name="reset", description="Reset the bot by clearing user data and token usage statistics.")
    @check_blacklist()
    async def reset(interaction: discord.Interaction):
        """Resets the bot by clearing user data and files."""
        user_id = interaction.user.id
        
        # Clear conversation history
        await db_handler.save_history(user_id, [])
        
        # Reset token statistics
        await db_handler.reset_user_token_stats(user_id)
        
        # Delete all user files (from disk and database)
        result = await delete_all_user_files(user_id, db_handler)
        
        # Build response message
        message = "✅ Your conversation history and token usage statistics have been cleared and reset!"
        
        if result.get('success') and result.get('deleted_count', 0) > 0:
            message += f"\n🗑️ Deleted {result['deleted_count']} file(s)."
        elif result.get('success'):
            message += "\n📁 No files to delete."
        else:
            message += f"\n⚠️ Warning: Could not delete some files. {result.get('error', '')}"
        
        await interaction.response.send_message(message, ephemeral=True)

    @tree.command(name="user_stat", description="Get your current token usage, costs, and model.")
    @check_blacklist()
    async def user_stat(interaction: discord.Interaction):
        """Fetches and displays the current token usage, costs, and model for the user."""
        await interaction.response.defer(thinking=True, ephemeral=True)
        
        async def process_user_stat(interaction: discord.Interaction):
            user_id = interaction.user.id
            model = await db_handler.get_user_model(user_id) or DEFAULT_MODEL

            # Get token usage from database
            token_stats = await db_handler.get_user_token_usage(user_id)
            
            total_input_tokens = token_stats.get('total_input_tokens', 0)
            total_output_tokens = token_stats.get('total_output_tokens', 0)
            total_text_tokens = token_stats.get('total_text_tokens', 0)
            total_image_tokens = token_stats.get('total_image_tokens', 0)
            total_cost = token_stats.get('total_cost', 0.0)
            
            # Get usage by model for detailed breakdown
            model_usage = await db_handler.get_user_token_usage_by_model(user_id)
            
            # Create the statistics message
            stat_message = (
                f"**📊 User Statistics**\n"
                f"Current Model: `{model}`\n\n"
                f"**Token Usage:**\n"
                f"• Total Input: `{total_input_tokens:,}` tokens\n"
                f"  ├─ Text: `{total_text_tokens:,}` tokens\n"
                f"  └─ Images: `{total_image_tokens:,}` tokens\n"
                f"• Total Output: `{total_output_tokens:,}` tokens\n"
                f"• Combined: `{total_input_tokens + total_output_tokens:,}` tokens\n\n"
                f"**💰 Total Cost: `${total_cost:.6f}`**\n\n"
            )
            
            # Add breakdown by model if available
            if model_usage:
                stat_message += "**Per-Model Breakdown:**\n"
                for model_name, usage in sorted(
                    model_usage.items(), 
                    key=lambda x: x[1].get('cost', 0), 
                    reverse=True
                )[:10]:
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    text_tokens = usage.get('text_tokens', 0)
                    image_tokens = usage.get('image_tokens', 0)
                    cost = usage.get('cost', 0.0)
                    requests = usage.get('requests', 0)
                    
                    model_short = model_name.replace('openai/', '')
                    stat_message += (
                        f"`{model_short}`\n"
                        f"  • {requests:,} requests, ${cost:.6f}\n"
                        f"  • In: {input_tokens:,} ({text_tokens:,} text + {image_tokens:,} img)\n"
                        f"  • Out: {output_tokens:,}\n"
                    )

            # Send the response
            await interaction.followup.send(stat_message, ephemeral=True)
        
        await process_request(interaction, process_user_stat)

    @tree.command(name="prices", description="Display pricing information for all available AI models.")
    @check_blacklist()
    async def prices_command(interaction: discord.Interaction):
        """Displays pricing information for all available AI models."""
        await interaction.response.defer(thinking=True, ephemeral=True)
        
        async def process_prices(interaction: discord.Interaction):
            # Create the pricing message
            pricing_message = (
                "**💰 Model Pricing (per 1M tokens)**\n"
                "```\n"
                f"{'Model':<20} {'Input':<8} {'Output':<8}\n"
                f"{'-' * 40}\n"
            )
            
            for model, pricing in MODEL_PRICING.items():
                model_short = model.replace("openai/", "")
                pricing_message += f"{model_short:<20} ${pricing['input']:<7.2f} ${pricing['output']:<7.2f}\n"
            
            pricing_message += "```\n"
            pricing_message += (
                "**💡 Cost Examples:**\n"
                "• A typical conversation (~1,000 tokens) with `gpt-4o-mini`: ~$0.002\n"
                "• A typical conversation (~1,000 tokens) with `gpt-4o`: ~$0.025\n"
                "• A typical conversation (~1,000 tokens) with `o1-preview`: ~$0.075\n\n"
                "Use `/user_stat` to see your total usage and costs!"
            )

            # Send the response
            await interaction.followup.send(pricing_message, ephemeral=True)
        
        await process_request(interaction, process_prices)

    @tree.command(name="help", description="Display a list of available commands.")
    @check_blacklist()
    async def help_command(interaction: discord.Interaction):
        """Sends a list of available commands to the user."""
        help_message = (
            "**🤖 Available Commands:**\n\n"
            "**Model Selection:**\n"
            "• `/choose_model` - Select AI model from a dropdown menu\n"
            "• `/set_model <model>` - Set model directly with autocomplete\n\n"
            "**Search & Web:**\n"
            "• `/search <query>` - Search Google and analyze results with AI\n"
            "• `/web <url>` - Scrape and analyze a webpage\n\n"
            "**Image Generation:**\n"
            "• `/generate <prompt>` - Generate images from text\n\n"
            "**Settings & Stats:**\n"
            "• `/toggle_tools` - Toggle tool execution details display\n"
            "• `/user_stat` - View your token usage and costs\n"
            "• `/prices` - Display model pricing information\n"
            "• `/reset` - Clear your chat history and statistics\n\n"
            "**Help:**\n"
            "• `/help` - Display this help message\n"
        )
        await interaction.response.send_message(help_message, ephemeral=True)

    @tree.command(name="toggle_tools", description="Toggle the display of tool execution details (code, input, output).")
    @check_blacklist()
    async def toggle_tools(interaction: discord.Interaction):
        """Toggle the display of tool execution details for the user."""
        await interaction.response.defer(ephemeral=True)
        
        user_id = interaction.user.id
        current_setting = await db_handler.get_user_tool_display(user_id)
        new_setting = not current_setting
        
        await db_handler.set_user_tool_display(user_id, new_setting)
        
        status = "enabled" if new_setting else "disabled"
        description = (
            "You will now see detailed execution information including code, input, and output when tools are used."
            if new_setting else
            "Tool execution details are now hidden. You'll only see the final results."
        )
        
        await interaction.followup.send(
            f"🔧 **Tool Display {status.title()}**\n{description}",
            ephemeral=True
        )

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