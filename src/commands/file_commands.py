"""
File Management Commands

Slash commands for managing user files.
Files are accessible by all tools (code_interpreter, analyze_data_file, etc.)
"""

import discord
from discord import app_commands
from discord.ext import commands
from typing import Optional
import logging
from datetime import datetime
import os
import io

logger = logging.getLogger(__name__)

class FileCommands(commands.Cog):
    """File management commands."""
    
    def __init__(self, bot):
        self.bot = bot
        self.db_handler = bot.db_handler
    
    @app_commands.command(name="files", description="📁 Manage your uploaded files")
    async def list_files(self, interaction: discord.Interaction):
        """List all files uploaded by the user with download/delete options."""
        await interaction.response.defer(ephemeral=True)
        
        try:
            from src.utils.code_interpreter import list_user_files
            
            user_id = interaction.user.id
            files = await list_user_files(user_id, self.db_handler)
            
            if not files:
                embed = discord.Embed(
                    title="📁 Your Files",
                    description="You don't have any files uploaded yet.\n\n"
                               "📤 **Upload files** by attaching them to your messages!\n"
                               "💡 The AI can automatically access and analyze them.",
                    color=discord.Color.blue()
                )
                
                # Check if files never expire
                expiration_hours = int(os.getenv('FILE_EXPIRATION_HOURS', '48'))
                if expiration_hours == -1:
                    embed.set_footer(text="Files never expire (permanent storage)")
                else:
                    embed.set_footer(text=f"Files expire after {expiration_hours} hours")
                
                await interaction.followup.send(embed=embed, ephemeral=True)
                return
            
            # Sort by upload date (newest first)
            files.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)
            
            # Create embed with file list
            embed = discord.Embed(
                title="📁 Your Files",
                description=f"You have **{len(files)}** file(s) uploaded.\n"
                           "Select a file below to download or delete it.",
                color=discord.Color.green()
            )
            
            # File type emojis
            type_emojis = {
                'csv': '📊', 'excel': '📊', 'json': '📋', 'text': '📝',
                'image': '🖼️', 'pdf': '📄', 'python': '💻', 'code': '💻',
                'data': '📊', 'database': '🗄️', 'archive': '📦', 
                'markdown': '📝', 'html': '🌐', 'xml': '📋',
                'yaml': '📋', 'sql': '🗄️', 'jupyter': '📓'
            }
            
            # Display files (max 10 in embed to avoid clutter)
            display_count = min(len(files), 10)
            for i, file in enumerate(files[:display_count], 1):
                file_id = file.get('file_id', 'unknown')
                filename = file.get('filename', 'Unknown')
                file_type = file.get('file_type', 'file')
                file_size = file.get('file_size', 0)
                uploaded_at = file.get('uploaded_at', '')
                expires_at = file.get('expires_at', '')
                
                # Format size
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"
                
                # Format dates
                try:
                    uploaded_dt = datetime.fromisoformat(uploaded_at)
                    uploaded_str = uploaded_dt.strftime("%Y-%m-%d %H:%M")
                    
                    # Check expiration
                    expiration_hours = int(os.getenv('FILE_EXPIRATION_HOURS', '48'))
                    if expiration_hours == -1:
                        expires_str = "♾️ Never"
                    else:
                        expires_dt = datetime.fromisoformat(expires_at)
                        time_left = expires_dt - datetime.now()
                        hours_left = int(time_left.total_seconds() / 3600)
                        
                        if hours_left < 0:
                            expires_str = "⚠️ Expired"
                        elif hours_left < 1:
                            mins_left = int(time_left.total_seconds() / 60)
                            expires_str = f"⏰ {mins_left}m left"
                        else:
                            expires_str = f"⏰ {hours_left}h left"
                except:
                    uploaded_str = "Unknown"
                    expires_str = "Unknown"
                
                # Get emoji
                emoji = type_emojis.get(file_type, '📎')
                
                # Truncate long filenames
                display_name = filename if len(filename) <= 40 else filename[:37] + "..."
                
                # Add field
                embed.add_field(
                    name=f"{emoji} {display_name}",
                    value=f"**Type:** {file_type} • **Size:** {size_str}\n"
                          f"**Uploaded:** {uploaded_str} • {expires_str}",
                    inline=False
                )
            
            if len(files) > 10:
                embed.add_field(
                    name="📌 Note",
                    value=f"Showing 10 of {len(files)} files. Files are listed from newest to oldest.",
                    inline=False
                )
            
            # Check expiration setting for footer
            expiration_hours = int(os.getenv('FILE_EXPIRATION_HOURS', '48'))
            if expiration_hours == -1:
                embed.set_footer(text="💡 Files are stored permanently • Use the menu below to manage files")
            else:
                embed.set_footer(text=f"💡 Files expire after {expiration_hours}h • Use the menu below to manage files")
            
            # Add interactive view with download/delete options
            view = FileManagementView(user_id, files, self.db_handler, self.bot)
            await interaction.followup.send(embed=embed, view=view, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            import traceback
            traceback.print_exc()
            await interaction.followup.send(
                "❌ An error occurred while listing your files.",
                ephemeral=True
            )


class FileManagementView(discord.ui.View):
    """Interactive view for file management with download/delete options."""
    
    def __init__(self, user_id: int, files: list, db_handler, bot):
        super().__init__(timeout=300)  # 5 minute timeout
        self.user_id = user_id
        self.files = files
        self.db_handler = db_handler
        self.bot = bot
        
        # Add file selection dropdown
        if files:
            self.add_item(FileSelectMenu(files))


class FileSelectMenu(discord.ui.Select):
    """Dropdown menu for selecting a file to download or delete."""
    
    def __init__(self, files: list):
        self.files_map = {}
        options = []
        
        type_emojis = {
            'csv': '📊', 'excel': '📊', 'json': '📋', 'text': '📝',
            'image': '🖼️', 'pdf': '📄', 'python': '💻', 'code': '💻',
            'data': '📊', 'database': '🗄️', 'archive': '📦'
        }
        
        # Limit to 25 options (Discord's limit)
        for i, file in enumerate(files[:25]):
            file_id = file.get('file_id', 'unknown')
            filename = file.get('filename', 'Unknown')
            file_type = file.get('file_type', 'file')
            file_size = file.get('file_size', 0)
            
            # Store file data for later
            self.files_map[file_id] = file
            
            # Format size
            if file_size < 1024:
                size_str = f"{file_size}B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f}KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f}MB"
            
            emoji = type_emojis.get(file_type, '📎')
            
            # Truncate filename if too long (Discord limit: 100 chars for label)
            display_name = filename if len(filename) <= 80 else filename[:77] + "..."
            
            options.append(
                discord.SelectOption(
                    label=display_name,
                    description=f"{file_type} • {size_str}",
                    value=file_id,
                    emoji=emoji
                )
            )
        
        super().__init__(
            placeholder="📂 Select a file to download or delete...",
            options=options,
            min_values=1,
            max_values=1
        )
    
    async def callback(self, interaction: discord.Interaction):
        """Handle file selection - show download/delete buttons."""
        file_id = self.values[0]
        file_data = self.files_map.get(file_id)
        
        if not file_data:
            await interaction.response.send_message("❌ File not found.", ephemeral=True)
            return
        
        filename = file_data.get('filename', 'Unknown')
        file_type = file_data.get('file_type', 'file')
        file_size = file_data.get('file_size', 0)
        
        # Format size
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.2f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.2f} MB"
        
        # Create action view
        action_view = FileActionView(
            user_id=interaction.user.id,
            file_id=file_id,
            file_data=file_data,
            db_handler=self.view.db_handler
        )
        
        embed = discord.Embed(
            title=f"📄 {filename}",
            description=f"**Type:** {file_type}\n**Size:** {size_str}",
            color=discord.Color.blue()
        )
        embed.set_footer(text="Choose an action below")
        
        await interaction.response.send_message(embed=embed, view=action_view, ephemeral=True)


class FileActionView(discord.ui.View):
    """View with download and delete buttons for a specific file."""
    
    def __init__(self, user_id: int, file_id: str, file_data: dict, db_handler):
        super().__init__(timeout=60)
        self.user_id = user_id
        self.file_id = file_id
        self.file_data = file_data
        self.db_handler = db_handler
    
    @discord.ui.button(label="⬇️ Download", style=discord.ButtonStyle.primary)
    async def download_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Download the file."""
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("❌ This isn't your file!", ephemeral=True)
            return
        
        await interaction.response.defer(ephemeral=True)
        
        try:
            file_path = self.file_data.get('file_path')
            filename = self.file_data.get('filename', 'file')
            
            # Check if file exists
            if not os.path.exists(file_path):
                await interaction.followup.send("❌ File not found on disk. It may have been deleted.", ephemeral=True)
                return
            
            # Read file
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Check size (Discord limit: 25MB for non-nitro, 500MB for nitro)
            if len(file_bytes) > 25 * 1024 * 1024:
                await interaction.followup.send(
                    "❌ File is too large to download via Discord (>25MB).\n"
                    "The file is still available for use in code execution.",
                    ephemeral=True
                )
                return
            
            # Send file
            discord_file = discord.File(io.BytesIO(file_bytes), filename=filename)
            await interaction.followup.send(
                f"✅ **Downloaded:** `{filename}`",
                file=discord_file,
                ephemeral=True
            )
            
            logger.info(f"User {self.user_id} downloaded file {self.file_id}")
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            await interaction.followup.send("❌ An error occurred while downloading the file.", ephemeral=True)
    
    @discord.ui.button(label="🗑️ Delete", style=discord.ButtonStyle.danger)
    async def delete_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Delete the file (with confirmation)."""
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("❌ This isn't your file!", ephemeral=True)
            return
        
        # Show confirmation dialog
        confirm_view = ConfirmDeleteView(
            user_id=self.user_id,
            file_id=self.file_id,
            filename=self.file_data.get('filename', 'file'),
            db_handler=self.db_handler
        )
        
        embed = discord.Embed(
            title="⚠️ Confirm Deletion",
            description=f"Are you sure you want to delete:\n**{self.file_data.get('filename')}**?\n\n"
                       "This action cannot be undone!",
            color=discord.Color.orange()
        )
        
        await interaction.response.send_message(embed=embed, view=confirm_view, ephemeral=True)


class ConfirmDeleteView(discord.ui.View):
    """Confirmation view for deleting a file (requires 2 confirmations)."""
    
    def __init__(self, user_id: int, file_id: str, filename: str, db_handler):
        super().__init__(timeout=30)
        self.user_id = user_id
        self.file_id = file_id
        self.filename = filename
        self.db_handler = db_handler
        self.first_confirmation = False
    
    @discord.ui.button(label="⚠️ Yes, Delete", style=discord.ButtonStyle.danger)
    async def confirm_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handle delete confirmation."""
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("❌ This isn't your confirmation!", ephemeral=True)
            return
        
        # First confirmation
        if not self.first_confirmation:
            self.first_confirmation = True
            
            # Update button text and require second click
            button.label = "🔴 Click Again to Confirm"
            button.style = discord.ButtonStyle.danger
            
            embed = discord.Embed(
                title="⚠️ Final Confirmation",
                description=f"Click **'🔴 Click Again to Confirm'** to permanently delete:\n"
                           f"**{self.filename}**\n\n"
                           f"This is your last chance to cancel!",
                color=discord.Color.red()
            )
            
            await interaction.response.edit_message(embed=embed, view=self)
            return
        
        # Second confirmation - actually delete
        await interaction.response.defer(ephemeral=True)
        
        try:
            from src.utils.code_interpreter import delete_file
            
            result = await delete_file(self.file_id, self.user_id, self.db_handler)
            
            if result['success']:
                embed = discord.Embed(
                    title="✅ File Deleted",
                    description=f"Successfully deleted: **{self.filename}**",
                    color=discord.Color.green()
                )
                await interaction.followup.send(embed=embed, ephemeral=True)
                
                logger.info(f"User {self.user_id} deleted file {self.file_id}")
            else:
                embed = discord.Embed(
                    title="❌ Delete Failed",
                    description=result.get('error', 'Could not delete file'),
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=embed, ephemeral=True)
            
            # Disable all buttons (try to edit, but ignore if message is gone)
            try:
                for item in self.children:
                    item.disabled = True
                await interaction.message.edit(view=self)
            except discord.errors.NotFound:
                # Message was already deleted or is ephemeral and expired
                pass
            except Exception as edit_error:
                logger.debug(f"Could not edit message after deletion: {edit_error}")
            
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            await interaction.followup.send("❌ An error occurred while deleting the file.", ephemeral=True)
    
    @discord.ui.button(label="❌ Cancel", style=discord.ButtonStyle.secondary)
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Cancel deletion."""
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("❌ This isn't your confirmation!", ephemeral=True)
            return
        
        embed = discord.Embed(
            title="✅ Cancelled",
            description=f"File **{self.filename}** was not deleted.",
            color=discord.Color.blue()
        )
        
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
        # Disable all buttons (try to edit, but ignore if message is gone)
        try:
            for item in self.children:
                item.disabled = True
            await interaction.message.edit(view=self)
        except discord.errors.NotFound:
            # Message was already deleted or is ephemeral and expired
            pass
        except Exception as edit_error:
            logger.debug(f"Could not edit message after cancellation: {edit_error}")


async def setup(bot):
    """Load the cog."""
    await bot.add_cog(FileCommands(bot))
