import unittest
from unittest.mock import AsyncMock, MagicMock
from bot import bot, search, generate_image, web, translate

class TestDiscordBotCommands(unittest.TestCase):
    def setUp(self):
        self.bot = bot
        self.interaction = AsyncMock()
        self.interaction.user.id = 123456789  # Mock user ID

    async def test_search_command(self):
        # Set up mocks for interaction methods
        self.interaction.response.defer = AsyncMock()
        self.interaction.followup.send = AsyncMock()

        # Call the search command with a sample query
        await search(self.interaction, query="Python")

        # Check if followup.send was called
        self.interaction.followup.send.assert_called()
        self.interaction.response.defer.assert_called_with(thinking=True)

    async def test_generate_image_command(self):
        # Mock the deferred response
        self.interaction.response.defer = AsyncMock()
        self.interaction.followup.send = AsyncMock()

        # Patch Runware API to return a mock image URL
        with unittest.mock.patch('bot.runware.imageInference', return_value=[MagicMock(imageURL="http://example.com/image.png")]):
            await generate_image(self.interaction, prompt="Sunset over mountains")

            # Check if defer and followup were called
            self.interaction.response.defer.assert_called_with(thinking=True)
            self.interaction.followup.send.assert_called()

    async def test_web_scraping_command(self):
        # Mock the interaction methods
        self.interaction.response.defer = AsyncMock()
        self.interaction.followup.send = AsyncMock()

        # Call the web command with a mock URL
        await web(self.interaction, url="https://vnexpress.net/nguon-con-khien-arm-huy-giay-phep-chip-voi-qualcomm-4807985.html")

        # Ensure a followup message was sent
        self.interaction.followup.send.assert_called()
        self.interaction.response.defer.assert_called_with(thinking=True)

    async def test_message_processing(self):
        # Mock a direct message
        message = MagicMock()
        message.author.id = 987654321
        message.content = "Hello, bot!"
        message.guild = None  # Simulate a DM

        # Mock channel.send to test if the bot sends a message
        message.channel.send = AsyncMock()

        # Test the bot's response
        await bot.on_message(message)
        message.channel.send.assert_called()  # Check if the bot replied

    async def test_search_command_empty_query(self):
        # Set up mocks for interaction methods
        self.interaction.response.defer = AsyncMock()
        self.interaction.followup.send = AsyncMock()

        # Call the search command with an empty query
        await search(self.interaction, query="")

        # Check if followup.send was called with an error message
        self.interaction.followup.send.assert_called_with("Query cannot be empty.")
        self.interaction.response.defer.assert_called_with(thinking=True)

    async def test_generate_image_command_invalid_prompt(self):
        # Mock the deferred response
        self.interaction.response.defer = AsyncMock()
        self.interaction.followup.send = AsyncMock()

        # Patch Runware API to return an error
        with unittest.mock.patch('bot.runware.imageInference', side_effect=Exception("Invalid prompt")):
            await generate_image(self.interaction, prompt="")

            # Check if defer and followup were called with an error message
            self.interaction.response.defer.assert_called_with(thinking=True)
            self.interaction.followup.send.assert_called_with("Failed to generate image. Invalid prompt.")

    async def test_web_scraping_command_invalid_url(self):
        # Mock the interaction methods
        self.interaction.response.defer = AsyncMock()
        self.interaction.followup.send = AsyncMock()

        # Call the web command with an invalid URL
        await web(self.interaction, url="invalid_url")

        # Ensure a followup message was sent with an error message
        self.interaction.followup.send.assert_called_with("Invalid URL provided.")
        self.interaction.response.defer.assert_called_with(thinking=True)

    async def test_message_processing_command(self):
        # Mock a guild message
        message = MagicMock()
        message.author.id = 987654321
        message.content = "!command"
        message.guild = MagicMock()  # Simulate a guild message

        # Mock channel.send to test if the bot sends a message
        message.channel.send = AsyncMock()

        # Test the bot's response to the command
        await bot.on_message(message)
        message.channel.send.assert_called_with("Command received.")  # Check if the bot replied to the command

    async def test_translate_command(self):
        # Mock the deferred response
        self.interaction.response.defer = AsyncMock()
        self.interaction.followup.send = AsyncMock()

        # Call the translate command with a sample text
        await translate(self.interaction, text="Hello", target_language="es")

        # Check if defer and followup were called
        self.interaction.response.defer.assert_called_with(thinking=True)
        self.interaction.followup.send.assert_called_with("Translation: Hola")

if __name__ == '__main__':
    unittest.main()
