import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from bot import bot, search, generate_image, web, choose_model, reset, help_command, trim_history


class TestDiscordBotCommands(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = bot
        self.interaction = AsyncMock()
        self.interaction.user.id = 123456789  # Mock user ID

    async def test_search_command(self):
        self.interaction.response.defer = AsyncMock()
        self.interaction.followup.send = AsyncMock()

        await search(self.interaction, query="Python")
        self.interaction.followup.send.assert_called()
        self.interaction.response.defer.assert_called_with(thinking=True)

    async def test_generate_image_command(self):
        self.interaction.response.defer = AsyncMock()
        self.interaction.followup.send = AsyncMock()

        with patch('bot.runware.imageInference', return_value=[MagicMock(imageURL="http://example.com/image.png")]):
            await generate_image(self.interaction, prompt="Sunset over mountains")
            self.interaction.response.defer.assert_called_with(thinking=True)
            self.interaction.followup.send.assert_called()

    async def test_web_scraping_command(self):
        self.interaction.response.defer = AsyncMock()
        self.interaction.followup.send = AsyncMock()

        await web(self.interaction, url="https://example.com")
        self.interaction.followup.send.assert_called()
        self.interaction.response.defer.assert_called_with(thinking=True)

    async def test_choose_model_command(self):
        self.interaction.response.send_message = AsyncMock()

        with patch('bot.save_user_model') as mock_save_model:
            await choose_model(self.interaction)
            mock_save_model.assert_called()
            self.interaction.response.send_message.assert_called_with(
                "Choose a model:", view=MagicMock(), ephemeral=True
            )

    async def test_reset_command(self):
        self.interaction.response.send_message = AsyncMock()

        with patch('bot.db.user_histories.delete_one') as mock_delete:
            await reset(self.interaction)
            mock_delete.assert_called_with({'user_id': self.interaction.user.id})
            self.interaction.response.send_message.assert_called_with(
                "Your data has been cleared and reset!", ephemeral=True
            )

    async def test_help_command(self):
        self.interaction.response.send_message = AsyncMock()

        await help_command(self.interaction)
        self.interaction.response.send_message.assert_called_with(unittest.mock.ANY, ephemeral=True)

    async def test_message_processing(self):
        message = MagicMock()
        message.author.id = 987654321
        message.content = "Hello, bot!"
        message.guild = None  # Simulate a DM
        message.channel.send = AsyncMock()

        await bot.on_message(message)
        message.channel.send.assert_called()

    def test_trim_history(self):
        history = [
            {"role": "user", "content": "This is a test " * 500},
            {"role": "assistant", "content": "This is a response " * 500},
        ]
        trim_history(history)
        tokens_used = sum(len(item['content']) for item in history)
        self.assertTrue(tokens_used <= 9000)

    async def test_rate_limit_handling(self):
        self.interaction.response.defer = AsyncMock()
        self.interaction.followup.send = AsyncMock()

        with patch('bot.process_request') as mock_process_request:
            await bot.process_request(self.interaction, lambda x: x, "args")
            self.interaction.followup.send.assert_called_with(
                "You are sending requests too quickly. Please wait a moment.", ephemeral=True
            )

if __name__ == '__main__':
    unittest.main()
