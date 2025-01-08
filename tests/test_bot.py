import unittest
import unittest.mock as mock
from unittest.mock import MagicMock, patch, AsyncMock
from unittest.mock import patch
import asyncio
import requests
from flask import Flask
from bot import (
    app,
    run_flask,
    client,
    statuses,
    MODEL_OPTIONS,
    WEB_SCRAPING_PROMPT,
    NORMAL_CHAT_PROMPT,
    SEARCH_PROMPT,
    google_custom_search,
    scrape_web_content,
    get_history,
    save_history,
    get_user_model,
    save_user_model,
    get_remaining_turns,
    update_remaining_turns,
    reset_remaining_turns,
    bot,
    process_request,
    process_queue,
    choose_model,
    search,
    web,
    reset,
    remaining_turns,
    help_command,
    should_respond_to_message,
    handle_user_message,
    trim_history,
    generate_image,
    _generate_image_command,
    change_status,
    daily_reset,
    on_ready
)

class TestFullBot(unittest.TestCase):
    def setUp(self):
        # You can set up mocks or initial states here.
        self.app = app.test_client()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def test_flask_health_endpoint(self):
        with patch("bot.bot.is_closed", return_value=False), \
             patch("bot.bot.is_ready", return_value=True):
            response = self.app.get('/health')
            self.assertEqual(response.status_code, 200, "Health endpoint should return 200 if bot is ready.")

    def test_run_flask(self):
        # We can just check if run_flask starts up without error; a real test would be more involved.
        with patch.object(Flask, 'run') as mock_run:
            run_flask()
            mock_run.assert_called_once()

    @patch("requests.get")
    def test_google_custom_search(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"items": [{"title": "Result 1"}]}
        mock_get.return_value = mock_resp
        results = google_custom_search("test")
        self.assertEqual(len(results), 1, "Should return 1 search result when JSON has items.")

    @patch("requests.get")
    def test_scrape_web_content(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<p>Some scraped text</p>"
        mock_get.return_value = mock_response
        result = scrape_web_content("https://example.com")
        self.assertIn("Some scraped text", result, "Scraped content should include known text.")

    def test_get_history(self):
        # Mock database behavior if needed
        with patch("bot.db.user_histories.find_one", return_value={"history": [{"role": "system", "content": NORMAL_CHAT_PROMPT}]}):
            history = get_history(1234)
            self.assertIsInstance(history, list, "History should be a list.")

    async def async_test_search(self):
        interaction = AsyncMock()
        interaction.user.id = 1234
        await search.callback(interaction, query="Python")
        interaction.response.defer.assert_called()
        interaction.followup.send.assert_called()

    def test_search_command(self):
        self.loop.run_until_complete(self.async_test_search())

    async def async_test_web(self):
        interaction = AsyncMock()
        interaction.user.id = 1234
        await web.callback(interaction, url="https://test.com")
        interaction.response.defer.assert_called()
        interaction.followup.send.assert_called()

    def test_web_command(self):
        self.loop.run_until_complete(self.async_test_web())

    async def async_test_reset(self):
        interaction = AsyncMock()
        interaction.user.id = 1234
        await reset.callback(interaction)
        interaction.response.send_message.assert_called()

    def test_reset_command(self):
        self.loop.run_until_complete(self.async_test_reset())

    async def async_test_help_command(self):
        interaction = AsyncMock()
        await help_command(interaction)
        interaction.response.send_message.assert_called()

    def test_trim_history(self):
        sample_history = [{"role": "user", "content": "x" * 5000}, {"role": "user", "content": "y" * 5000}]
        trim_history(sample_history)
        def test_process_message_with_attachment(self):
            # Test handling message with text file attachment 
            message = AsyncMock()
            message.author.id = 1234
            message.content = "Check this file"
            message.attachments = [
                AsyncMock(
                    filename="test.txt",
                    read=AsyncMock(return_value=b"File content")
                )
            ]
            
            async def run_test():
                with patch("bot.handle_user_message") as mock_handle:
                    await bot.on_message(message)
                    mock_handle.assert_called_once()
            
            self.loop.run_until_complete(run_test())

    def test_trim_history_with_empty_history(self):
            # Test trim_history with empty history
            history = []
            trim_history(history)
            self.assertEqual(len(history), 0)

    def test_trim_history_with_single_message(self):
            # Test trim_history with single message
            history = [{"role": "user", "content": "test"}]
            trim_history(history) 
            self.assertEqual(len(history), 1)

    def test_get_user_model_default(self):
            # Test get_user_model returns default model
            with patch("bot.db.user_preferences.find_one", return_value=None):
                model = get_user_model(1234)
                self.assertEqual(model, "gpt-4o")
