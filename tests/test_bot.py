import unittest
import unittest.mock as mock
from unittest.mock import MagicMock, patch, AsyncMock
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
    bot,
    process_request,
    process_queue,
    choose_model,
    search,
    web,
    reset,
    help_command,
    should_respond_to_message,
    handle_user_message,
    trim_history,
    generate_image,
    _generate_image_command,
    change_status,
    on_ready
)

class TestFullBot(unittest.TestCase):

    def setUp(self):
        # Sử dụng app của Flask để test endpoint
        self.app = app.test_client()

    def test_flask_health_endpoint(self):
        with patch("bot.bot.is_closed", return_value=False), \
             patch("bot.bot.is_ready", return_value=True):
            response = self.app.get('/health')
            self.assertEqual(response.status_code, 200, "Health endpoint should return 200 if bot is ready.")

    def test_run_flask(self):
        # Kiểm tra run_flask khởi động mà không báo lỗi
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

    async def async_test_get_history(self):
        with patch("bot.db.user_histories.find_one", new_callable=AsyncMock) as mock_find_one:
            mock_find_one.return_value = {"history": [{"role": "system", "content": NORMAL_CHAT_PROMPT}]}
            history = await get_history(1234)
            self.assertIsInstance(history, list, "History should be a list.")

    def test_get_history(self):
        asyncio.run(self.async_test_get_history())

    async def async_test_get_user_model_default(self):
        with patch("bot.db.user_preferences.find_one", new_callable=AsyncMock) as mock_find_one:
            mock_find_one.return_value = None
            model = await get_user_model(1234)
            self.assertEqual(model, "gpt-4o")


    def test_trim_history_with_large_content(self):
        sample_history = [
            {"role": "user", "content": "x" * 5000},
            {"role": "user", "content": "y" * 5000}
        ]
        trim_history(sample_history)
        # Giả sử hàm trim_history không xóa hết nội dung mà vẫn giữ lại tối thiểu 1 message
        self.assertGreaterEqual(len(sample_history), 1, "History should not be completely removed.")

    def test_trim_history_with_empty_history(self):
        history = []
        trim_history(history)
        self.assertEqual(len(history), 0, "Empty history should remain empty.")

    def test_trim_history_with_single_message(self):
        history = [{"role": "user", "content": "test"}]
        trim_history(history)
        self.assertEqual(len(history), 1, "Single message history should remain unchanged.")

    async def async_test_process_message_with_attachment(self):
        message = AsyncMock()
        message.author.id = 1234
        message.content = "Check this file"
        message.attachments = [
            AsyncMock(filename="test.txt", read=AsyncMock(return_value=b"File content"))
        ]
        # Patch bot.user để tránh lỗi khi gọi mentioned_in (bot.user có thể là None trong môi trường test)
        with patch.object(bot.user, 'mentioned_in', return_value=False):
            await bot.on_message(message)


    async def async_test_search(self):
        interaction = AsyncMock()
        interaction.user.id = 1234
        # Patch get_history để trả về list thay vì coroutine, tránh lỗi khi gọi append
        with patch("bot.get_history", new=AsyncMock(return_value=[])):
            await search.callback(interaction, query="Python")
            interaction.response.defer.assert_called()
            interaction.followup.send.assert_called()

    def test_search_command(self):
        asyncio.run(self.async_test_search())

    async def async_test_web(self):
        interaction = AsyncMock()
        interaction.user.id = 1234
        with patch("bot.get_history", new=AsyncMock(return_value=[])):
            await web.callback(interaction, url="https://test.com")
            interaction.response.defer.assert_called()
            interaction.followup.send.assert_called()

    def test_web_command(self):
        asyncio.run(self.async_test_web())

    async def async_test_reset(self):
        interaction = AsyncMock()
        interaction.user.id = 1234
        await reset.callback(interaction)
        interaction.response.send_message.assert_called()

    async def test_reset_command(self):
        await self.async_test_reset()

    async def async_test_help_command(self):
        interaction = AsyncMock()
        # Nếu help_command được đăng ký dưới dạng command (Command object),
        # bạn cần gọi .callback thay vì gọi trực tiếp đối tượng.
        await help_command.callback(interaction)
        interaction.response.send_message.assert_called()

    def test_help_command(self):
        asyncio.run(self.async_test_help_command())

if __name__ == "__main__":
    unittest.main()
