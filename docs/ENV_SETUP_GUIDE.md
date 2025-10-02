# Environment Variables Setup Guide

## 📋 Quick Setup

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and fill in your actual values

3. Restart the bot

## 🔑 Required Variables

These **must** be configured for the bot to work:

### 1. DISCORD_TOKEN
- **What**: Your Discord bot token
- **Where**: https://discord.com/developers/applications
- **Steps**:
  1. Go to Discord Developer Portal
  2. Select your application
  3. Go to "Bot" section
  4. Click "Reset Token" and copy it
- **Example**: `DISCORD_TOKEN=MT3u19203u0dua0d9s`

### 2. OPENAI_API_KEY
- **What**: API key for AI models
- **Where**: 
  - GitHub Models (free): https://github.com/settings/tokens
  - OpenAI (paid): https://platform.openai.com/api-keys
- **Steps**:
  - For GitHub Models: Create a Personal Access Token with model access
  - For OpenAI: Create an API key
- **Example**: `OPENAI_API_KEY=ghp_xxxxxxxxxxxxxxxxxxxx` (GitHub) or `sk-xxxxxxxxxxxx` (OpenAI)

### 3. OPENAI_BASE_URL
- **What**: API endpoint for AI models
- **Options**:
  - `https://models.github.ai/inference` - GitHub Models (free)
  - `https://api.openai.com/v1` - OpenAI (paid)
- **Example**: `OPENAI_BASE_URL=https://models.github.ai/inference`

### 4. MONGODB_URI
- **What**: Database connection string
- **Where**: https://cloud.mongodb.com/
- **Steps**:
  1. Create a free MongoDB Atlas cluster
  2. Click "Connect" → "Connect your application"
  3. Copy the connection string
  4. Replace `<password>` with your database password
- **Example**: `MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority`

### 5. ADMIN_ID
- **What**: Your Discord user ID
- **Steps**:
  1. Enable Discord Developer Mode (User Settings → Advanced → Developer Mode)
  2. Right-click your username
  3. Click "Copy ID"
- **Example**: `ADMIN_ID=1231312312313`

## 🎨 Optional Variables

These enhance functionality but aren't required:

### RUNWARE_API_KEY (Image Generation)
- **What**: API key for generating images
- **Where**: https://runware.ai
- **Feature**: Enables `/generate` command
- **Leave empty**: Image generation will be disabled

### GOOGLE_API_KEY + GOOGLE_CX (Web Search)
- **What**: Google Custom Search credentials
- **Where**: 
  - API Key: https://console.cloud.google.com/apis/credentials
  - CX: https://programmablesearchengine.google.com/
- **Feature**: Enables `/search` command
- **Leave empty**: Search will be disabled

### LOGGING_WEBHOOK_URL (Logging)
- **What**: Discord webhook for bot logs
- **Where**: Discord channel settings → Integrations → Webhooks
- **Feature**: Sends bot logs to Discord channel
- **Leave empty**: Logs only to console/file

### ENABLE_WEBHOOK_LOGGING
- **What**: Enable/disable webhook logging
- **Options**: `true` or `false`
- **Default**: `true`

### TIMEZONE
- **What**: Timezone for timestamps
- **Options**: Any IANA timezone (e.g., `America/New_York`, `Europe/London`, `Asia/Tokyo`)
- **Default**: `UTC`
- **List**: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

### FILE_EXPIRATION_HOURS
- **What**: How long files are kept before auto-deletion
- **Options**:
  - `24` - 1 day
  - `48` - 2 days (default)
  - `72` - 3 days
  - `168` - 1 week
  - `-1` - Never expire (permanent)
- **Default**: `48`

## 📝 Example Configurations

### Minimal Setup (Free)
```bash
# Required only
DISCORD_TOKEN=your_token
OPENAI_API_KEY=ghp_your_github_token
OPENAI_BASE_URL=https://models.github.ai/inference
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
ADMIN_ID=your_discord_id

# Optional - use defaults
FILE_EXPIRATION_HOURS=48
ENABLE_WEBHOOK_LOGGING=false
TIMEZONE=UTC
```

### Full Setup (All Features)
```bash
# Required
DISCORD_TOKEN=your_token
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://models.github.ai/inference
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
ADMIN_ID=your_discord_id

# Optional - all features enabled
RUNWARE_API_KEY=your_runware_key
GOOGLE_API_KEY=your_google_key
GOOGLE_CX=your_cx_id
LOGGING_WEBHOOK_URL=your_webhook_url
ENABLE_WEBHOOK_LOGGING=true
TIMEZONE=Asia/Ho_Chi_Minh
FILE_EXPIRATION_HOURS=-1
```

## 🔒 Security Best Practices

1. **Never commit `.env` to Git**
   - `.env` is in `.gitignore` by default
   - Only commit `.env.example`

2. **Keep tokens secure**
   - Don't share your `.env` file
   - Don't post tokens in public channels
   - Regenerate tokens if exposed

3. **Use environment-specific files**
   - `.env.development` for dev
   - `.env.production` for prod
   - Never mix them up

4. **Restrict MongoDB access**
   - Use strong passwords
   - Whitelist only necessary IPs
   - Enable authentication

## 🐛 Troubleshooting

### Bot won't start
- ✅ Check all required variables are set
- ✅ Verify MongoDB connection string
- ✅ Test with `mongosh "your-mongodb-uri"`
- ✅ Check Discord token is valid

### Commands don't work
- ✅ Bot needs proper Discord permissions
- ✅ Commands must be synced (automatic on startup)
- ✅ Wait 5-10 minutes after bot restart for sync

### Image generation fails
- ✅ Verify `RUNWARE_API_KEY` is set
- ✅ Check Runware account has credits
- ✅ See error logs for details

### Search doesn't work
- ✅ Both `GOOGLE_API_KEY` and `GOOGLE_CX` must be set
- ✅ Enable Custom Search API in Google Cloud Console
- ✅ Verify API quota not exceeded

### Files not expiring
- ✅ Check `FILE_EXPIRATION_HOURS` value
- ✅ `-1` means never expire (by design)
- ✅ Cleanup task runs every 6 hours

## 📚 Related Documentation

- **File Management**: `docs/FILE_MANAGEMENT_GUIDE.md`
- **Quick Reference**: `docs/QUICK_REFERENCE_FILE_MANAGEMENT.md`
- **Commands**: Use `/help` in Discord

---

**Need help?** Check the logs or create an issue on GitHub!
