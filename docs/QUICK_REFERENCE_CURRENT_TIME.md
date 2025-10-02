# Quick Reference: Current Time in Context

## ⚡ Quick Setup

Add to your `.env` file:
```bash
TIMEZONE=Asia/Ho_Chi_Minh
```

Restart the bot:
```bash
python3 bot.py
# or
docker-compose restart
```

## 🎯 What It Does

The AI model now sees the current date and time **on every message**:

```
Current date and time: Thursday, October 02, 2025 at 09:30:45 PM ICT

[System prompt continues...]
```

## 📝 Format

- **Pattern**: `DayName, Month DD, YYYY at HH:MM:SS AM/PM TZ`
- **Example**: `Thursday, October 02, 2025 at 09:30:45 PM ICT`

## 🌍 Common Timezones

```bash
# Asia
TIMEZONE=Asia/Ho_Chi_Minh    # Vietnam
TIMEZONE=Asia/Tokyo          # Japan
TIMEZONE=Asia/Singapore      # Singapore
TIMEZONE=Asia/Shanghai       # China

# Americas
TIMEZONE=America/New_York    # US East
TIMEZONE=America/Los_Angeles # US West
TIMEZONE=America/Chicago     # US Central
TIMEZONE=America/Toronto     # Canada

# Europe
TIMEZONE=Europe/London       # UK
TIMEZONE=Europe/Paris        # France
TIMEZONE=Europe/Berlin       # Germany

# Others
TIMEZONE=Australia/Sydney    # Australia
TIMEZONE=UTC                 # Universal Time
```

## ✅ Features

- ✅ Updates **dynamically** on every message
- ✅ Works with **all models** (GPT-4, GPT-5, o1, etc.)
- ✅ Respects **daylight saving time**
- ✅ **Low overhead** (~15 tokens)
- ✅ **Docker compatible**

## 🧪 Test It

Ask the bot:
```
What time is it now?
How many hours until midnight?
Is it morning or evening?
```

## 🐛 Troubleshooting

### Wrong time showing?
```bash
# Check .env
grep TIMEZONE .env

# Restart bot
python3 bot.py
```

### Timezone error in Docker?
```bash
# Rebuild with tzdata
docker-compose build --no-cache
docker-compose up -d
```

## 📊 Impact

- **Token cost**: +15-20 tokens per message (~3% increase)
- **Latency**: <1ms (negligible)
- **Memory**: No additional usage

## 💡 Use Cases

- ⏰ Time-aware responses
- 📅 Scheduling and reminders
- 🗓️ Historical context
- 🌅 Time-based greetings
- 🕰️ Relative time calculations

## 🔗 Related

- Full documentation: [CURRENT_TIME_IN_CONTEXT.md](CURRENT_TIME_IN_CONTEXT.md)
- Timezone list: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
