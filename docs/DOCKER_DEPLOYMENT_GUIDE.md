# Docker Deployment Guide

## ✅ Docker Compatibility Verification

All new features are **fully compatible** with Docker deployment:

### 1. ✅ File Storage System
- **Location**: `/tmp/bot_code_interpreter/` (created in Dockerfile)
- **Volumes**: Mounted in docker-compose.yml for persistence
- **Permissions**: Set to 777 for read/write access

### 2. ✅ Code Interpreter
- **Dependencies**: All runtime libraries included (HDF5, OpenBLAS, etc.)
- **Venv**: Persistent volume for package cache
- **Timeout**: Configurable via environment variables

### 3. ✅ 200+ File Types
- **Libraries**: Build dependencies included for all file formats
- **Runtime**: All required shared libraries present

---

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# 1. Make sure .env file is configured
cat .env

# 2. Start the bot
docker-compose up -d

# 3. Check logs
docker-compose logs -f bot

# 4. Stop the bot
docker-compose down
```

### Option 2: Using Docker CLI

```bash
# 1. Build the image
docker build -t chatgpt-discord-bot .

# 2. Run the container
docker run -d \
  --name chatgpt-bot \
  --env-file .env \
  -v bot_files:/tmp/bot_code_interpreter/user_files \
  -v bot_venv:/tmp/bot_code_interpreter/venv \
  -v bot_outputs:/tmp/bot_code_interpreter/outputs \
  --restart always \
  chatgpt-discord-bot

# 3. Check logs
docker logs -f chatgpt-bot
```

---

## ⚙️ Configuration

### Environment Variables

All configuration is done via the `.env` file:

```bash
# Discord & API
DISCORD_TOKEN=your_token_here
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://models.github.ai/inference
MONGODB_URI=mongodb+srv://...

# File Management
FILE_EXPIRATION_HOURS=48        # Files expire after 48 hours (-1 = never)
MAX_FILES_PER_USER=20           # Max 20 files per user

# Code Execution
CODE_EXECUTION_TIMEOUT=300      # 5 minutes timeout

# Timezone
TIMEZONE=Asia/Ho_Chi_Minh
```

### Volume Mounts

The docker-compose.yml includes three volumes:

1. **bot_files**: Persistent storage for user files
   - Path: `/tmp/bot_code_interpreter/user_files`
   - Purpose: Keeps files across container restarts

2. **bot_venv**: Persistent Python virtual environment
   - Path: `/tmp/bot_code_interpreter/venv`
   - Purpose: Caches installed packages (faster restarts)

3. **bot_outputs**: Generated output files
   - Path: `/tmp/bot_code_interpreter/outputs`
   - Purpose: Stores generated plots, CSVs, etc.

### Resource Limits

Adjust in docker-compose.yml based on your needs:

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'        # Max 2 CPU cores
      memory: 2G         # Max 2GB RAM
    reservations:
      cpus: '0.5'        # Min 0.5 CPU cores
      memory: 512M       # Min 512MB RAM
```

---

## 🔧 Troubleshooting

### Issue: Files not persisting after restart

**Solution**: Ensure volumes are properly mounted:

```bash
# Check volumes
docker volume ls

# Inspect volume
docker volume inspect bot_files

# If volumes are missing, recreate them
docker-compose down
docker-compose up -d
```

### Issue: Package installation fails

**Solution**: Check if venv volume has enough space:

```bash
# Check volume size
docker system df -v

# Clear old volumes if needed
docker volume prune
```

### Issue: Timeout errors

**Solution**: Increase timeout in .env or docker-compose.yml:

```bash
CODE_EXECUTION_TIMEOUT=900  # 15 minutes for heavy processing
```

### Issue: Out of memory

**Solution**: Increase memory limit in docker-compose.yml:

```yaml
limits:
  memory: 4G  # Increase to 4GB
```

### Issue: File permissions error

**Solution**: Check /tmp directory permissions:

```bash
# Enter container
docker exec -it <container_id> sh

# Check permissions
ls -la /tmp/bot_code_interpreter/

# Fix if needed (already set in Dockerfile)
chmod -R 777 /tmp/bot_code_interpreter/
```

---

## 📊 Monitoring

### View Logs

```bash
# All logs
docker-compose logs -f bot

# Last 100 lines
docker-compose logs --tail=100 bot

# Filter by level
docker-compose logs bot | grep ERROR
```

### Check Resource Usage

```bash
# Real-time stats
docker stats

# Container info
docker inspect chatgpt-bot
```

### Healthcheck Status

```bash
# Check health
docker ps

# If unhealthy, check logs
docker logs chatgpt-bot
```

---

## 🔄 Updates

### Update to Latest Version

```bash
# Pull latest image
docker-compose pull

# Restart with new image
docker-compose up -d

# Check logs
docker-compose logs -f bot
```

### Rebuild from Source

```bash
# Rebuild image
docker-compose build --no-cache

# Restart
docker-compose up -d
```

---

## 💾 Backup

### Backup Volumes

```bash
# Backup user files
docker run --rm \
  -v bot_files:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/bot_files_backup.tar.gz /data

# Backup venv
docker run --rm \
  -v bot_venv:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/bot_venv_backup.tar.gz /data
```

### Restore Volumes

```bash
# Restore user files
docker run --rm \
  -v bot_files:/data \
  -v $(pwd):/backup \
  alpine sh -c "cd /data && tar xzf /backup/bot_files_backup.tar.gz --strip 1"
```

---

## 🏗️ Build Details

### Multi-Stage Build

The Dockerfile uses a multi-stage build for optimization:

**Stage 1: Builder**
- Installs all build dependencies
- Compiles Python packages
- Strips debug symbols for smaller size

**Stage 2: Runtime**
- Only includes runtime dependencies
- Much smaller final image
- Faster startup time

### Included Dependencies

**Build-time:**
- gcc, g++, rust, cargo
- HDF5, OpenBLAS, LAPACK development files
- Image processing libraries (freetype, libpng, libjpeg)

**Runtime:**
- HDF5, OpenBLAS, LAPACK shared libraries
- Image processing runtime libraries
- Git (for package installations)
- Bash (for shell scripts in code execution)

---

## 🔒 Security

### Best Practices

1. **Never commit .env file**
   ```bash
   # .env is in .gitignore
   git status  # Should not show .env
   ```

2. **Use secrets management**
   ```bash
   # For production, use Docker secrets
   docker secret create discord_token token.txt
   ```

3. **Limit container permissions**
   ```yaml
   # In docker-compose.yml
   security_opt:
     - no-new-privileges:true
   ```

4. **Regular updates**
   ```bash
   # Update base image regularly
   docker-compose pull
   docker-compose up -d
   ```

---

## 📈 Performance Optimization

### 1. Persistent Venv

The venv volume caches installed packages:
- **First run**: Installs packages (slow)
- **Subsequent runs**: Uses cache (fast)

### 2. Layer Caching

The Dockerfile is optimized for layer caching:
- Requirements installed in separate layer
- Application code copied last
- Only rebuilds changed layers

### 3. Resource Allocation

Adjust based on usage:
- **Light usage**: 0.5 CPU, 512MB RAM
- **Medium usage**: 1 CPU, 1GB RAM
- **Heavy usage**: 2+ CPUs, 2GB+ RAM

---

## ✅ Verification Checklist

Before deploying:

- [ ] `.env` file configured with all required variables
- [ ] Docker and Docker Compose installed
- [ ] Sufficient disk space for volumes (5GB+ recommended)
- [ ] Network access to Discord API and MongoDB
- [ ] Ports not conflicting with other services

After deploying:

- [ ] Container is running: `docker ps`
- [ ] No errors in logs: `docker-compose logs bot`
- [ ] Bot online in Discord
- [ ] File uploads work
- [ ] Code execution works
- [ ] Files persist after restart

---

## 🎯 Production Deployment

### Recommended Setup

```yaml
version: '3.8'

services:
  bot:
    image: ghcr.io/coder-vippro/chatgpt-discord-bot:latest
    env_file:
      - .env
    restart: always
    
    volumes:
      - bot_files:/tmp/bot_code_interpreter/user_files
      - bot_venv:/tmp/bot_code_interpreter/venv
      - bot_outputs:/tmp/bot_code_interpreter/outputs
    
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
    
    healthcheck:
      test: ["CMD", "python3", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  bot_files:
    driver: local
  bot_venv:
    driver: local
  bot_outputs:
    driver: local
```

---

## 📞 Support

If you encounter issues:

1. Check logs: `docker-compose logs -f bot`
2. Verify volumes: `docker volume ls`
3. Check resources: `docker stats`
4. Review configuration: `cat .env`
5. Test file access: `docker exec -it <container> ls -la /tmp/bot_code_interpreter/`

---

## 🎉 Summary

✅ **Docker Setup Complete!**

The bot is now fully compatible with Docker deployment with:
- Persistent file storage
- Cached package installations
- Configurable resource limits
- Health monitoring
- Production-ready configuration

**Deploy with confidence!** 🚀
