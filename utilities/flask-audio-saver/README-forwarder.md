# N8N Forwarder

Monitors `/exports/final` for completed transcriptions and forwards them to n8n via webhook.

## Features

- **File Watching**: Monitors for new `.txt` transcript files
- **Automatic Processing**: Sends transcript, audio, and metadata files to n8n
- **Deduplication**: Tracks processed files to avoid duplicates
- **File Stability**: Waits for files to be fully written before sending
- **Size Limits**: Configurable max file size for uploads
- **Persistent Markers**: Creates `.n8n_sent` marker files to track sent transcripts

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WATCH_DIR` | `/exports/final` | Directory to monitor for transcripts |
| `N8N_WEBHOOK_URL` | `http://n8n.ml.svc.cluster.local:5678/webhook/audio-transcription` | n8n webhook endpoint |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `STABILITY_DELAY` | `2` | Seconds to wait for file stability |
| `MAX_FILE_SIZE` | `104857600` | Max file size in bytes (100MB default) |

## Usage

### Docker Compose (Recommended)

```bash
# Start both services
docker-compose up -d

# View logs
docker-compose logs -f n8n-forwarder

# Stop services
docker-compose down
```

### Docker Run

```bash
# Build the image
docker build -f Dockerfile.forwarder -t n8n-forwarder .

# Run the container
docker run -d \
  --name n8n-forwarder \
  -v ./exports:/exports \
  -e N8N_WEBHOOK_URL=http://your-n8n-url/webhook/audio-transcription \
  n8n-forwarder
```

## N8N Webhook Setup

Create a webhook workflow in n8n with these settings:

1. **Webhook Node**:
   - Method: POST
   - Path: `/webhook/audio-transcription`
   - Response Mode: Immediately

2. **Expected Payload**:
   - `transcript` (file): The transcript text file
   - `audio` (file): The audio WAV file (if < MAX_FILE_SIZE)
   - `metadata` (file): The JSON metadata file
   - `payload` (string): JSON with file info and metadata

3. **Sample Workflow**:
   ```
   Webhook → Extract Files → Process Transcript → [Your Logic]
   ```

## File Structure

For each transcription session, the forwarder sends:
- `final_omi_inference_*.txt` - Transcript with speaker labels
- `final_omi_inference_*.wav` - Combined audio
- `final_omi_inference_*.json` - Detailed metadata (segments, language, speakers)
- Creates `*.txt.n8n_sent` marker after successful upload

## Monitoring

Check container logs:
```bash
docker logs -f n8n-forwarder
```

Expected log entries:
- `New transcript detected` - File watcher triggered
- `Sending to n8n` - Upload started
- `✓ Successfully sent to n8n` - Upload completed
- `Already processed` - Skipped duplicate

## Troubleshooting

**Files not being sent?**
- Check n8n webhook URL is correct
- Verify n8n is accessible from container
- Check file permissions on `/exports/final`
- Review logs: `docker logs n8n-forwarder`

**Duplicates being sent?**
- Marker files may have been deleted
- Check `.n8n_sent` files exist alongside transcripts

**Large files failing?**
- Increase `MAX_FILE_SIZE` environment variable
- Check n8n webhook payload size limits
- Review n8n node memory settings
