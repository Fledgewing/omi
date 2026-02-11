#!/usr/bin/env python3
"""
N8N Forwarder: Monitors /exports/final for completed transcriptions and forwards to n8n.
"""
import os
import time
import json
import logging
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path

# Configure logging
log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s.%(msecs)03d [%(threadName)-12s %(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()],
    force=True
)
logger = logging.getLogger(__name__)

# Configuration
WATCH_DIR = os.environ.get('WATCH_DIR', '/exports/final')
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL', 'http://n8n.ml.svc.cluster.local:5678/webhook/audio-transcription')
STABILITY_DELAY = float(os.environ.get('STABILITY_DELAY', '2'))  # Wait for file to be fully written
SCAN_INTERVAL = float(os.environ.get('SCAN_INTERVAL', '300'))  # Periodic scan every 5 minutes (300s)
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', str(512 * 1024 * 1024)))  # 100MB default

logger.info(f"N8N Forwarder starting...")
logger.info(f"Watch directory: {WATCH_DIR}")
logger.info(f"N8N webhook URL: {N8N_WEBHOOK_URL}")
logger.info(f"Stability delay: {STABILITY_DELAY}s")
logger.info(f"Scan interval: {SCAN_INTERVAL}s")
logger.info(f"Max file size: {MAX_FILE_SIZE / (1024*1024):.1f}MB")

# Track processed files in memory for current session to avoid duplicate processing
processed_files = set()


def delete_files(txt_path):
    """Delete transcript and related files after successful send."""
    base_path = txt_path.rsplit('.', 1)[0]
    files_to_delete = [
        txt_path,
        base_path + '.wav',
        base_path + '.json',
        base_path + '.txto',
    ]
    
    deleted_count = 0
    for filepath in files_to_delete:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.debug(f"Deleted: {os.path.basename(filepath)}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {filepath}: {e}")
    
    logger.info(f"Deleted {deleted_count} files for {os.path.basename(txt_path)}")


def wait_for_file_stability(filepath, timeout=10):
    """Wait for file to stop being modified (file size stable)."""
    if not os.path.exists(filepath):
        return False
    
    prev_size = -1
    stable_count = 0
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            current_size = os.path.getsize(filepath)
            if current_size == prev_size:
                stable_count += 1
                if stable_count >= 2:  # Size stable for 2 checks
                    return True
            else:
                stable_count = 0
                prev_size = current_size
            time.sleep(STABILITY_DELAY / 2)
        except Exception as e:
            logger.warning(f"Error checking file stability: {e}")
            time.sleep(0.5)
    
    return True  # Proceed anyway after timeout


def send_to_n8n(txt_path):
    """Send transcription files to n8n webhook."""
    logger.info(f"Processing: {os.path.basename(txt_path)}")
    
    # Wait for file to be stable
    if not wait_for_file_stability(txt_path):
        logger.warning(f"File unstable, skipping: {os.path.basename(txt_path)}")
        return
    
    # Find related files
    base_path = txt_path.rsplit('.', 1)[0]
    wav_path = base_path + '.wav'
    json_path = base_path + '.json'
    
    # Read transcript
    try:
        with open(txt_path, 'r') as f:
            transcript = f.read()
    except Exception as e:
        logger.error(f"Failed to read transcript {txt_path}: {e}")
        return
    
    # Skip empty transcripts - just delete them
    if not transcript.strip():
        logger.warning(f"Empty transcript, deleting: {os.path.basename(txt_path)}")
        delete_files(txt_path)
        processed_files.add(txt_path)
        return
    
    # Read metadata if available
    metadata = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read metadata {json_path}: {e}")
    
    # Prepare payload
    payload = {
        'transcript': transcript,
        'metadata': metadata,
        'files': {
            'transcript_file': os.path.basename(txt_path),
            'audio_file': os.path.basename(wav_path) if os.path.exists(wav_path) else None,
            'metadata_file': os.path.basename(json_path) if os.path.exists(json_path) else None,
        },
        'file_sizes': {
            'transcript': os.path.getsize(txt_path),
            'audio': os.path.getsize(wav_path) if os.path.exists(wav_path) else 0,
            'metadata': os.path.getsize(json_path) if os.path.exists(json_path) else 0,
        },
        'timestamp': time.time(),
    }
    
    # Prepare multipart upload with actual files
    files = {}
    try:
        # Add transcript file
        files['transcript'] = (os.path.basename(txt_path), open(txt_path, 'rb'), 'text/plain')
        
        # Add audio file if exists and not too large
        if os.path.exists(wav_path):
            wav_size = os.path.getsize(wav_path)
            if wav_size <= MAX_FILE_SIZE:
                files['audio'] = (os.path.basename(wav_path), open(wav_path, 'rb'), 'audio/wav')
            else:
                logger.warning(f"Audio file too large ({wav_size / (1024*1024):.1f}MB), skipping upload")
                payload['audio_skipped'] = True
                payload['audio_size_mb'] = wav_size / (1024*1024)
        
        # Add metadata file if exists
        if os.path.exists(json_path):
            files['metadata'] = (os.path.basename(json_path), open(json_path, 'rb'), 'application/json')
        
        # Send to n8n
        logger.info(f"Sending to n8n: {os.path.basename(txt_path)} (transcript={len(transcript)} chars, files={len(files)})")
        
        response = requests.post(
            N8N_WEBHOOK_URL,
            data={'payload': json.dumps(payload)},
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            logger.info(f"✓ Successfully sent to n8n: {os.path.basename(txt_path)}")
            delete_files(txt_path)
            processed_files.add(txt_path)
        else:
            logger.error(f"Failed to send to n8n (status {response.status_code}): {response.text[:200]}")
            logger.error(f"Files retained for retry: {os.path.basename(txt_path)}")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send to n8n: {e}")
    except Exception as e:
        logger.error(f"Error processing {txt_path}: {e}")
    finally:
        # Close all file handles
        for f in files.values():
            try:
                f[1].close()
            except:
                pass


class TranscriptHandler(FileSystemEventHandler):
    """Watch for new .txt files in the final directory."""
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith('.txt'):
            logger.info(f"New transcript detected: {os.path.basename(event.src_path)}")
            # Small delay to ensure file is fully written
            time.sleep(STABILITY_DELAY)
            send_to_n8n(event.src_path)
    
    def on_modified(self, event):
        """Handle file modifications (in case creation event was missed)."""
        if event.is_directory:
            return
        
        if event.src_path.endswith('.txt'):
            if event.src_path not in processed_files:
                logger.info(f"Modified transcript detected: {os.path.basename(event.src_path)}")
                time.sleep(STABILITY_DELAY)
                send_to_n8n(event.src_path)


def scan_existing_files():
    """Process any existing .txt files on startup."""
    logger.info("Scanning for existing transcripts...")
    count = 0
    for filename in os.listdir(WATCH_DIR):
        if filename.endswith('.txt'):
            txt_path = os.path.join(WATCH_DIR, filename)
            send_to_n8n(txt_path)
            count += 1
    logger.info(f"Processed {count} existing transcripts")


def main():
    """Main entry point."""
    os.makedirs(WATCH_DIR, exist_ok=True)
    
    # Process existing files first
    scan_existing_files()
    
    # Set up file watcher
    event_handler = TranscriptHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()
    
    logger.info(f"👀 Watching {WATCH_DIR} for new transcripts...")
    logger.info(f"🔄 Periodic scan every {SCAN_INTERVAL}s for missed files...")
    
    last_scan_time = time.time()
    
    try:
        while True:
            time.sleep(1)
            
            # Periodically scan for any missed files
            if time.time() - last_scan_time >= SCAN_INTERVAL:
                logger.info("Running periodic scan for unprocessed files...")
                scan_existing_files()
                last_scan_time = time.time()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        observer.stop()
    
    observer.join()


if __name__ == '__main__':
    main()
