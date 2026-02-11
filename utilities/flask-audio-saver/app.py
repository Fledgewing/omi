from flask import Flask, request, jsonify
import base64
import os
import json
import time
import threading
import wave
import torch
import logging
import pandas as pd
from faster_whisper import WhisperModel
from datetime import datetime
import whisperx  # New: for transcription + alignment + diarization
from pyannote.audio import Pipeline  # For speaker diarization
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pyannote')
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')
warnings.filterwarnings('ignore', category=UserWarning, module='speechbrain')

# Configure logging first (thread-safe, timestamps, levels)
log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s.%(msecs)03d [%(threadName)-12s %(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()],  # Ensure stdout
    force=True
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Config: Set your save path (local folder/drive) - use env var for Docker flexibility
SAVE_DIR = os.environ.get('SAVE_DIR', '/exports')  # Default to /exports in container
FINAL_DIR = os.path.join(SAVE_DIR, 'final')
PERSISTENT_SESSIONS_PATH = os.path.join(SAVE_DIR, 'sessions.json')
INACTIVITY_TIMEOUT = int(os.environ.get('INACTIVITY_TIMEOUT', '90'))  # Seconds of inactivity before finalizing
CHECK_INTERVAL = 5.0  # Background check interval (seconds)
STALE_FILE_TIMEOUT = 3600  # 1 hour: remove files older than this from lists
IP_TIMEOUT = 600  # 10min: remove inactive IP sessions
SMALL_MODEL_SIZE = os.environ.get('SMALL_MODEL_SIZE', 'tiny')
LARGE_MODEL_SIZE = os.environ.get('LARGE_MODEL_SIZE', 'tiny')
USE_LARGE_MODEL_FOR_FINAL = os.environ.get('USE_LARGE_MODEL_FOR_FINAL', 'true').lower() in ('true', '1', 'yes')
HF_TOKEN = os.environ.get('HF_TOKEN')  # New: For pyannote diarization model access
MIN_SPEECH_CHARS = int(os.environ.get('MIN_SPEECH_CHARS', '5'))  # Min chars in small-model text to keep as non-silent
MODEL_DIR = os.environ.get('WHISPER_MODEL_DIR', os.path.join(SAVE_DIR, 'whisper-models'))
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Set HuggingFace cache to persist models
os.environ['HF_HOME'] = os.path.join(SAVE_DIR, 'huggingface-cache')
os.environ['TORCH_HOME'] = os.path.join(SAVE_DIR, 'torch-cache')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)

logger.info(f"LOG_LEVEL={log_level_str}, SAVE_DIR={SAVE_DIR}, FINAL_DIR={FINAL_DIR}")
logger.info(f"INACTIVITY_TIMEOUT={INACTIVITY_TIMEOUT}s, MIN_SPEECH_CHARS={MIN_SPEECH_CHARS}")
logger.info(f"Small model: {SMALL_MODEL_SIZE}, Large model: {LARGE_MODEL_SIZE}")
logger.info(f"Use large model for final: {USE_LARGE_MODEL_FOR_FINAL}")
logger.info(f"HF_TOKEN: {'set' if HF_TOKEN else 'NOT SET (diarization skipped)'}")

# Monkey-patch torch.load to use weights_only=False
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False  # Force False, don't check if it exists
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# Load faster-whisper models at startup (downloads to MODEL_DIR if missing)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
logger.info(f"Loading models on {DEVICE} ({COMPUTE_TYPE}) from/to {MODEL_DIR}")

small_model = None
large_model = None
whisperx_model = None
diarize_pipeline = None

try:
    small_model = WhisperModel(SMALL_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=MODEL_DIR)
    logger.info("Small faster-whisper model loaded successfully.")

    if USE_LARGE_MODEL_FOR_FINAL:
        large_model = WhisperModel(LARGE_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=MODEL_DIR)
        logger.info("Large faster-whisper model loaded successfully (fallback).")

        # New: Load WhisperX for enhanced final transcription + diarization
        whisperx_model = whisperx.load_model(LARGE_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=MODEL_DIR)
        logger.info("WhisperX ASR model loaded successfully.")

        if HF_TOKEN:
            diarize_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN
            ).to(torch.device(DEVICE))
            
            logger.info("Pyannote diarization pipeline loaded successfully.")
        else:
            logger.warning("No HF_TOKEN: Diarization skipped. Set HF_TOKEN and accept https://huggingface.co/pyannote/speaker-diarization-3.1")
    else:
        logger.info("Large model disabled (USE_LARGE_MODEL_FOR_FINAL=false)")
        large_model = None
        whisperx_model = None
        diarize_pipeline = None
except Exception as e:
    logger.error(f"Failed to load one or more models: {e}")
    # Don't raise; fallback to basic transcription

logger.info("Models ready. For diarization: Ensure HF_TOKEN is set and models accepted on HF Hub.")

# Globals for persistent sessions (now only track WAV paths, no transcripts)
recent_sessions_per_ip = {}
last_activity_per_ip = {}
sessions_lock = threading.RLock()  # RLock for reentrancy

def init_sessions():
    global recent_sessions_per_ip, last_activity_per_ip
    recent_sessions_per_ip = {}
    last_activity_per_ip = {}
    current_time = time.time()
    
    # Load persisted sessions first
    if os.path.exists(PERSISTENT_SESSIONS_PATH):
        try:
            with open(PERSISTENT_SESSIONS_PATH, 'r') as f:
                data = json.load(f)
            # Migrate old sessions: extract 'wav' paths only, discard transcripts
            migrated_count = 0
            for ip, old_sessions in data.get('sessions', {}).items():
                new_sessions = []
                for s in old_sessions:
                    wav_path = s.get('wav') or s.get('path', '')
                    if wav_path and os.path.exists(wav_path):
                        new_sessions.append({'wav': wav_path})
                        migrated_count += 1
                if new_sessions:
                    recent_sessions_per_ip[ip] = new_sessions
            last_activity_per_ip = data.get('last_activities', {})
            logger.info(f"Loaded/migrated {migrated_count} sessions from {PERSISTENT_SESSIONS_PATH}")
        except Exception as e:
            logger.error(f"Failed to load/migrate sessions: {e}")
    
    # Scan SAVE_DIR for existing WAV files and rebuild sessions
    logger.info(f"Scanning {SAVE_DIR} for existing WAV files...")
    scanned_wavs = 0
    added_to_session = 0
    orphaned_json = 0
    orphaned_txt = 0
    stale_files = 0
    
    try:
        for filename in os.listdir(SAVE_DIR):
            filepath = os.path.join(SAVE_DIR, filename)
            
            # Skip directories
            if os.path.isdir(filepath):
                continue
            
            file_age = current_time - os.path.getmtime(filepath)
            
            # Clean up stale files (older than timeout)
            # if file_age > STALE_FILE_TIMEOUT:
            #     try:
            #         os.remove(filepath)
            #         stale_files += 1
            #         logger.debug(f"Removed stale file (age {file_age/3600:.1f}h): {filename}")
            #     except Exception as e:
            #         logger.error(f"Failed to remove stale file {filename}: {e}")
            #     continue
            
            # Process WAV files
            if filename.startswith('omi_inference_') and filename.endswith('.wav'):
                scanned_wavs += 1
                json_path = os.path.splitext(filepath)[0] + '.json'
                small_txt_path = os.path.splitext(filepath)[0] + '_small.txt'
                
                # Skip if still has .json marker (not yet processed by silence checker)
                if os.path.exists(json_path):
                    continue
                
                # Extract IP from small model txt file if it exists
                ip = None
                if os.path.exists(small_txt_path):
                    # WAV has been through silence check and kept - add to session
                    # Try to determine IP from existing session data or use a default
                    # Check if this WAV is already in any session
                    found = False
                    for existing_ip, sessions in recent_sessions_per_ip.items():
                        if any(s['wav'] == filepath for s in sessions):
                            ip = existing_ip
                            found = True
                            break
                    
                    if not found:
                        # New WAV not in sessions - try to infer IP or use 'unknown'
                        ip = 'unknown'
                        for existing_ip in recent_sessions_per_ip.keys():
                            # Simple heuristic: assign to most recent IP
                            ip = existing_ip
                            break
                    
                    if ip:
                        ip_sessions = recent_sessions_per_ip.setdefault(ip, [])
                        if not any(s['wav'] == filepath for s in ip_sessions):
                            ip_sessions.append({'wav': filepath})
                            added_to_session += 1
                            # Update last activity to current time (prevents immediate finalization on startup)
                            last_activity_per_ip[ip] = current_time
            
            # Clean up orphaned .json status files (WAV no longer exists)
            elif filename.startswith('omi_inference_') and filename.endswith('.json'):
                wav_path = os.path.splitext(filepath)[0] + '.wav'
                if not os.path.exists(wav_path):
                    try:
                        os.remove(filepath)
                        orphaned_json += 1
                        logger.debug(f"Removed orphaned JSON: {filename}")
                    except Exception as e:
                        logger.error(f"Failed to remove orphaned JSON {filename}: {e}")
            
            # Clean up orphaned _small.txt files (WAV no longer exists)
            elif filename.startswith('omi_inference_') and filename.endswith('_small.txt'):
                wav_path = filepath.rsplit('_small.txt', 1)[0] + '.wav'
                if not os.path.exists(wav_path):
                    try:
                        os.remove(filepath)
                        orphaned_txt += 1
                        logger.debug(f"Removed orphaned small.txt: {filename}")
                    except Exception as e:
                        logger.error(f"Failed to remove orphaned txt {filename}: {e}")
        
        logger.info(f"Startup scan complete: scanned_wavs={scanned_wavs}, added_to_session={added_to_session}, "
                   f"orphaned_json={orphaned_json}, orphaned_txt={orphaned_txt}, stale_files={stale_files}")
        logger.info(f"Active sessions: {len(recent_sessions_per_ip)} IPs, "
                   f"{sum(len(s) for s in recent_sessions_per_ip.values())} total WAVs")
        
    except Exception as e:
        logger.error(f"Failed during startup scan: {e}")
    
    # Always save session state after initialization (includes both loaded and scanned sessions)
    save_sessions()

def save_sessions():
    try:
        with sessions_lock:
            data = {
                'sessions': recent_sessions_per_ip,
                'last_activities': last_activity_per_ip
            }
            tmp_path = PERSISTENT_SESSIONS_PATH + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, PERSISTENT_SESSIONS_PATH)
    except Exception as e:
        logger.error(f"Failed to save sessions: {e}")

def combine_wavs(filepaths, output_path):
    """Concatenate WAV files preserving format and fixing header."""
    if not filepaths:
        return
    # Read first for params
    with wave.open(filepaths[0], 'rb') as first_wav:
        nchannels, sampwidth, framerate, comptype, compname, _ = first_wav.getparams()
        frames = first_wav.readframes(first_wav.getnframes())
        total_frames = first_wav.getnframes()
    # Append rest
    for path in filepaths[1:]:
        with wave.open(path, 'rb') as wav_file:
            if (wav_file.getnchannels() != nchannels or
                wav_file.getsampwidth() != sampwidth or
                wav_file.getframerate() != framerate):
                raise ValueError(f"Incompatible WAV format in {path}")
            additional_frames = wav_file.readframes(wav_file.getnframes())
            frames += additional_frames
            total_frames += wav_file.getnframes()
    # Write output
    params = (nchannels, sampwidth, framerate, comptype, compname, total_frames)
    with wave.open(output_path, 'wb') as out_wav:
        out_wav.setparams(params)
        out_wav.writeframes(frames)

def process_pending_transcriptions():
    """Background task: scan for WAVs with .json status='saved', check silence with small model, add to session if speech."""
    logger.info("=== Starting silence check scan ===")
    processed_count = 0
    silence_deleted = 0
    speech_added = 0
    current_time = time.time()
    for filename in os.listdir(SAVE_DIR):
        if not (filename.startswith('omi_inference_') and filename.endswith('.wav')):
            continue
        wav_path = os.path.join(SAVE_DIR, filename)
        if not os.path.exists(wav_path):
            continue
        json_path = os.path.splitext(wav_path)[0] + '.json'
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            logger.warning(f"Invalid JSON at {json_path}: {e}, skipping")
            continue
        if meta.get('status') != 'saved':
            continue
        ip = meta.get('ip')
        if not ip:
            logger.warning(f"No IP in metadata {json_path}, skipping")
            continue
        logger.info(f"[{ip}] Quick silence check on {os.path.basename(wav_path)}")
        text = ''
        try:
            segments, _ = small_model.transcribe(wav_path, beam_size=1)
            text = ' '.join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            logger.error(f"[{ip}] Small model error for {wav_path}: {e}")
            continue  # retry later
        if len(text) < MIN_SPEECH_CHARS:
            logger.info(f"[{ip}] Silence/short ({len(text)} chars): {wav_path}, deleting")
            try:
                os.remove(wav_path)
                os.remove(json_path)
            except Exception as e:
                logger.error(f"Failed to delete silence {wav_path}: {e}")
            silence_deleted += 1
            continue

        # Save small model result for debugging/comparison
        small_model_txt = os.path.splitext(wav_path)[0] + '_small.txt'
        try:
            with open(small_model_txt, 'w') as f:
                f.write(f"Small Model ({SMALL_MODEL_SIZE}) Transcription:\n")
                f.write(f"Length: {len(text)} chars\n")
                f.write(f"Threshold: {MIN_SPEECH_CHARS} chars\n")
                f.write(f"Status: KEPT (speech detected)\n")
                f.write(f"\nTranscript:\n{text}")
            logger.debug(f"[{ip}] Saved small model result to {os.path.basename(small_model_txt)}")
        except Exception as e:
            logger.error(f"[{ip}] Failed to save small model result: {e}")
        # Add to session safely (no transcript stored)
        with sessions_lock:
            ip_sessions = recent_sessions_per_ip.setdefault(ip, [])
            if any(s['wav'] == wav_path for s in ip_sessions):
                logger.info(f"[{ip}] {wav_path} already in session, skipping")
                try:
                    os.remove(json_path)
                except Exception:
                    pass
                continue
            ip_sessions.append({'wav': wav_path})
            last_activity_per_ip[ip] = current_time
            save_sessions()
            logger.info(f"[{ip}] Added speech chunk ({len(ip_sessions)} total in session)")
        # Cleanup marker
        try:
            os.remove(json_path)
        except Exception:
            pass
        speech_added += 1
        processed_count += 1
    logger.info(f"=== Silence check scan complete: processed={processed_count}, speech_added={speech_added}, silence_deleted={silence_deleted} ===")
    # Reschedule
    timer = threading.Timer(CHECK_INTERVAL, process_pending_transcriptions)
    timer.daemon = True
    timer.start()

def finalize_pending_files():
    """Background task: check per-IP sessions after inactivity. Merge WAVs, transcribe with WhisperX (+diarization), clear pending."""
    logger.info("=== Starting finalize scan ===")
    global recent_sessions_per_ip, last_activity_per_ip
    finalized_count = 0
    stale_cleaned = 0
    with sessions_lock:
        current_time = time.time()
        to_remove_ips = []
        for ip in list(recent_sessions_per_ip):
            session_list = recent_sessions_per_ip[ip]
            
            # Separate stale from recent sessions
            stale_sessions = []
            recent_sessions = []
            for s in session_list:
                if not os.path.exists(s['wav']):
                    continue  # Skip missing files
                file_age = current_time - os.path.getmtime(s['wav'])
                if file_age > STALE_FILE_TIMEOUT:
                    stale_sessions.append(s)
                else:
                    recent_sessions.append(s)
            
            if ip not in last_activity_per_ip:
                last_activity_per_ip[ip] = current_time
            inactivity = current_time - last_activity_per_ip[ip]
            
            # Determine which sessions to finalize
            sessions_to_finalize = []
            finalize_reason = ""
            
            # Process stale sessions immediately (old files that need to be finalized)
            if stale_sessions:
                sessions_to_finalize = stale_sessions
                finalize_reason = f"stale (age > {STALE_FILE_TIMEOUT/3600:.1f}h)"
                stale_cleaned += len(stale_sessions)
            # Or process recent sessions if inactive long enough
            elif inactivity > INACTIVITY_TIMEOUT and recent_sessions:
                sessions_to_finalize = recent_sessions
                finalize_reason = f"inactive for {inactivity:.1f}s"
            
            # Finalize if we have sessions to process
            if sessions_to_finalize:
                logger.info(f"[{ip}] Finalizing {len(sessions_to_finalize)} sessions ({finalize_reason})")
                valid_sessions = sorted(sessions_to_finalize, key=lambda s: os.path.getmtime(s['wav']))
                num_valid = len(valid_sessions)
                final_filename = f"final_omi_inference_{ip}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                final_wav_path = os.path.join(FINAL_DIR, final_filename)
                wav_paths = [s['wav'] for s in valid_sessions]
                try:
                    if num_valid > 1:
                        combine_wavs(wav_paths, final_wav_path)
                        logger.info(f"[{ip}] Combined {num_valid} WAVs into {final_wav_path}")
                    else:
                        os.rename(valid_sessions[0]['wav'], final_wav_path)
                        logger.info(f"[{ip}] Moved single WAV to {final_wav_path}")

                    # New: WhisperX transcription + alignment + speaker diarization
                    final_txt_path = final_wav_path.rsplit('.', 1)[0] + '.txto'
                    final_json_path = final_wav_path.rsplit('.', 1)[0] + '.json'
                    if USE_LARGE_MODEL_FOR_FINAL and whisperx_model:
                        logger.info(f"[{ip}] Transcribing/aligning/diarizing with WhisperX: {os.path.basename(final_wav_path)}")
                        try:
                            audio = whisperx.load_audio(final_wav_path)
                            result = whisperx_model.transcribe(audio, batch_size=16, language=None)
                            language = result.get('language', 'en')
                            logger.info(f"[{ip}] Language: {language}")

                            # Align
                            try:
                                model_a, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
                                result_before = len(result.get("segments", []))
                                result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE)
                                result_after = len(result.get("segments", []))
                                
                                logger.info(f"[{ip}] Alignment complete. Segments before: {result_before}, after: {result_after}")
                                
                                # Check if alignment produced word-level data
                                if result.get("segments"):
                                    sample_seg = result["segments"][0]
                                    has_words = 'words' in sample_seg and sample_seg.get('words')
                                    logger.info(f"[{ip}] First segment has word-level data: {has_words}")
                                    if has_words:
                                        logger.info(f"[{ip}] Sample: {len(sample_seg['words'])} words in first segment")
                            except Exception as align_e:
                                logger.error(f"[{ip}] Alignment failed: {align_e}", exc_info=True)
                                logger.warning(f"[{ip}] Continuing with segment-level transcription only")

                            # Diarize & assign speakers (only if we have segments)
                            if diarize_pipeline and result.get("segments"):
                                try:
                                    logger.info(f"[{ip}] Running speaker diarization...")
                                    diarize_segments = diarize_pipeline(final_wav_path)
                                    
                                    # Convert pyannote Annotation to pandas DataFrame (what WhisperX expects internally)
                                    import pandas as pd
                                    
                                    diarize_list = []
                                    for turn, _, speaker in diarize_segments.itertracks(yield_label=True):
                                        diarize_list.append({
                                            'start': float(turn.start),
                                            'end': float(turn.end),
                                            'speaker': speaker
                                        })
                                    
                                    # Create DataFrame from list
                                    diarize_df = pd.DataFrame(diarize_list)
                                    
                                    unique_speakers = len(diarize_df['speaker'].unique()) if len(diarize_df) > 0 else 0
                                    logger.info(f"[{ip}] Diarization found {unique_speakers} speakers in {len(diarize_df)} segments")
                                    logger.debug(f"[{ip}] Diarization DataFrame:\n{diarize_df.head()}")
                                    
                                    # Check for word-level data
                                    has_words = result.get('segments') and any('words' in seg and seg.get('words') for seg in result['segments'])
                                    logger.info(f"[{ip}] Has word-level alignments: {has_words}")
                                    
                                    if not has_words:
                                        logger.warning(f"[{ip}] No word-level alignments, skipping speaker assignment")
                                    elif len(diarize_df) == 0:
                                        logger.warning(f"[{ip}] No diarization segments, skipping speaker assignment")
                                    else:
                                        # Call assign_word_speakers with DataFrame
                                        logger.info(f"[{ip}] Calling assign_word_speakers with DataFrame...")
                                        logger.debug(f"[{ip}] Number of transcript segments: {len(result['segments'])}")
                                        
                                        result = whisperx.assign_word_speakers(diarize_df, result)
                                        logger.info(f"[{ip}] Speaker diarization assigned successfully.")
                                    
                                except Exception as diar_e:
                                    logger.error(f"[{ip}] Diarization failed: {diar_e}", exc_info=True)
                                    logger.warning(f"[{ip}] Continuing without speaker labels")

                            # Build speaker-labeled transcript
                            segments = result.get("segments", [])
                            transcript_lines = []
                            speakers = set()

                            for seg in segments:
                                text = seg.get('text', '').strip()
                                if not text:
                                    continue
                                
                                speaker = seg.get('speaker', 'Unknown')
                                if speaker and speaker != 'Unknown':
                                    speakers.add(speaker)
                                
                                transcript_lines.append(f"{speaker}: {text}")

                            final_transcript = '\n'.join(transcript_lines)
                            speakers_list = list(speakers) if speakers else []
                            logger.info(f"[{ip}] Found {len(speakers_list)} speakers: {speakers_list}")

                            # Save TXT
                            with open(final_txt_path, 'w') as f:
                                f.write(final_transcript)
                            logger.info(f"[{ip}] Saved speaker transcript ({len(final_transcript)} chars) to {os.path.basename(final_txt_path)}")

                            # Save detailed JSON
                            full_result = {
                                'transcript': final_transcript,
                                'language': language,
                                'segments': segments,
                                'num_speakers': len(speakers_list)
                            }
                            with open(final_json_path, 'w') as f:
                                json.dump(full_result, f, indent=2)
                            logger.info(f"[{ip}] Saved detailed JSON to {os.path.basename(final_json_path)}")

                            # Check if final transcript is too short - delete if so
                            if len(final_transcript.strip()) < MIN_SPEECH_CHARS:
                                logger.info(f"[{ip}] Final transcript too short ({len(final_transcript)} chars), deleting output files")
                                try:
                                    if os.path.exists(final_wav_path):
                                        os.remove(final_wav_path)
                                    if os.path.exists(final_txt_path):
                                        os.remove(final_txt_path)
                                    if os.path.exists(final_json_path):
                                        os.remove(final_json_path)
                                    # Also remove small model comparison file
                                    small_comparison_path = final_wav_path.rsplit('.', 1)[0] + '_small_comparison.txt'
                                    if os.path.exists(small_comparison_path):
                                        os.remove(small_comparison_path)
                                    logger.info(f"[{ip}] Deleted short/empty final output")
                                    
                                    # Clean up original WAVs and their small model files
                                    for s in valid_sessions:
                                        orig_wav = s['wav']
                                        if os.path.exists(orig_wav):
                                            os.remove(orig_wav)
                                        small_txt = orig_wav.rsplit('.', 1)[0] + '_small.txt'
                                        if os.path.exists(small_txt):
                                            os.remove(small_txt)
                                except Exception as del_e:
                                    logger.error(f"[{ip}] Failed to delete short output files: {del_e}")
                                # Clear session and skip
                                session_list[:] = []
                                continue
                        except Exception as wx_e:
                            logger.error(f"[{ip}] WhisperX failed: {wx_e}", exc_info=True)  # Add exc_info=True for full traceback
                            raise wx_e  # Trigger fallback
                    else:
                        # Simple mode: Combine small model transcripts
                        logger.info(f"[{ip}] Combining small model transcripts (simple mode)")
                        try:
                            combined_transcript_parts = []
                            
                            for i, s in enumerate(valid_sessions, 1):
                                small_txt_path = s['wav'].rsplit('.', 1)[0] + '_small.txt'
                                if os.path.exists(small_txt_path):
                                    try:
                                        with open(small_txt_path, 'r') as f:
                                            content = f.read()
                                            # Extract just the transcript part (after "Transcript:\n")
                                            if "Transcript:\n" in content:
                                                transcript = content.split("Transcript:\n", 1)[1].strip()
                                                combined_transcript_parts.append(transcript)
                                            else:
                                                combined_transcript_parts.append(content.strip())
                                    except Exception as e:
                                        logger.warning(f"[{ip}] Failed to read {small_txt_path}: {e}")
                            
                            final_transcript = '\n'.join(combined_transcript_parts)
                            
                            # Save TXT
                            with open(final_txt_path, 'w') as f:
                                f.write(final_transcript)
                            logger.info(f"[{ip}] Saved combined transcript ({len(final_transcript)} chars) to {os.path.basename(final_txt_path)}")
                            
                            # Save JSON
                            full_result = {
                                'transcript': final_transcript,
                                'language': 'unknown',
                                'model': SMALL_MODEL_SIZE,
                                'mode': 'combined_small_model',
                                'num_chunks': len(combined_transcript_parts)
                            }
                            with open(final_json_path, 'w') as f:
                                json.dump(full_result, f, indent=2)
                            logger.info(f"[{ip}] Saved JSON to {os.path.basename(final_json_path)}")
                            
                        except Exception as simple_e:
                            logger.error(f"[{ip}] Simple combine failed: {simple_e}", exc_info=True)
                            raise simple_e  # Trigger fallback

                    # Clean up originals (after success)
                    removed_count = 0
                    for s in valid_sessions:
                        orig_wav = s['wav']
                        if orig_wav != final_wav_path and os.path.exists(orig_wav):
                            try:
                                os.remove(orig_wav)
                                removed_count += 1
                                # Also remove the small model txt if it exists
                                small_txt = orig_wav.rsplit('.', 1)[0] + '_small.txt'
                                if os.path.exists(small_txt):
                                    os.remove(small_txt)
                                    logger.debug(f"[{ip}] Removed small model file: {os.path.basename(small_txt)}")
                            except Exception as e:
                                logger.error(f"[{ip}] Failed to remove {orig_wav}: {e}")
                    logger.debug(f"[{ip}] Removed {removed_count} original WAVs")

                    # Update session list: remove finalized sessions, keep remaining recent ones
                    session_list[:] = [s for s in recent_sessions if s not in sessions_to_finalize]
                    logger.info(f"[{ip}] Session finalized and cleared ({len(session_list)} sessions remaining)")
                    finalized_count += 1
                except Exception as e:
                    logger.error(f"[{ip}] Finalize failed: {e}", exc_info=True)
                    # Fallback: Basic large-model transcript (no speakers)
                    if USE_LARGE_MODEL_FOR_FINAL and large_model:
                        try:
                            segments, _ = large_model.transcribe(final_wav_path, beam_size=5)
                            final_transcript = ' '.join(seg.text.strip() for seg in segments).strip()
                            if final_transcript:
                                final_txt_path = final_wav_path.rsplit('.', 1)[0] + '.txto'
                                with open(final_txt_path, 'w') as f:
                                    f.write(final_transcript)
                                logger.info(f"[{ip}] Fallback transcript ({len(final_transcript)} chars) saved.")
                        except Exception as fb_e:
                            logger.error(f"[{ip}] Fallback also failed: {fb_e}")
            # Remove empty inactive IPs
            if not session_list and (current_time - last_activity_per_ip.get(ip, 0)) > IP_TIMEOUT:
                to_remove_ips.append(ip)
                logger.debug(f"Removing inactive empty IP: {ip}")
        for ip in to_remove_ips:
            recent_sessions_per_ip.pop(ip, None)
            last_activity_per_ip.pop(ip, None)
        # Save after processing
        save_sessions()
    logger.info(f"=== Finalize scan complete: finalized={finalized_count}, stale_cleaned={stale_cleaned} ===")
    # Reschedule daemon timer
    timer = threading.Timer(CHECK_INTERVAL, finalize_pending_files)
    timer.daemon = True
    timer.start()

# Initialize sessions
init_sessions()
logger.info(f"Sessions persistence: {PERSISTENT_SESSIONS_PATH}")
logger.info("Background tasks starting...")

# Start background tasks
transcribe_timer = threading.Timer(3.0, process_pending_transcriptions)
transcribe_timer.daemon = True
transcribe_timer.start()
finalize_timer = threading.Timer(3.0, finalize_pending_files)
finalize_timer.daemon = True
finalize_timer.start()

def get_timestamp_from_request():
    """Get timestamp from X-Original-Timestamp header or use current time.
    Returns datetime object."""
    timestamp_header = request.headers.get('X-Original-Timestamp')
    if timestamp_header:
        try:
            # Try parsing ISO format first
            logger.info(f"Using X-Original-Timestamp header: {timestamp_header}")
            return datetime.fromisoformat(timestamp_header.replace('Z', '+00:00'))
        except:
            try:
                # Try Unix timestamp (seconds since epoch)
                logger.info(f"Using X-Original-Timestamp header: {timestamp_header}")
                return datetime.fromtimestamp(float(timestamp_header))
            except:
                logger.warning(f"Invalid X-Original-Timestamp header: {timestamp_header}, using current time")
    else:
        logger.warning("No X-Original-Timestamp header, using current time")
    return datetime.now()

@app.before_request
def log_request_info():
    """Log incoming request for debugging"""
    logger.debug(f"Client IP: {request.remote_addr}, Request: {request.method} {request.url}, Content-Type: {request.content_type}")

@app.route('/transcribe', methods=['POST'])
def fake_transcribe():
    content_type = request.content_type
    if not content_type:
        return jsonify({'error': 'Missing Content-Type header. Use application/json or multipart/form-data.'}), 415
    ip = request.remote_addr
    timestamp = get_timestamp_from_request()
    filepath = None
    if 'multipart/form-data' in content_type:
        # Handle file upload
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided in multipart form-data. Use key "audio".'}), 400
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Empty audio file.'}), 400
        filename = f"omi_raw_{timestamp.strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = os.path.join(SAVE_DIR, filename)
        audio_file.save(filepath)
    elif 'application/json' in content_type:
        # Handle base64 JSON
        try:
            data = request.get_json()
        except Exception:
            return jsonify({'error': 'Invalid JSON. Ensure Content-Type: application/json.'}), 400
        audio_b64 = data.get('audio_base64', '')
        if not audio_b64:
            return jsonify({'error': 'No audio_base64 in JSON payload.'}), 400
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            return jsonify({'error': 'Invalid base64 audio data.'}), 400
        filename = f"omi_raw_{timestamp.strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = os.path.join(SAVE_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(audio_bytes)
    else:
        return jsonify({'error': f'Unsupported Content-Type: {content_type}. Use application/json or multipart/form-data.'}), 415

    logger.info(f"[{ip}] Saved raw audio (transcribe endpoint): {os.path.basename(filepath)}")
    mock_response = {
        'text': '',
        'segments': [],
        'language': 'en',
        'success': True,
        'error': None
    }
    return jsonify(mock_response)

@app.route('/inference', methods=['POST'])
def fake_inference():
    ip = request.remote_addr
    timestamp = get_timestamp_from_request()

    # Log full request details
    logger.info(f"[{ip}] === Incoming /inference request ===")
    logger.info(f"[{ip}] Method: {request.method}")
    logger.info(f"[{ip}] Content-Type: {request.content_type}")
    logger.info(f"[{ip}] Headers: {dict(request.headers)}")
    logger.info(f"[{ip}] Content-Length: {request.content_length}")

    content_type = request.content_type
    if not content_type:
        return jsonify({'error': 'Missing Content-Type header. Use application/json or multipart/form-data.'}), 415
    filepath = None
    if 'multipart/form-data' in content_type:
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided in multipart form-data. Use key "file".'}), 400
        input_file = request.files['file']
        if input_file.filename == '':
            return jsonify({'error': 'Empty file.'}), 400
        # Use original filename's extension, but prefix with timestamp (include microseconds for uniqueness)
        original_ext = os.path.splitext(input_file.filename)[1] or '.wav'
        filename = f"omi_inference_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}{original_ext}"
        filepath = os.path.join(SAVE_DIR, filename)
        input_file.save(filepath)
        logger.info(f"[{ip}] Saved inference file: {os.path.basename(filepath)}")
        # Queue for async silence check/transcription if WAV
        if filepath.lower().endswith('.wav'):
            json_path = os.path.splitext(filepath)[0] + '.json'
            meta = {
                'ip': ip,
                'status': 'saved',
                'timestamp': time.time()
            }
            try:
                with open(json_path, 'w') as f:
                    json.dump(meta, f)
                logger.info(f"[{ip}] Queued WAV for async silence check: {os.path.basename(filepath)}")
            except Exception as e:
                logger.error(f"[{ip}] Failed to queue {filepath}: {e}")
    elif 'application/json' in content_type:
        # Handle JSON prompt (no transcription or finalizer)
        try:
            data = request.get_json()
        except Exception:
            return jsonify({'error': 'Invalid JSON. Ensure Content-Type: application/json.'}), 400
        prompt = data.get('prompt', '') or data.get('text', '')
        if not prompt:
            return jsonify({'error': 'No prompt or text in JSON payload.'}), 400
        # Save the raw input prompt to a file
        filename = f"omi_inference_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(SAVE_DIR, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump({'input': prompt, 'timestamp': timestamp.isoformat(), 'ip': ip}, f, indent=2)
            logger.info(f"[{ip}] Saved text prompt: {os.path.basename(filepath)}")
        except Exception as e:
            logger.error(f"[{ip}] Failed to save prompt: {e}")
            return jsonify({'error': 'Failed to save prompt.'}), 500
    else:
        return jsonify({'error': f'Unsupported Content-Type: {content_type}. Use application/json or multipart/form-data.'}), 415

    mock_response = {
        'result': '',
        'success': True,
        'error': None
    }
    return jsonify(mock_response)

if __name__ == '__main__':
    logger.info(f"Server starting on host=0.0.0.0:5000, debug=False")
    logger.info(f"Small: {SMALL_MODEL_SIZE}, Large/WhisperX: {LARGE_MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE}), models in {MODEL_DIR}")
    logger.info(f"Diarization: {'Enabled' if diarize_pipeline else 'Disabled (set HF_TOKEN)'}")
    app.run(host='0.0.0.0', port=5000, debug=False)