import 'dart:io';
import 'dart:typed_data';

import 'package:omi/backend/preferences.dart';
import 'package:omi/backend/schema/conversation.dart';
import 'package:omi/models/custom_stt_config.dart';
import 'package:omi/models/stt_provider.dart';
import 'package:omi/models/stt_result.dart';
import 'package:omi/services/freemium_transcription_service.dart';
import 'package:omi/services/sockets/on_device_whisper_provider.dart';
import 'package:omi/services/sockets/transcription_polling_service.dart';
import 'package:omi/utils/audio/wav_utils.dart';
import 'package:omi/utils/logger.dart';

/// Service to process audio files locally using custom STT instead of uploading to Omi servers
class LocalAudioProcessor {
  static final LocalAudioProcessor _instance = LocalAudioProcessor._internal();
  factory LocalAudioProcessor() => _instance;
  LocalAudioProcessor._internal();

  /// Process audio files locally using the configured custom STT provider
  /// Returns a SyncLocalFilesResponse compatible with the server response format
  Future<SyncLocalFilesResponse> processLocalFiles(List<File> files) async {
    Logger.debug('LocalAudioProcessor: Processing ${files.length} files locally');

    final prefs = SharedPreferencesUtil();
    final customSttConfig = prefs.customSttConfig;

    if (!customSttConfig.isEnabled) {
      throw Exception('Custom STT is not enabled. Cannot process files locally.');
    }

    // For now, we'll process the files but return empty conversation IDs
    // since creating local conversations requires more integration
    final response = SyncLocalFilesResponse(
      newConversationIds: [],
      updatedConversationIds: [],
    );

    for (final file in files) {
      try {
        await _processFile(file, customSttConfig);
      } catch (e) {
        Logger.debug('LocalAudioProcessor: Error processing file ${file.path}: $e');
        rethrow;
      }
    }

    Logger.debug('LocalAudioProcessor: Completed processing ${files.length} files');
    return response;
  }

  Future<void> _processFile(File file, CustomSttConfig config) async {
    Logger.debug('LocalAudioProcessor: Processing file ${file.path}');

    // Read the audio file
    final audioBytes = await file.readAsBytes();

    // Extract timestamp from file's last modified time
    final fileTimestamp = await file.lastModified();
    Logger.debug('LocalAudioProcessor: File timestamp: ${fileTimestamp.toIso8601String()}');

    // Convert to WAV format if needed (assuming files are already in correct format)
    // For WAL files from the device, they should already be in the correct format
    Uint8List processedAudio = audioBytes;

    // Check if we need to convert based on file extension or format
    // WAL files are typically already in 16kHz PCM format
    if (!file.path.endsWith('.wav')) {
      // If it's a raw PCM file, we might need to add WAV header
      processedAudio = WavUtils.createWavFromPcm16(audioBytes, sampleRate: 16000, channels: 1);
    }

    // Transcribe using the appropriate provider
    SttTranscriptionResult? result;

    if (config.provider == SttProvider.localWhisper) {
      result = await _transcribeWithLocalWhisper(processedAudio, config, fileTimestamp);
    } else if (config.provider == SttProvider.onDeviceWhisper) {
      result = await _transcribeWithOnDeviceWhisper(processedAudio, config, fileTimestamp);
    } else if (config.provider == SttProvider.custom) {
      result = await _transcribeWithCustomProvider(processedAudio, config, fileTimestamp);
    } else {
      Logger.debug('LocalAudioProcessor: Provider ${config.provider} not supported for local processing');
      throw Exception('Provider ${config.provider.name} not supported for local file processing');
    }

    if (result != null) {
      Logger.debug('LocalAudioProcessor: Transcription result: ${result.rawText}');
      // TODO: Store transcription locally or create local conversation
      // For now, we just log it to prove it's working
    } else {
      Logger.debug('LocalAudioProcessor: No transcription result');
    }
  }

  Future<SttTranscriptionResult?> _transcribeWithLocalWhisper(
    Uint8List audioData,
    CustomSttConfig config,
    DateTime timestamp,
  ) async {
    Logger.debug('LocalAudioProcessor: Using Local Whisper server at ${config.host}:${config.port}');

    // Create provider for the self-hosted whisper server
    final provider = SchemaBasedSttProvider.localWhisper(
      host: config.host ?? '127.0.0.1',
      port: config.port ?? 8080,
    );

    return await provider.transcribe(audioData, timestamp: timestamp);
  }

  Future<SttTranscriptionResult?> _transcribeWithOnDeviceWhisper(
    Uint8List audioData,
    CustomSttConfig config,
    DateTime timestamp,
  ) async {
    // Try to find a downloaded model
    final freemiumService = FreemiumTranscriptionService();
    await freemiumService.checkReadiness();
    final modelPath = freemiumService.cachedModelPath;

    if (modelPath == null) {
      throw Exception('No Whisper model found. Please download a model first.');
    }

    Logger.debug('LocalAudioProcessor: Using on-device Whisper model at $modelPath');

    final provider = OnDeviceWhisperProvider(
      modelPath: modelPath,
      language: config.language ?? 'en',
    );

    try {
      return await provider.transcribe(audioData, timestamp: timestamp);
    } finally {
      provider.dispose();
    }
  }

  Future<SttTranscriptionResult?> _transcribeWithCustomProvider(
    Uint8List audioData,
    CustomSttConfig config,
    DateTime timestamp,
  ) async {
    Logger.debug('LocalAudioProcessor: Using custom provider at ${config.effectiveUrl}');

    // Build the provider from the custom config
    final provider = SchemaBasedSttProvider(
      apiUrl: config.effectiveUrl,
      schema: config.schema,
      defaultHeaders: config.headers ?? {},
      defaultFields: config.params ?? {},
      audioFieldName: config.audioFieldName ?? 'file',
      requestBodyType: SttRequestBodyType.fromString(config.effectiveRequestType),
    );

    return await provider.transcribe(audioData, timestamp: timestamp);
  }
}
