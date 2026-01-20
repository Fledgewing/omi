import 'dart:typed_data';

/// Utility class for WAV file operations
class WavUtils {
  /// Create a WAV file from raw PCM16 data
  /// PCM16 = 16-bit signed integer samples
  static Uint8List createWavFromPcm16(
    Uint8List pcmData, {
    int sampleRate = 16000,
    int channels = 1,
    int bitsPerSample = 16,
  }) {
    final byteRate = sampleRate * channels * (bitsPerSample ~/ 8);
    final blockAlign = channels * (bitsPerSample ~/ 8);
    final dataSize = pcmData.length;
    final fileSize = 36 + dataSize;

    final buffer = BytesBuilder();

    // RIFF header
    buffer.add('RIFF'.codeUnits);
    buffer.add(_int32ToBytes(fileSize));
    buffer.add('WAVE'.codeUnits);

    // fmt chunk
    buffer.add('fmt '.codeUnits);
    buffer.add(_int32ToBytes(16)); // fmt chunk size
    buffer.add(_int16ToBytes(1)); // audio format (1 = PCM)
    buffer.add(_int16ToBytes(channels));
    buffer.add(_int32ToBytes(sampleRate));
    buffer.add(_int32ToBytes(byteRate));
    buffer.add(_int16ToBytes(blockAlign));
    buffer.add(_int16ToBytes(bitsPerSample));

    // data chunk
    buffer.add('data'.codeUnits);
    buffer.add(_int32ToBytes(dataSize));
    buffer.add(pcmData);

    return buffer.toBytes();
  }

  static Uint8List _int16ToBytes(int value) {
    return Uint8List.fromList([
      value & 0xff,
      (value >> 8) & 0xff,
    ]);
  }

  static Uint8List _int32ToBytes(int value) {
    return Uint8List.fromList([
      value & 0xff,
      (value >> 8) & 0xff,
      (value >> 16) & 0xff,
      (value >> 24) & 0xff,
    ]);
  }
}
