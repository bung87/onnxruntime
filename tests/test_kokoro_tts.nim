## test_kokoro_tts.nim
## Test Text-to-Speech inference using Kokoro-82M ONNX model
##
## Model download:
##   mkdir -p tests/testdata/kokoro-82m
##   cd tests/testdata/kokoro-82m
##   
##   # Download ONNX model (choose one):
##   # FP32 model (326MB)
##   curl -L "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx" -o model.onnx
##   
##   # Or quantized model (86MB, faster)
##   curl -L "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_quantized.onnx" -o model.onnx
##   
##   # Download voices
##   curl -L "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af.bin" -o voices/af.bin
##
## Note: This test uses pre-computed phoneme sequences for testing without
## external phonemizer dependency. In production, use misaki or espeak-ng
## to convert text to phonemes.

import unittest
import os
import tables

import onnx_rt, kokoro_utils

# Test paths
const TestDataDir = "tests/testdata/kokoro-82m"
const ModelPath = TestDataDir / "model.onnx"
const VoicesDir = TestDataDir / "voices"
const VoicePath = VoicesDir / "af.bin"

# Sample phoneme sequences for testing
# "Hello world" in ARPAbet-like notation
const TestPhonemesHello = "h ə ˈ l o ʊ  ˈ w ɜː r l d"

# Pre-computed token IDs for "Hello world" (for testing without phonemizer)
# This is a simplified example - real usage requires proper phoneme conversion
const TestTokenIds = @[50'i64, 157'i64, 43'i64, 135'i64, 16'i64, 53'i64, 135'i64, 46'i64]

proc writeWavFile(path: string, pcmData: seq[int16], sampleRate: int) =
  ## Write PCM data to WAV file
  let numChannels = 1.uint16
  let bitsPerSample = 16.uint16
  let byteRate = sampleRate.uint32 * numChannels.uint32 * (bitsPerSample div 8).uint32
  let blockAlign = numChannels * (bitsPerSample div 8)
  let dataSize = (pcmData.len * sizeof(int16)).uint32
  let chunkSize = 36.uint32 + dataSize

  var f = open(path, fmWrite)
  defer: f.close()

  # RIFF header
  f.write("RIFF")
  discard f.writeBuffer(chunkSize.addr, 4)
  f.write("WAVE")

  # fmt subchunk
  f.write("fmt ")
  let subchunkSize = 16.uint32
  discard f.writeBuffer(subchunkSize.addr, 4)
  let audioFormat = 1.uint16
  discard f.writeBuffer(audioFormat.addr, 2)
  discard f.writeBuffer(numChannels.addr, 2)
  let sr = sampleRate.uint32
  discard f.writeBuffer(sr.addr, 4)
  discard f.writeBuffer(byteRate.addr, 4)
  discard f.writeBuffer(blockAlign.addr, 2)
  discard f.writeBuffer(bitsPerSample.addr, 2)

  # data subchunk
  f.write("data")
  discard f.writeBuffer(dataSize.addr, 4)

  # Write PCM data
  for sample in pcmData:
    discard f.writeBuffer(sample.addr, sizeof(int16))

proc checkModelFiles(): bool =
  ## Check if required model files exist
  if not fileExists(ModelPath):
    echo "Skipping: Model not found: ", ModelPath
    echo "Please download the model:"
    echo "  mkdir -p ", TestDataDir
    echo "  curl -L \"https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx\" -o ", ModelPath
    return false
  if not fileExists(VoicePath):
    echo "Skipping: Voice file not found: ", VoicePath
    echo "Please download voices:"
    echo "  mkdir -p ", VoicesDir
    echo "  curl -L \"https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af.bin\" -o ", VoicePath
    return false
  return true

suite "Kokoro-82M TTS":

  test "Load voice data":
    if not fileExists(VoicePath):
      echo "Skipping: Voice file not found: ", VoicePath
      echo "Please download voices:"
      echo "  mkdir -p ", VoicesDir
      echo "  curl -L \"https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af.bin\" -o ", VoicePath
      skip()
    else:
      let voices = loadVoices(VoicePath)
      check voices.data.len > 0
      check voices.maxLength > 0
      
      echo "Loaded voice: ", voices.data.len, " floats, max length: ", voices.maxLength

  test "Get style vector for token length":
    if not fileExists(VoicePath):
      echo "Skipping: Voice file not found: ", VoicePath
      skip()
    else:
      let voices = loadVoices(VoicePath)
      
      # Test getting style for different token lengths
      let style10 = voices.getStyle(10)
      let style50 = voices.getStyle(50)
      
      check style10.len == 256
      check style50.len == 256
      
      # Different lengths should give different styles
      var different = false
      for i in 0 ..< 256:
        if style10[i] != style50[i]:
          different = true
          break
      check different == true
      
      echo "Style vector dimension: ", style10.len

  test "Convert phonemes to token IDs":
    # Test basic phoneme conversion
    let tokenIds = phonemesToIds("h ə ˈ l o ʊ")
    check tokenIds.len > 0
    
    # Check that known phonemes are converted
    if KokoroVocab.hasKey("h"):
      check tokenIds[0] == KokoroVocab["h"]
    
    echo "Phoneme conversion: input length 6, output length ", tokenIds.len

  test "Prepare input IDs with padding":
    let rawIds = @[1'i64, 2'i64, 3'i64]
    let paddedIds = prepareInputIds(rawIds)
    
    check paddedIds.len == 5  # 1 (pad) + 3 (data) + 1 (pad)
    check paddedIds[0] == 0   # Start pad
    check paddedIds[1] == 1   # Original data
    check paddedIds[2] == 2
    check paddedIds[3] == 3
    check paddedIds[4] == 0   # End pad

  test "Full pipeline - token IDs to WAV file":
    if not checkModelFiles():
      skip()
    else:
      # Load voice
      let voices = loadVoices(VoicePath)
      
      # Use pre-computed token IDs (in real usage, convert text -> phonemes -> IDs)
      let rawTokenIds = TestTokenIds
      check rawTokenIds.len > 0
      
      # Prepare input with padding
      let inputIds = prepareInputIds(rawTokenIds)
      check inputIds.len == rawTokenIds.len + 2  # Added padding
      
      # Get style vector for this token length
      let style = voices.getStyle(rawTokenIds.len)
      check style.len == 256
      
      # Load model and run inference
      let model = loadModel(ModelPath)
      
      let output = runKokoroTTS(
        model,
        inputIds,
        style,
        speed = 1.0'f32
      )
      
      check output.data.len > 0
      echo "Generated ", output.data.len, " audio samples (~", output.data.len / SampleRate, " seconds)"
      
      # Convert to int16
      let samples = output.toInt16Samples()
      
      # Save to WAV file (24kHz)
      let outputPath = TestDataDir / "test_output.wav"
      writeWavFile(outputPath, samples, SampleRate)
      check fileExists(outputPath)
      echo "Saved output to: ", outputPath
      
      model.close()

  test "Test with different speech speeds":
    if not checkModelFiles():
      skip()
    else:
      let voices = loadVoices(VoicePath)
      let inputIds = prepareInputIds(TestTokenIds)
      let style = voices.getStyle(TestTokenIds.len)
      
      let model = loadModel(ModelPath)
      
      # Test normal speed
      let outputNormal = runKokoroTTS(model, inputIds, style, speed = 1.0'f32)
      
      # Test slower speed
      let outputSlow = runKokoroTTS(model, inputIds, style, speed = 0.8'f32)
      
      # Test faster speed  
      let outputFast = runKokoroTTS(model, inputIds, style, speed = 1.2'f32)
      
      check outputNormal.data.len > 0
      check outputSlow.data.len > 0
      check outputFast.data.len > 0
      
      # Slower speed should produce more samples, faster should produce fewer
      # Note: This depends on the model's implementation
      echo "Normal speed (1.0): ", outputNormal.data.len, " samples"
      echo "Slow speed (0.8):   ", outputSlow.data.len, " samples"
      echo "Fast speed (1.2):   ", outputFast.data.len, " samples"
      
      model.close()

  test "Test vocabulary coverage":
    # Test that our vocab matches expected phonemes
    check KokoroVocab.hasKey(" ") == true
    check KokoroVocab.hasKey("a") == true
    check KokoroVocab.hasKey("ˈ") == true  # Stress marker
    check KokoroVocab.hasKey(".") == true  # Period
    check KokoroVocab.hasKey(",") == true  # Comma
    
    # Test some common English phonemes
    check KokoroVocab["h"] == 50'i64
    check KokoroVocab["ə"] == 83'i64
    check KokoroVocab["l"] == 54'i64
    check KokoroVocab["o"] == 57'i64
    check KokoroVocab["ʊ"] == 135'i64
    
    echo "Vocab size: ", KokoroVocab.len
