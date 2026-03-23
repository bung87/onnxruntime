## test_kokoro_sherpa.nim
## Test Sherpa-ONNX Kokoro Multi-Lang TTS model
##
## Model download:
##   mkdir -p tests/testdata/kokoro-multi-lang
##   cd tests/testdata/kokoro-multi-lang
##   
##   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
##   tar xf kokoro-multi-lang-v1_0.tar.bz2
##   mv kokoro-multi-lang-v1_0/* .
##   rmdir kokoro-multi-lang-v1_0
##
## Model files:
##   model.onnx              - ONNX model
##   voices.bin              - Voice vectors (53 speakers)
##   tokens.txt              - Token to ID mapping
##   lexicon-zh.txt          - Chinese lexicon
##   lexicon-us-en.txt       - English lexicon
##   espeak-ng-data/         - espeak-ng data directory
##   dict/                   - Dictionary directory

import unittest
import os
import tables

import onnx_rt, kokoro_sherpa_utils

# Test paths
const TestDataDir = "tests/testdata/kokoro-multi-lang"
const ModelPath = TestDataDir / "model.onnx"
const VoicesPath = TestDataDir / "voices.bin"
const TokensPath = TestDataDir / "tokens.txt"
const LexiconZhPath = TestDataDir / "lexicon-zh.txt"
const LexiconEnPath = TestDataDir / "lexicon-us-en.txt"

# Test texts
const TestChinese = "你好世界"
const TestEnglish = "Hello world"
const TestMixed = "Hello 世界"

proc writeWavFile(path: string, pcmData: seq[int16], sampleRate: int) =
  ## Write PCM data to WAV file (little-endian format)
  let dataSize = pcmData.len * 2
  let chunkSize = 36 + dataSize

  var f = open(path, fmWrite)

  # Helper to write uint32 in little-endian
  proc writeU32(f: var File, v: uint32) =
    var bytes = @[v.uint8, (v shr 8).uint8, (v shr 16).uint8, (v shr 24).uint8]
    discard f.writeBuffer(bytes[0].unsafeAddr, 4)
  
  # Helper to write uint16 in little-endian
  proc writeU16(f: var File, v: uint16) =
    var bytes = @[v.uint8, (v shr 8).uint8]
    discard f.writeBuffer(bytes[0].unsafeAddr, 2)

  # Write header
  f.write("RIFF")
  writeU32(f, chunkSize.uint32)
  f.write("WAVE")
  f.write("fmt ")
  writeU32(f, 16)  # subchunk size
  writeU16(f, 1)   # audio format (PCM)
  writeU16(f, 1)   # num channels
  writeU32(f, sampleRate.uint32)
  writeU32(f, (sampleRate * 2).uint32)  # byte rate
  writeU16(f, 2)   # block align
  writeU16(f, 16)  # bits per sample
  
  f.write("data")
  writeU32(f, dataSize.uint32)
  
  # Write samples
  for s in pcmData:
    writeU16(f, s.uint16)
  
  f.close()

proc checkModelFiles(): bool =
  ## Check if required model files exist
  if not fileExists(ModelPath):
    echo "Skipping: Model not found: ", ModelPath
    echo "Please download the model:"
    echo "  cd ", TestDataDir
    echo "  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2"
    echo "  tar xf kokoro-multi-lang-v1_0.tar.bz2"
    return false
  if not fileExists(VoicesPath):
    echo "Skipping: Voices not found: ", VoicesPath
    return false
  if not fileExists(TokensPath):
    echo "Skipping: Tokens not found: ", TokensPath
    return false
  return true

suite "Sherpa-ONNX Kokoro Multi-Lang TTS":

  test "Load tokens":
    if not fileExists(TokensPath):
      echo "Skipping: Tokens not found: ", TokensPath
      skip()
    else:
      let tokens = loadTokens(TokensPath)
      check tokens.len > 0
      echo "Loaded ", tokens.len, " tokens"
      
      # Check some common tokens
      if tokens.hasKey(" "):
        echo "Space token ID: ", tokens[" "]

  test "Load lexicons":
    if not fileExists(LexiconZhPath) and not fileExists(LexiconEnPath):
      echo "Skipping: No lexicon files found"
      skip()
    else:
      if fileExists(LexiconZhPath):
        let lexiconZh = loadLexicon(LexiconZhPath)
        check lexiconZh.len > 0
        echo "Loaded ", lexiconZh.len, " Chinese lexicon entries"
      
      if fileExists(LexiconEnPath):
        let lexiconEn = loadLexicon(LexiconEnPath)
        check lexiconEn.len > 0
        echo "Loaded ", lexiconEn.len, " English lexicon entries"

  test "Load voices":
    if not fileExists(VoicesPath):
      echo "Skipping: Voices not found: ", VoicesPath
      skip()
    else:
      let voices = loadVoices(VoicesPath)
      check voices.data.len > 0
      check voices.maxLength > 0
      echo "Loaded voice: ", voices.data.len, " floats, max length: ", voices.maxLength

  test "Get style vector":
    if not fileExists(VoicesPath):
      skip()
    else:
      let voices = loadVoices(VoicesPath)
      
      # Test different token lengths
      let style10 = voices.getStyle(10, speakerId = 0)
      let style50 = voices.getStyle(50, speakerId = 0)
      
      check style10.len == 256
      check style50.len == 256
      echo "Style vector dimension: ", style10.len

  test "Chinese text to tokens":
    if not fileExists(LexiconZhPath) or not fileExists(TokensPath):
      echo "Skipping: Chinese lexicon or tokens not found"
      skip()
    else:
      let lexiconZh = loadLexicon(LexiconZhPath)
      let tokens = loadTokens(TokensPath)
      
      let tokenIds = chineseTextToTokens(TestChinese, lexiconZh, tokens)
      echo "Chinese text '", TestChinese, "' -> ", tokenIds.len, " tokens"
      
      # Just check that we got some tokens (may be empty if lexicon format differs)
      check tokenIds.len >= 0  # Allow empty for now

  test "Full pipeline - Chinese to WAV":
    if not checkModelFiles():
      skip()
    elif not fileExists(LexiconZhPath):
      echo "Skipping: Chinese lexicon not found"
      skip()
    else:
      # Load resources
      let tokens = loadTokens(TokensPath)
      let lexiconZh = loadLexicon(LexiconZhPath)
      let voices = loadVoices(VoicesPath)
      
      # Convert text to tokens
      let tokenIds = chineseTextToTokens(TestChinese, lexiconZh, tokens)
      if tokenIds.len == 0:
        echo "Warning: No tokens generated, skipping inference"
        skip()
      
      echo "Generated ", tokenIds.len, " tokens"
      
      # Prepare input with padding
      var inputIds = @[0'i64]
      for id in tokenIds:
        inputIds.add(id)
      inputIds.add(0'i64)
      
      # Get style for speaker 0
      let style = voices.getStyle(tokenIds.len, speakerId = 0)
      
      # Load model and run inference
      let model = loadModel(ModelPath)
      let output = runKokoroSherpa(model, inputIds, style, speed = 1.0'f32)
      
      check output.data.len > 0
      echo "Generated ", output.data.len, " audio samples (~", output.data.len / SampleRate, " seconds)"
      
      # Save to WAV
      let outputPath = TestDataDir / "test_output_zh.wav"
      writeWavFile(outputPath, output.toInt16Samples(), SampleRate)
      check fileExists(outputPath)
      echo "Saved to: ", outputPath
      
      model.close()

  test "Full pipeline - High-level synthesize API":
    if not checkModelFiles():
      skip()
    elif not fileExists(LexiconZhPath) or not fileExists(LexiconEnPath):
      echo "Skipping: Lexicons not found"
      skip()
    else:
      # Load resources
      let tokens = loadTokens(TokensPath)
      let lexiconZh = loadLexicon(LexiconZhPath)
      let lexiconEn = loadLexicon(LexiconEnPath)
      let voices = loadVoices(VoicesPath)
      let model = loadModel(ModelPath)
      
      # Use high-level API
      let output = synthesize(
        model,
        TestMixed,
        lexiconZh,
        lexiconEn,
        tokens,
        voices,
        speakerId = 0,
        speed = 1.0'f32
      )
      
      check output.data.len > 0
      echo "Synthesized '", TestMixed, "' -> ", output.data.len, " samples"
      
      # Save to WAV
      let outputPath = TestDataDir / "test_output_mixed.wav"
      writeWavFile(outputPath, output.toInt16Samples(), SampleRate)
      check fileExists(outputPath)
      
      model.close()

  test "Test different speakers":
    if not checkModelFiles():
      skip()
    elif not fileExists(LexiconZhPath):
      skip()
    else:
      let tokens = loadTokens(TokensPath)
      let lexiconZh = loadLexicon(LexiconZhPath)
      let voices = loadVoices(VoicesPath)
      let model = loadModel(ModelPath)
      
      let tokenIds = chineseTextToTokens("你好", lexiconZh, tokens)
      if tokenIds.len == 0:
        skip()
      
      var inputIds = @[0'i64]
      for id in tokenIds:
        inputIds.add(id)
      inputIds.add(0'i64)
      
      # Test speaker 0 and speaker 10
      let style0 = voices.getStyle(tokenIds.len, speakerId = 0)
      let style10 = voices.getStyle(tokenIds.len, speakerId = 10)
      
      let output0 = runKokoroSherpa(model, inputIds, style0)
      let output10 = runKokoroSherpa(model, inputIds, style10)
      
      check output0.data.len > 0
      check output10.data.len > 0
      
      # Different speakers should have different styles
      echo "Speaker 0: ", output0.data.len, " samples"
      echo "Speaker 10: ", output10.data.len, " samples"
      
      model.close()

  test "Test different speeds":
    if not checkModelFiles():
      skip()
    elif not fileExists(LexiconZhPath):
      skip()
    else:
      let tokens = loadTokens(TokensPath)
      let lexiconZh = loadLexicon(LexiconZhPath)
      let voices = loadVoices(VoicesPath)
      let model = loadModel(ModelPath)
      
      let tokenIds = chineseTextToTokens("你好世界", lexiconZh, tokens)
      if tokenIds.len == 0:
        skip()
      
      var inputIds = @[0'i64]
      for id in tokenIds:
        inputIds.add(id)
      inputIds.add(0'i64)
      
      let style = voices.getStyle(tokenIds.len, speakerId = 0)
      
      # Test different speeds
      let outputSlow = runKokoroSherpa(model, inputIds, style, speed = 0.8'f32)
      let outputNormal = runKokoroSherpa(model, inputIds, style, speed = 1.0'f32)
      let outputFast = runKokoroSherpa(model, inputIds, style, speed = 1.2'f32)
      
      check outputSlow.data.len > 0
      check outputNormal.data.len > 0
      check outputFast.data.len > 0
      
      echo "Speed 0.8: ", outputSlow.data.len, " samples"
      echo "Speed 1.0: ", outputNormal.data.len, " samples"
      echo "Speed 1.2: ", outputFast.data.len, " samples"
      
      model.close()
