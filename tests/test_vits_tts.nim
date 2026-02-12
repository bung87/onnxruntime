## test_vits_tts.nim
## Test Text-to-Speech inference using VITS Melo TTS model
## VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
## This model supports Chinese and English mixed text

import unittest
import strutils
import tables
import os
import strformat

import onnxruntime
import onnxruntime/ort_bindings

# Test paths
const TestDataDir = "tests/testdata/vits-melo-tts-zh_en"
const ModelPath = TestDataDir / "model.onnx"
const TokensPath = TestDataDir / "tokens.txt"
const LexiconPath = TestDataDir / "lexicon.txt"

# Token mapping
type
  VitsConfig* = object
    ## Configuration for VITS TTS model
    sampleRate*: int
    hopLength*: int
    sampleRateFloat*: float32  # For scaling

var
  TokenToId: Table[string, int64]
  IdToToken: Table[int64, string]
  TokensLoaded = false
  Lexicon: Table[string, seq[string]]  # word -> phonemes
  LexiconLoaded = false

proc loadTokens(path: string) =
  ## Load token mapping from tokens.txt
  ## Format: token id (space-separated, one per line)
  TokenToId.clear()
  IdToToken.clear()
  
  let content = readFile(path)
  for line in content.splitLines:
    let parts = line.strip().splitWhitespace()
    if parts.len >= 2:
      let token = parts[0]
      let id = parseInt(parts[1]).int64
      TokenToId[token] = id
      IdToToken[id] = token
  
  TokensLoaded = true
  echo &"Loaded {TokenToId.len} tokens"

proc loadLexicon(path: string) =
  ## Load lexicon from lexicon.txt
  ## Format: word phoneme1 phoneme2 ... (space-separated)
  Lexicon.clear()
  
  let content = readFile(path)
  for line in content.splitLines:
    let parts = line.strip().splitWhitespace()
    if parts.len >= 2:
      let word = parts[0].toLowerAscii()
      var phonemes: seq[string] = @[]
      for i in 1 ..< parts.len:
        phonemes.add(parts[i])
      Lexicon[word] = phonemes
  
  LexiconLoaded = true
  echo &"Loaded {Lexicon.len} lexicon entries"

proc textToTokens(text: string): seq[int64] =
  ## Convert text to token IDs using lexicon and token mapping
  ## This is a simplified version - full implementation would need proper G2P
  result = @[]
  
  if not TokensLoaded:
    echo "Warning: Tokens not loaded, returning empty sequence"
    return result
  
  # Add start token if available
  if TokenToId.hasKey("^"):
    result.add(TokenToId["^"])
  elif TokenToId.hasKey("<s>"):
    result.add(TokenToId["<s>"])
  
  # Simple character-by-character tokenization for now
  # In a full implementation, this would use the lexicon for proper G2P conversion
  for c in text:
    let token = $c
    if TokenToId.hasKey(token):
      result.add(TokenToId[token])
    elif TokenToId.hasKey(token.toLowerAscii()):
      result.add(TokenToId[token.toLowerAscii()])
    else:
      # Use unknown token if available
      if TokenToId.hasKey("<unk>"):
        result.add(TokenToId["<unk>"])
      elif TokenToId.hasKey("_"):
        result.add(TokenToId["_"])
  
  # Add end token if available
  if TokenToId.hasKey("$"):
    result.add(TokenToId["$"])
  elif TokenToId.hasKey("</s>"):
    result.add(TokenToId["</s>"])
  
  echo &"Converted text to {result.len} tokens"

proc runVitsInference(
  model: OnnxModel,
  tokens: seq[int64],
  speed: float32 = 1.0'f32
): OnnxOutputTensor =
  ## Run inference on VITS TTS model
  ## VITS models typically expect:
  ##   - tokens: token IDs [batch_size, seq_len]
  ##   - And may have additional inputs for speed control
  
  let batchSize = 1'i64
  let seqLen = tokens.len.int64
  
  var status: OrtStatusPtr
  
  # Create CPU memory info
  var memoryInfo: OrtMemoryInfo
  status = CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)
  checkStatus(status)
  
  # Prepare shapes as variables (needed for taking address)
  var inputShape = @[batchSize, seqLen]
  
  # Create input tensor (tokens)
  var inputOrtValue: OrtValue = nil
  let inputDataSize = tokens.len * sizeof(int64)
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    tokens[0].unsafeAddr,
    inputDataSize.csize_t,
    inputShape[0].unsafeAddr,
    inputShape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    inputOrtValue.addr
  )
  checkStatus(status)
  
  # Get input/output names by introspection
  var allocator: OrtAllocator
  status = GetAllocatorWithDefaultOptions(allocator.addr)
  checkStatus(status)
  
  var inputCount: csize_t
  status = SessionGetInputCount(getSession(model), inputCount.addr)
  checkStatus(status)
  
  var outputCount: csize_t
  status = SessionGetOutputCount(getSession(model), outputCount.addr)
  checkStatus(status)
  
  echo &"Model has {inputCount} inputs and {outputCount} outputs"
  
  # Get input names
  var inputNames: seq[cstring] = @[]
  for i in 0 ..< inputCount.int:
    var namePtr: cstring
    status = SessionGetInputName(getSession(model), i.csize_t, allocator, namePtr.addr)
    checkStatus(status)
    if namePtr != nil:
      inputNames.add(namePtr)
      echo &"  Input {i}: {namePtr}"
  
  # Get output names
  var outputNames: seq[cstring] = @[]
  for i in 0 ..< outputCount.int:
    var namePtr: cstring
    status = SessionGetOutputName(getSession(model), i.csize_t, allocator, namePtr.addr)
    checkStatus(status)
    if namePtr != nil:
      outputNames.add(namePtr)
      echo &"  Output {i}: {namePtr}"
  
  # Prepare inputs - for now just use the tokens input
  # VITS models typically have inputs like: x, x_lengths, noise_scale, length_scale, noise_scale_w
  var inputs: seq[OrtValue] = @[inputOrtValue]
  
  # Run inference
  var outputOrtValue: OrtValue = nil
  
  if outputNames.len > 0:
    status = Run(
      getSession(model),
      nil,  # run_options
      inputNames[0].addr,
      inputs[0].addr,
      inputs.len.csize_t,
      outputNames[0].addr,
      1.csize_t,  # Just get the first output (audio)
      outputOrtValue.addr
    )
    checkStatus(status)
  else:
    raise newException(Exception, "No output names found in model")
  
  # Get output info
  var typeInfo: OrtTypeInfo
  status = GetTypeInfo(outputOrtValue, typeInfo.addr)
  checkStatus(status)
  
  var tensorInfo: OrtTensorTypeAndShapeInfo
  status = CastTypeInfoToTensorInfo(typeInfo, tensorInfo.addr)
  checkStatus(status)
  
  # Get output shape
  var dimsCount: csize_t
  status = GetDimensionsCount(tensorInfo, dimsCount.addr)
  checkStatus(status)
  
  var outputShape = newSeq[int64](dimsCount)
  if dimsCount > 0:
    status = GetDimensions(tensorInfo, outputShape[0].addr, dimsCount)
    checkStatus(status)
  
  # Get output data
  var outputDataPtr: pointer
  status = GetTensorMutableData(outputOrtValue, outputDataPtr.addr)
  checkStatus(status)
  
  var elemCount: csize_t
  status = GetTensorShapeElementCount(tensorInfo, elemCount.addr)
  checkStatus(status)
  
  # Copy data to Nim seq
  let floatPtr = cast[ptr UncheckedArray[float32]](outputDataPtr)
  var outputData = newSeq[float32](elemCount)
  for i in 0 ..< elemCount.int:
    outputData[i] = floatPtr[i]
  
  # Store result before cleanup
  result = OnnxOutputTensor(
    data: outputData,
    shape: outputShape
  )
  
  # Cleanup
  if outputOrtValue != nil:
    ReleaseValue(outputOrtValue)
  if inputOrtValue != nil:
    ReleaseValue(inputOrtValue)
  ReleaseMemoryInfo(memoryInfo)

# Helper functions for WAV header (little-endian byte conversion)
proc toBytes(x: uint16): array[2, char] =
  ## Convert uint16 to little-endian bytes
  result[0] = char(x and 0xFF)
  result[1] = char((x shr 8) and 0xFF)

proc toBytes(x: uint32): array[4, char] =
  ## Convert uint32 to little-endian bytes
  result[0] = char(x and 0xFF)
  result[1] = char((x shr 8) and 0xFF)
  result[2] = char((x shr 16) and 0xFF)
  result[3] = char((x shr 24) and 0xFF)

proc saveWavAudio(
  audioData: seq[float32],
  outputPath: string,
  sampleRate: int = 44100
) =
  ## Save audio data as a WAV file (16-bit signed integers, mono)
  var pcmData = newSeq[int16](audioData.len)
  
  for i in 0 ..< audioData.len:
    # Convert float32 [-1, 1] to int16 [-32768, 32767]
    let sample = audioData[i]
    let clamped = max(-1.0'f32, min(1.0'f32, sample))
    pcmData[i] = int16(clamped * 32767.0'f32)
  
  # Write WAV file with header
  var f = open(outputPath, fmWrite)
  
  # WAV header
  let numChannels = 1.uint16      # Mono
  let sampleRateU32 = sampleRate.uint32
  let bitsPerSample = 16.uint16
  let byteRate = sampleRateU32 * numChannels.uint32 * (bitsPerSample div 8).uint32
  let blockAlign = numChannels * (bitsPerSample div 8)
  let dataSize = (pcmData.len * sizeof(int16)).uint32
  let chunkSize = 36.uint32 + dataSize
  
  # Prepare all byte values
  let chunkSizeBytes = chunkSize.toBytes
  let subchunkSizeBytes = 16.uint32.toBytes
  let audioFormatBytes = 1.uint16.toBytes
  let numChannelsBytes = numChannels.toBytes
  let sampleRateBytes = sampleRateU32.toBytes
  let byteRateBytes = byteRate.toBytes
  let blockAlignBytes = blockAlign.toBytes
  let bitsPerSampleBytes = bitsPerSample.toBytes
  let dataSizeBytes = dataSize.toBytes
  
  # RIFF chunk
  f.write("RIFF")
  discard f.writeBuffer(chunkSizeBytes[0].addr, 4)
  f.write("WAVE")
  
  # fmt subchunk
  f.write("fmt ")
  discard f.writeBuffer(subchunkSizeBytes[0].addr, 4)
  discard f.writeBuffer(audioFormatBytes[0].addr, 2)
  discard f.writeBuffer(numChannelsBytes[0].addr, 2)
  discard f.writeBuffer(sampleRateBytes[0].addr, 4)
  discard f.writeBuffer(byteRateBytes[0].addr, 4)
  discard f.writeBuffer(blockAlignBytes[0].addr, 2)
  discard f.writeBuffer(bitsPerSampleBytes[0].addr, 2)
  
  # data subchunk
  f.write("data")
  discard f.writeBuffer(dataSizeBytes[0].addr, 4)
  
  # Write PCM data
  for sample in pcmData:
    discard f.writeBuffer(sample.addr, sizeof(int16))
  f.close()
  
  echo &"Saved WAV audio to: {outputPath}"
  echo &"  Sample rate: {sampleRate} Hz"
  echo &"  Duration: {audioData.len.float / sampleRate.float:.2f} seconds"
  echo &"  Samples: {audioData.len}"

suite "VITS Melo TTS Tests":
  test "Load VITS model and tokens":
    echo "\n=== VITS Melo TTS Model Loading Test ==="
    
    if not fileExists(ModelPath):
      echo "Model not found at: " & ModelPath
      echo "Please run: nimble download_vits_melo"
      skip()
    
    if not fileExists(TokensPath):
      echo "Tokens file not found at: " & TokensPath
      echo "Please run: nimble download_vits_melo"
      skip()
    
    # Load tokens
    loadTokens(TokensPath)
    check TokensLoaded
    check TokenToId.len > 0
    
    # Load the model
    echo "Loading VITS model..."
    let model = newOnnxModel(ModelPath)
    echo "Model loaded successfully!"
    
    # Verify model can be introspected
    var allocator: OrtAllocator
    var status = GetAllocatorWithDefaultOptions(allocator.addr)
    checkStatus(status)
    
    var inputCount: csize_t
    status = SessionGetInputCount(getSession(model), inputCount.addr)
    checkStatus(status)
    check inputCount > 0
    
    echo &"Model has {inputCount} inputs"
    
    model.close()
    echo "=== Model loading test complete ===\n"
  
  test "Generate audio from tokens":
    echo "\n=== VITS Audio Generation Test ==="
    
    if not fileExists(ModelPath):
      echo "Model not found, skipping test"
      skip()
    
    if not fileExists(TokensPath):
      echo "Tokens file not found, skipping test"
      skip()
    
    # Load tokens and model
    loadTokens(TokensPath)
    
    if LexiconPath.fileExists:
      loadLexicon(LexiconPath)
    
    let model = newOnnxModel(ModelPath)
    
    # Test text - simple Chinese phrase
    let testText = "你好世界"
    let tokens = textToTokens(testText)
    
    if tokens.len == 0:
      echo "No tokens generated, skipping inference"
      model.close()
      skip()
    
    echo &"Test text: '{testText}'"
    echo &"Tokens: {tokens}"
    
    # Run inference
    echo "Running inference..."
    let output = runVitsInference(model, tokens)
    
    echo &"Output shape: {output.shape}"
    echo &"Output samples: {output.data.len}"
    
    check output.data.len > 0
    
    # Save audio to file
    let outputPath = TestDataDir / "test_output.wav"
    saveWavAudio(output.data, outputPath, sampleRate = 44100)
    
    model.close()
    
    check fileExists(outputPath)
    
    echo "=== Audio generation test complete ===\n"
    echo &"To play: ffplay {outputPath}"
