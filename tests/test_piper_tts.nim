## test_piper_tts.nim
## Test Text-to-Speech inference using Piper voice model
## Piper models convert phoneme IDs to raw audio samples

import unittest
import json
import strutils
import tables
import os
import strformat

import onnxruntime
import onnxruntime/ort_bindings

# Test paths
const TestDataDir = "tests/testdata/piper-voices"
const ModelPath = TestDataDir / "zh_CN-chaowen-medium.onnx"
const ConfigPath = TestDataDir / "zh_CN-chaowen-medium.onnx.json"

# Model configuration globals
var ConfigLoaded = false
var PhonemeIdMap: Table[string, seq[int64]]

type
  PiperConfig* = object
    ## Configuration for Piper TTS model
    sampleRate*: int
    numSpeakers*: int
    noiseScale*: float32
    lengthScale*: float32
    noiseW*: float32
    hopLength*: int
    phonemeIdMap*: Table[string, seq[int64]]
    speakerIdMap*: Table[string, int64]  ## Maps speaker names to IDs

proc loadConfig(path: string): PiperConfig =
  ## Load Piper model configuration from JSON file
  let jsonContent = readFile(path)
  let configJson = parseJson(jsonContent)

  result = PiperConfig(
    sampleRate: 22050,
    numSpeakers: 1,
    noiseScale: 0.667'f32,
    lengthScale: 1.0'f32,
    noiseW: 0.8'f32,
    hopLength: 256
  )

  if configJson.hasKey("audio") and configJson["audio"].hasKey("sample_rate"):
    result.sampleRate = configJson["audio"]["sample_rate"].getInt()

  if configJson.hasKey("num_speakers"):
    result.numSpeakers = configJson["num_speakers"].getInt()

  if configJson.hasKey("inference"):
    let inference = configJson["inference"]
    if inference.hasKey("noise_scale"):
      result.noiseScale = inference["noise_scale"].getFloat().float32
    if inference.hasKey("length_scale"):
      result.lengthScale = inference["length_scale"].getFloat().float32
    if inference.hasKey("noise_w"):
      result.noiseW = inference["noise_w"].getFloat().float32

  if configJson.hasKey("hop_length"):
    result.hopLength = configJson["hop_length"].getInt()

  # Load phoneme ID map
  if configJson.hasKey("phoneme_id_map"):
    for phoneme, ids in configJson["phoneme_id_map"]:
      var idSeq: seq[int64] = @[]
      for id in ids:
        idSeq.add(id.getInt().int64)
      result.phonemeIdMap[phoneme] = idSeq

  # Load speaker ID map
  if configJson.hasKey("speaker_id_map"):
    for speakerName, speakerId in configJson["speaker_id_map"]:
      result.speakerIdMap[speakerName] = speakerId.getInt().int64

  echo &"Piper config loaded:"
  echo &"  Sample rate: {result.sampleRate} Hz"
  echo &"  Num speakers: {result.numSpeakers}"
  echo &"  Hop length: {result.hopLength}"
  echo &"  Phonemes: {result.phonemeIdMap.len}"

# Static test phoneme sequence
# "你好我是中国人" - ni3 hao3 wo3 shi4 zhong1 guo2 ren2
const TestPhonemes = @[
  "^",                                      # Start token
  "q", "i", "a", "n", "1", "w", "a", "n", "4", " ",   # "qian wan" (千万)
  "b", "u", "2", " ",                       # "bu" (不)
  "y", "a", "o", "4", " ",                  # "yao" (要)
  "w", "a", "n", "g", "4", " ",             # "wang" (忘)
  "j", "i", "4", " ",                       # "ji" (记)
  "j", "i", "e", "1", " ",                  # "jie" (阶)
  "j", "i", "3", " ",                       # "ji" (级)
  "d", "o", "u", "4", " ",                  # "dou" (斗)
  "zh", "e", "n", "g", "1",                 # "zheng" (争)
  "$"                                       # End token
]

proc phonemesToIds(phonemes: seq[string], phonemeMap: Table[string, seq[int64]]): seq[int64] =
  ## Convert phoneme sequence to ID sequence
  result = @[]

  for phoneme in phonemes:
    if phonemeMap.hasKey(phoneme):
      for id in phonemeMap[phoneme]:
        result.add(id)
    else:
      # Use unknown token (usually "_")
      if phonemeMap.hasKey("_"):
        for id in phonemeMap["_"]:
          result.add(id)

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

proc getSpeakerId(config: PiperConfig, speakerName: string = ""): int64 =
  ## Get speaker ID by name. Returns 0 (default) for single-speaker models
  ## or when speaker name is not found.
  if config.numSpeakers <= 1:
    return 0'i64

  if speakerName.len > 0 and config.speakerIdMap.hasKey(speakerName):
    return config.speakerIdMap[speakerName]

  # Return first available speaker ID if no name specified
  if config.speakerIdMap.len > 0:
    for _, sid in config.speakerIdMap:
      return sid

  return 0'i64

proc runPiperInference(
  model: OnnxModel,
  phonemeIds: seq[int64],
  config: PiperConfig,
  speakerId: int = 0
): OnnxOutputTensor =
  ## Run inference on Piper TTS model
  ## Piper models typically expect:
  ##   - input: phoneme IDs [batch_size, seq_len]
  ##   - input_lengths: sequence lengths [batch_size]
  ##   - scales: [noise_scale, length_scale, noise_w] [3] or [batch_size, 3]
  ##   - sid: speaker ID (optional, for multi-speaker) [batch_size]

  let batchSize = 1'i64
  let seqLen = phonemeIds.len.int64

  var status: OrtStatusPtr

  # Create CPU memory info
  var memoryInfo: OrtMemoryInfo
  status = CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)
  checkStatus(status)

  # Prepare shapes as variables (needed for taking address)
  var inputShape = @[batchSize, seqLen]
  var lengthShape = @[batchSize]
  var scalesShape = @[3'i64]

  # Create input tensor (phoneme IDs)
  var inputOrtValue: OrtValue = nil
  let inputDataSize = phonemeIds.len * sizeof(int64)
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    phonemeIds[0].unsafeAddr,
    inputDataSize.csize_t,
    inputShape[0].unsafeAddr,
    inputShape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    inputOrtValue.addr
  )
  checkStatus(status)

  # Create input_lengths tensor
  var lengthData = @[seqLen]
  var lengthOrtValue: OrtValue = nil
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    lengthData[0].unsafeAddr,
    sizeof(int64).csize_t,
    lengthShape[0].unsafeAddr,
    lengthShape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    lengthOrtValue.addr
  )
  checkStatus(status)

  # Create scales tensor [noise_scale, length_scale, noise_w] as float32
  var scalesData = @[config.noiseScale, config.lengthScale, config.noiseW]
  var scalesOrtValue: OrtValue = nil
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    scalesData[0].unsafeAddr,
    (3 * sizeof(float32)).csize_t,
    scalesShape[0].unsafeAddr,
    scalesShape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    scalesOrtValue.addr
  )
  checkStatus(status)

  # Prepare input names and values
  # Typical Piper model inputs: input, input_lengths, scales, (optional) sid
  var inputNames: seq[cstring] = @["input".cstring, "input_lengths".cstring, "scales".cstring]
  var inputs: seq[OrtValue] = @[inputOrtValue, lengthOrtValue, scalesOrtValue]

  # Add speaker ID if multi-speaker model
  var sidOrtValue: OrtValue = nil
  if config.numSpeakers > 1:
    var sidData = @[speakerId.int64]
    status = CreateTensorWithDataAsOrtValue(
      memoryInfo,
      sidData[0].unsafeAddr,
      sizeof(int64).csize_t,
      lengthShape[0].unsafeAddr,  # Reuse batch size shape
      lengthShape.len.csize_t,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      sidOrtValue.addr
    )
    checkStatus(status)
    inputNames.add("sid".cstring)
    inputs.add(sidOrtValue)

  # Run inference
  let outputName = "output".cstring
  var outputOrtValue: OrtValue = nil
  status = Run(
    getSession(model),
    nil,  # run_options
    inputNames[0].addr,
    inputs[0].addr,
    inputs.len.csize_t,
    outputName.addr,
    1.csize_t,
    outputOrtValue.addr
  )
  checkStatus(status)

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

  # Copy data to Nim seq (must do this before releasing outputOrtValue)
  let floatPtr = cast[ptr UncheckedArray[float32]](outputDataPtr)
  var outputData = newSeq[float32](elemCount)
  for i in 0 ..< elemCount.int:
    outputData[i] = floatPtr[i]

  # Store result before cleanup
  result = OnnxOutputTensor(
    data: outputData,
    shape: outputShape
  )

  # Cleanup in reverse order of creation
  # Note: Don't release typeInfo or tensorInfo - they are owned by outputOrtValue
  # and releasing them causes double-free issues
  if outputOrtValue != nil:
    ReleaseValue(outputOrtValue)
  if sidOrtValue != nil:
    ReleaseValue(sidOrtValue)
  if scalesOrtValue != nil:
    ReleaseValue(scalesOrtValue)
  if lengthOrtValue != nil:
    ReleaseValue(lengthOrtValue)
  if inputOrtValue != nil:
    ReleaseValue(inputOrtValue)
  if memoryInfo != nil:
    ReleaseMemoryInfo(memoryInfo)

suite "Piper TTS Tests":
  test "Load Piper model configuration":
    echo "\n=== Piper TTS Configuration Test ==="

    if not fileExists(ConfigPath):
      echo "Config file not found, skipping test"
      skip()

    let config = loadConfig(ConfigPath)

    check config.sampleRate > 0
    check config.phonemeIdMap.len > 0

    echo &"Configuration loaded successfully!"
    echo &"  Found {config.phonemeIdMap.len} phoneme mappings"

    # Verify some expected phonemes exist
    check config.phonemeIdMap.hasKey("^")  # Start token
    check config.phonemeIdMap.hasKey("$")  # End token
    check config.phonemeIdMap.hasKey("_")  # Padding/unknown

    # For single-speaker models, speaker_id_map should be empty
    # For multi-speaker models, it would contain speaker name -> ID mappings
    if config.numSpeakers > 1:
      check config.speakerIdMap.len > 0

    ConfigLoaded = true

  test "Save generated audio to file":
    echo "\n=== Audio Save Test ==="

    if not fileExists(ModelPath):
      echo "Model not found, skipping test"
      skip()

    # Load configuration and model
    let config = loadConfig(ConfigPath)
    let model = newOnnxModel(ModelPath)

    # Use static test phonemes
    let phonemes = TestPhonemes
    let phonemeIds = phonemesToIds(phonemes, config.phonemeIdMap)
    let speakerId = getSpeakerId(config)
    let output = runPiperInference(model, phonemeIds, config, speakerId.int)

    # Save as WAV file (16-bit signed integers, mono)
    let outputPath = TestDataDir / "test_output.wav"
    var pcmData = newSeq[int16](output.data.len)

    for i in 0 ..< output.data.len:
      # Convert float32 [-1, 1] to int16 [-32768, 32767]
      let sample = output.data[i]
      let clamped = max(-1.0'f32, min(1.0'f32, sample))
      pcmData[i] = int16(clamped * 32767.0'f32)

    # Write WAV file with header
    var f = open(outputPath, fmWrite)

    # WAV header
    let numChannels = 1.uint16      # Mono
    let sampleRate = config.sampleRate.uint32
    let bitsPerSample = 16.uint16
    let byteRate = sampleRate * numChannels.uint32 * (bitsPerSample div 8).uint32
    let blockAlign = numChannels * (bitsPerSample div 8)
    let dataSize = (pcmData.len * sizeof(int16)).uint32
    let chunkSize = 36.uint32 + dataSize

    # Prepare all byte values
    let chunkSizeBytes = chunkSize.toBytes
    let subchunkSizeBytes = 16.uint32.toBytes
    let audioFormatBytes = 1.uint16.toBytes
    let numChannelsBytes = numChannels.toBytes
    let sampleRateBytes = sampleRate.toBytes
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
    echo &"File size: {pcmData.len * 2 + 44} bytes ({pcmData.len} samples + 44 byte header)"
    echo &"To play: ffplay {outputPath}  # or any media player"

    model.close()

    check fileExists(outputPath)

    # Note: Not cleaning up so you can test the audio:
    # ffplay tests/testdata/piper-voices/test_output.wav  # or any media player
    # To clean up manually: rm tests/testdata/piper-voices/test_output.wav
