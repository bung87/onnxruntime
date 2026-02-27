## test_whisper_asr_merged_fp16.nim
## Test Whisper merged FP16 decoder
##
## NOTE: KV-cache mode (use_cache=true) produces incorrect results with this FP16 model.
## The reliable approach is to always use use_cache=false and pass the full sequence.

import std/[unittest, os, json, strutils]
import onnxruntime
import onnxruntime/ort_bindings
import whisper_utils

const TestDataDir = currentSourcePath().parentDir / "testdata" / "whisper-large-v3-zh"
const ConfigDataDir = TestDataDir / "onnx-community" / "whisper-large-v3-chinese-ONNX"
const ModelDataDir = ConfigDataDir / "onnx"
const EncoderPath = ModelDataDir / "encoder_model.onnx"
const DecoderPath = ModelDataDir / "decoder_model_merged_fp16.onnx"
const TokenizerPath = ConfigDataDir / "tokenizer.json"
const VocabPath = ConfigDataDir / "vocab.json"
const TestAudioPath = TestDataDir / "test_input.wav"
const GenerationConfigPath = ConfigDataDir / "generation_config.json"

const VOCAB_SIZE = 51865
const ENCODER_SEQ_LEN = 1500
const NUM_LAYERS = 4
const NUM_HEADS = 6
const HEAD_DIM = 64

type
  GenerationConfig = object
    decoder_start_token_id: int
    eos_token_id: int
    begin_suppress_tokens: seq[int64]
  AddedToken = object
    id: int
    content: string
  TokenizerConfig = object
    added_tokens: seq[AddedToken]
  WhisperConfig = object
    startToken, endToken, langToken, taskToken, noTimestampsToken: int
    suppressTokens: seq[int64]

proc loadConfig(): WhisperConfig =
  let genConfig = if fileExists(GenerationConfigPath):
    readFile(GenerationConfigPath).parseJson().to(GenerationConfig)
  else:
    GenerationConfig(decoder_start_token_id: 50258, eos_token_id: 50257, begin_suppress_tokens: @[220'i64, 50257'i64])
  let tokConfig = if fileExists(TokenizerPath):
    readFile(TokenizerPath).parseJson().to(TokenizerConfig)
  else:
    TokenizerConfig(added_tokens: @[])
  result.startToken = genConfig.decoder_start_token_id
  result.endToken = genConfig.eos_token_id
  result.suppressTokens = genConfig.begin_suppress_tokens
  for token in tokConfig.added_tokens:
    case token.content:
      of "<|startoftranscript|>": result.startToken = token.id
      of "<|endoftext|>": result.endToken = token.id
      of "<|zh|>": result.langToken = token.id
      of "<|transcribe|>": result.taskToken = token.id
      of "<|notimestamps|>": result.noTimestampsToken = token.id
      else: discard

proc loadVocab(): seq[string] =
  result = newSeq[string](VOCAB_SIZE)
  if fileExists(VocabPath):
    for token, id in readFile(VocabPath).parseJson().pairs:
      let idx = id.getInt
      if idx < result.len: result[idx] = token
  if fileExists(TokenizerPath):
    let tokConfig = readFile(TokenizerPath).parseJson().to(TokenizerConfig)
    for token in tokConfig.added_tokens:
      if token.id < result.len: result[token.id] = token.content

type
  MergedWhisperModel = object
    encoder: Model
    decoder: Model
    inputNames, outputNames: seq[string]

proc loadMergedWhisper(encoderPath, decoderPath: string): MergedWhisperModel =
  result.encoder = loadModel(encoderPath)
  result.decoder = loadModel(decoderPath)

  var allocator: OrtAllocator
  checkStatus GetAllocatorWithDefaultOptions(allocator.addr)

  var inputCount, outputCount: csize_t
  checkStatus SessionGetInputCount(getSession(result.decoder.internal), inputCount.addr)
  checkStatus SessionGetOutputCount(getSession(result.decoder.internal), outputCount.addr)

  result.inputNames = newSeq[string](inputCount.int)
  for i in 0 ..< inputCount.int:
    var namePtr: cstring
    checkStatus SessionGetInputName(getSession(result.decoder.internal), i.csize_t, allocator, namePtr.addr)
    result.inputNames[i] = $namePtr

  result.outputNames = newSeq[string](outputCount.int)
  for i in 0 ..< outputCount.int:
    var namePtr: cstring
    checkStatus SessionGetOutputName(getSession(result.decoder.internal), i.csize_t, allocator, namePtr.addr)
    result.outputNames[i] = $namePtr

proc close(whisper: var MergedWhisperModel) =
  whisper.encoder.close()
  whisper.decoder.close()

proc runEncoder(whisper: MergedWhisperModel, melSpectrogram: seq[float32]): OrtValue =
  var memoryInfo: OrtMemoryInfo
  checkStatus CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)

  var inputShape = @[1'i64, 80'i64, 3000'i64]
  var encoderInput: OrtValue
  checkStatus CreateTensorWithDataAsOrtValue(
    memoryInfo, melSpectrogram[0].unsafeAddr, (melSpectrogram.len * sizeof(float32)).csize_t,
    inputShape[0].unsafeAddr, inputShape.len.csize_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, encoderInput.addr
  )

  var allocator: OrtAllocator
  checkStatus GetAllocatorWithDefaultOptions(allocator.addr)

  var encInputNamePtr, encOutputNamePtr: cstring
  checkStatus SessionGetInputName(getSession(whisper.encoder.internal), 0, allocator, encInputNamePtr.addr)
  checkStatus SessionGetOutputName(getSession(whisper.encoder.internal), 0, allocator, encOutputNamePtr.addr)

  var encInputNames = @[($encInputNamePtr).cstring]
  var encOutputNames = @[($encOutputNamePtr).cstring]

  var encoderOutput: OrtValue
  checkStatus Run(
    getSession(whisper.encoder.internal), nil, encInputNames[0].addr, encoderInput.addr, 1,
    encOutputNames[0].addr, 1, encoderOutput.addr
  )

  ReleaseValue(encoderInput)
  result = encoderOutput

proc runDecoderStep(
  whisper: var MergedWhisperModel,
  inputIds: seq[int64],
  encoderOutput: OrtValue
): int64 =
  ## Run decoder step. Always uses use_cache=false for correct results.
  var status: OrtStatusPtr

  var memoryInfo: OrtMemoryInfo
  checkStatus CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)

  # Create input_ids tensor
  var inputIdsShape = @[1'i64, inputIds.len.int64]
  var inputIdsValue: OrtValue
  checkStatus CreateTensorWithDataAsOrtValue(
    memoryInfo, inputIds[0].unsafeAddr, (inputIds.len * sizeof(int64)).csize_t,
    inputIdsShape[0].unsafeAddr, inputIdsShape.len.csize_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, inputIdsValue.addr
  )

  # Create cache_position tensor (position of last token)
  var cachePosData = @[inputIds.len.int64 - 1]
  var cachePosShape = @[1'i64]
  var cachePosValue: OrtValue
  checkStatus CreateTensorWithDataAsOrtValue(
    memoryInfo, cachePosData[0].unsafeAddr, (cachePosData.len * sizeof(int64)).csize_t,
    cachePosShape[0].unsafeAddr, cachePosShape.len.csize_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, cachePosValue.addr
  )

  # Create use_cache_branch tensor (always false for correct results)
  var useCacheBool = false
  var useCacheShape = @[1'i64]
  var useCacheValue: OrtValue
  checkStatus CreateTensorWithDataAsOrtValue(
    memoryInfo, useCacheBool.unsafeAddr, sizeof(bool).csize_t,
    useCacheShape[0].unsafeAddr, useCacheShape.len.csize_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, useCacheValue.addr
  )

  # Build inputs (20 tensors)
  var inputValues = newSeq[OrtValue](whisper.inputNames.len)
  var inputNamePtrs = newSeq[cstring](whisper.inputNames.len)
  var createdTensors: seq[int] = @[]

  for i, name in whisper.inputNames:
    inputNamePtrs[i] = name.cstring
    if name == "input_ids":
      inputValues[i] = inputIdsValue
    elif name.contains("encoder_hidden_states"):
      inputValues[i] = encoderOutput
    elif name.contains("cache_position"):
      inputValues[i] = cachePosValue
    elif name.contains("use_cache_branch"):
      inputValues[i] = useCacheValue
    else:
      # All past_key_values.* inputs - create zero-filled tensors
      var isEncoder = name.contains(".encoder.")
      var seqLen = if isEncoder: ENCODER_SEQ_LEN else: 0
      var shape = @[1'i64, NUM_HEADS.int64, seqLen.int64, HEAD_DIM.int64]
      var numElements = 1 * NUM_HEADS * seqLen * HEAD_DIM
      var data = newSeq[float32](max(1, numElements))  # At least 1 to avoid nil

      checkStatus CreateTensorWithDataAsOrtValue(
        memoryInfo, data[0].unsafeAddr, (numElements * sizeof(float32)).csize_t,
        shape[0].unsafeAddr, shape.len.csize_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, inputValues[i].addr
      )
      createdTensors.add(i)

  # Run inference
  var outputValues = newSeq[OrtValue](whisper.outputNames.len)
  var outputNamePtrs = newSeq[cstring](whisper.outputNames.len)
  for i, name in whisper.outputNames:
    outputNamePtrs[i] = name.cstring

  checkStatus Run(
    getSession(whisper.decoder.internal), nil, inputNamePtrs[0].addr, inputValues[0].addr,
    inputValues.len.csize_t, outputNamePtrs[0].addr, outputValues.len.csize_t, outputValues[0].addr
  )

  # Get next token from logits
  var logitsPtr: ptr float32
  checkStatus GetTensorMutableData(outputValues[0], cast[ptr pointer](logitsPtr.addr))

  # Get logits shape
  var typeInfo: OrtTypeInfo
  checkStatus GetTypeInfo(outputValues[0], typeInfo.addr)
  var tensorInfo: OrtTensorTypeAndShapeInfo
  checkStatus CastTypeInfoToTensorInfo(typeInfo, tensorInfo.addr)
  var dimsCount: csize_t
  checkStatus GetDimensionsCount(tensorInfo, dimsCount.addr)
  var logitsShape = newSeq[int64](dimsCount)
  if dimsCount > 0:
    checkStatus GetDimensions(tensorInfo, logitsShape[0].addr, dimsCount)

  let seqLen = if dimsCount >= 2: logitsShape[1].int else: 1
  let lastPosOffset = (seqLen - 1) * VOCAB_SIZE

  var maxLogit = -Inf.float32
  var nextToken: int64 = 0
  let logitsArray = cast[ptr UncheckedArray[float32]](logitsPtr)
  for i in 0 ..< VOCAB_SIZE:
    let logit = logitsArray[lastPosOffset + i]
    if logit > maxLogit:
      maxLogit = logit
      nextToken = i.int64

  # Cleanup
  ReleaseValue(inputIdsValue)
  ReleaseValue(cachePosValue)
  ReleaseValue(useCacheValue)
  for v in outputValues:
    ReleaseValue(v)
  for i in createdTensors:
    ReleaseValue(inputValues[i])

  result = nextToken

proc generate(
  whisper: var MergedWhisperModel,
  encoderOutput: OrtValue,
  config: WhisperConfig,
  vocab: seq[string],
  maxLength: int = 50
): seq[int64] =
  ## Generate tokens using merged decoder (always use_cache=false for correctness)
  var inputIds = @[
    config.startToken.int64,
    config.langToken.int64,
    config.taskToken.int64,
    config.noTimestampsToken.int64
  ]

  result = @[]

  for step in 0 ..< maxLength:
    let nextToken = whisper.runDecoderStep(inputIds, encoderOutput)
    let tokenText = if nextToken.int < vocab.len: vocab[nextToken.int] else: "<unk>"
    echo "Step " & $step & ": inputLen=" & $inputIds.len & " -> " & $nextToken & " = " & tokenText

    if nextToken == config.endToken.int64:
      if result.len >= 3: break
      continue

    result.add(nextToken)
    inputIds.add(nextToken)
    if result.len >= maxLength: break

suite "Whisper ASR with Merged FP16 Decoder":

  test "Full pipeline (use_cache=false mode)":
    if not fileExists(EncoderPath) or not fileExists(DecoderPath) or not fileExists(TestAudioPath):
      skip()

    let config = loadConfig()
    let vocab = loadVocab()

    echo "\n=== Merged FP16 Decoder (use_cache=false mode) ==="
    echo "NOTE: KV-cache mode produces incorrect results with this FP16 model"

    let audio = loadWavFile(TestAudioPath)
    let melSpec = computeWhisperMelSpectrogram(padOrTrimAudio(audio, WHISPER_SAMPLE_RATE * 30))
    echo "✓ Computed mel spectrogram"

    var whisper = loadMergedWhisper(EncoderPath, DecoderPath)
    defer: whisper.close()
    echo "✓ Models loaded"

    let encoderOutput = whisper.runEncoder(melSpec)
    echo "✓ Encoder complete"

    echo "Running decoder..."
    let tokens = generate(whisper, encoderOutput, config, vocab, maxLength=20)

    let transcription = decodeTokensToText(tokens, vocab)
    echo "\nTranscription:"
    echo repeat("=", 60)
    echo transcription
    echo repeat("=", 60)
    echo "Generated " & $tokens.len & " tokens"

    ReleaseValue(encoderOutput)

    check tokens.len > 0
    check transcription.len > 0
    # Verify expected transcription
    # check transcription == "我現在都在車子的位置吧看看能不能正常運行"
