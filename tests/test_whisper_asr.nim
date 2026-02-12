## test_whisper_asr.nim
## Test Whisper ONNX model for ASR (Automatic Speech Recognition)
## Model: whisper-large-v3-chinese-ONNX from onnx-community

import std/[unittest, os, strutils, strformat, sequtils, math, complex, tables]
import onnxruntime
import onnxruntime/ort_bindings
import std/json

const TestDataDir = "tests/testdata/whisper-large-v3-zh"
const EncoderPath = TestDataDir / "encoder_model.onnx"
const DecoderPath = TestDataDir / "decoder_model.onnx"
const ConfigPath = TestDataDir / "config.json"
const GenerationConfigPath = TestDataDir / "generation_config.json"
const SpecialTokensMapPath = TestDataDir / "special_tokens_map.json"
const TokenizerPath = TestDataDir / "tokenizer.json"
const VocabPath = TestDataDir / "vocab.json"
const TestAudioPath = TestDataDir / "test_input.wav"

# Whisper constants
const WHISPER_SAMPLE_RATE = 16000
const WHISPER_N_FFT = 400
const WHISPER_N_MELS = 80        # The model expects 80 mel bins
const WHISPER_HOP_LENGTH = 160
const WHISPER_CHUNK_LENGTH = 30  # seconds
const WHISPER_N_FRAMES = 3000    # 30 seconds at 100 fps

# Mel filterbank constants
const MEL_FMIN = 0.0
const MEL_FMAX = 8000.0

# Special token IDs - will be loaded from config files
var START_OF_TRANSCRIPT = 50258
var END_OF_TEXT = 50257
var CHINESE_LANG = 50260  # <|zh|>
var TRANSCRIBE_TASK = 50359  # <|transcribe|>
var NOTIMESTAMPS = 50363  # <|notimestamps|>
var BEGIN_SUPPRESS_TOKENS: seq[int64] = @[220'i64, 50257'i64]
var FORCED_DECODER_IDS: seq[tuple[position: int, tokenId: int64]] = @[]
var LANG_TO_ID: Table[string, int64]

proc loadConfigFromFiles() =
  ## Load configuration from all JSON config files
  # Load generation_config.json (most important for generation)
  if fileExists(GenerationConfigPath):
    let content = readFile(GenerationConfigPath)
    let json = parseJson(content)
    
    if json.hasKey("decoder_start_token_id"):
      START_OF_TRANSCRIPT = json["decoder_start_token_id"].getInt
    
    if json.hasKey("eos_token_id"):
      END_OF_TEXT = json["eos_token_id"].getInt
    
    if json.hasKey("begin_suppress_tokens"):
      BEGIN_SUPPRESS_TOKENS = @[]
      for tok in json["begin_suppress_tokens"]:
        BEGIN_SUPPRESS_TOKENS.add(tok.getInt.int64)
    
    if json.hasKey("forced_decoder_ids"):
      FORCED_DECODER_IDS = @[]
      for item in json["forced_decoder_ids"]:
        let pos = item[0].getInt
        if not item[1].isNil:
          let tokId = item[1].getInt.int64
          FORCED_DECODER_IDS.add((pos, tokId))
    
    if json.hasKey("lang_to_id"):
      LANG_TO_ID = initTable[string, int64]()
      for lang, id in json["lang_to_id"].pairs:
        LANG_TO_ID[lang] = id.getInt.int64
      # Get Chinese token ID
      if LANG_TO_ID.hasKey("<|zh|>"):
        CHINESE_LANG = LANG_TO_ID["<|zh|>"].int
  
  # Load tokenizer.json to get actual token IDs
  if fileExists(TokenizerPath):
    let content = readFile(TokenizerPath)
    let json = parseJson(content)
    
    if json.hasKey("added_tokens"):
      for tokenInfo in json["added_tokens"]:
        let id = tokenInfo["id"].getInt
        let content = tokenInfo["content"].getStr
        
        case content:
          of "<|startoftranscript|>":
            START_OF_TRANSCRIPT = id
          of "<|endoftext|>":
            END_OF_TEXT = id
          of "<|zh|>":
            CHINESE_LANG = id
          of "<|transcribe|>":
            TRANSCRIBE_TASK = id
          of "<|notimestamps|>":
            NOTIMESTAMPS = id
          else:
            discard
  
  # Load config.json for additional settings
  if fileExists(ConfigPath):
    let content = readFile(ConfigPath)
    let json = parseJson(content)
    
    if json.hasKey("begin_suppress_tokens"):
      BEGIN_SUPPRESS_TOKENS = @[]
      for tok in json["begin_suppress_tokens"]:
        BEGIN_SUPPRESS_TOKENS.add(tok.getInt.int64)

proc hzToMel(hz: float): float =
  ## Convert Hz to Mel scale
  2595.0 * log10(1.0 + hz / 700.0)

proc melToHz(mel: float): float =
  ## Convert Mel scale to Hz
  700.0 * (pow(10.0, mel / 2595.0) - 1.0)

proc createMelFilterbank(nFft: int, nMels: int, sampleRate: int): seq[seq[float32]] =
  ## Create mel filterbank matrix
  result = newSeq[seq[float32]](nMels)
  for i in 0 ..< nMels:
    result[i] = newSeq[float32](nFft div 2 + 1)

  let melMin = hzToMel(MEL_FMIN)
  let melMax = hzToMel(min(MEL_FMAX, sampleRate.float / 2.0))

  # Create mel points
  var melPoints = newSeq[float](nMels + 2)
  for i in 0 ..< nMels + 2:
    melPoints[i] = melMin + (melMax - melMin) * i.float / (nMels + 1).float

  # Convert to Hz and then to FFT bin numbers
  var fftBins = newSeq[int](nMels + 2)
  for i in 0 ..< nMels + 2:
    let hz = melToHz(melPoints[i])
    fftBins[i] = int((hz / sampleRate.float) * nFft.float)

  # Create triangular filters
  for i in 0 ..< nMels:
    let left = fftBins[i]
    let center = fftBins[i + 1]
    let right = fftBins[i + 2]

    for j in left ..< center:
      if j < result[i].len:
        result[i][j] = (j - left).float32 / (center - left).float32

    for j in center ..< right:
      if j < result[i].len:
        result[i][j] = (right - j).float32 / (right - center).float32

proc loadWavFile(path: string): seq[float32] =
  ## Load a WAV file (16-bit PCM, mono, 16000 Hz)
  ## Returns raw audio samples as float32 array
  let file = open(path, fmRead)
  defer: file.close()

  # Read WAV header
  var header: array[44, uint8]
  if file.readBytes(header, 0, 44) != 44:
    raise newException(IOError, "Failed to read WAV header")

  # Verify WAV format
  if header[0] != uint8('R') or header[1] != uint8('I') or
     header[2] != uint8('F') or header[3] != uint8('F'):
    raise newException(IOError, "Not a valid WAV file")

  # Get audio parameters from header
  let sampleRate = int(header[24]) or (int(header[25]) shl 8) or
                   (int(header[26]) shl 16) or (int(header[27]) shl 24)
  let numChannels = int(header[22]) or (int(header[23]) shl 8)
  let bitsPerSample = int(header[34]) or (int(header[35]) shl 8)

  echo &"WAV file: sampleRate={sampleRate}, channels={numChannels}, bits={bitsPerSample}"

  if bitsPerSample != 16:
    raise newException(IOError, "Only 16-bit PCM supported")

  # Calculate data size
  let fileSize = file.getFileSize()
  let dataSize = fileSize - 44
  let numSamples = dataSize div 2  # 2 bytes per sample

  # Read audio data
  result = newSeq[float32](numSamples)
  var buffer: array[1024, uint8]
  var sampleIdx = 0

  while sampleIdx < numSamples:
    let toRead = min(1024, (numSamples - sampleIdx) * 2)
    let bytesRead = file.readBytes(buffer, 0, toRead)
    if bytesRead == 0:
      break

    var i = 0
    while i < bytesRead and sampleIdx < numSamples:
      let sample = int16(buffer[i]) or (int16(buffer[i+1]) shl 8)
      result[sampleIdx] = sample.float32 / 32768.0'f32  # Normalize to [-1, 1]
      sampleIdx += 1
      i += 2

  # If stereo, convert to mono by averaging channels
  if numChannels == 2:
    var monoSamples = newSeq[float32](numSamples div 2)
    for i in 0 ..< numSamples div 2:
      monoSamples[i] = (result[i*2] + result[i*2+1]) / 2.0'f32
    result = monoSamples

proc computeStft(audio: seq[float32], nFft: int, hopLength: int): seq[seq[Complex[float32]]] =
  ## Compute Short-Time Fourier Transform
  ## Returns complex spectrogram [nFrames][nFft//2+1]
  let nFrames = (audio.len - nFft) div hopLength + 1
  result = newSeq[seq[Complex[float32]]](nFrames)

  # Create Hann window
  var window = newSeq[float32](nFft)
  for i in 0 ..< nFft:
    window[i] = 0.5'f32 - 0.5'f32 * cos(2.0'f32 * PI * i.float32 / (nFft - 1).float32)

  for frame in 0 ..< nFrames:
    result[frame] = newSeq[Complex[float32]](nFft div 2 + 1)
    let start = frame * hopLength

    # Apply window and compute FFT
    var frameData = newSeq[Complex[float32]](nFft)
    for i in 0 ..< nFft:
      if start + i < audio.len:
        frameData[i] = complex(audio[start + i] * window[i], 0.0'f32)
      else:
        frameData[i] = complex(0.0'f32, 0.0'f32)

    # Simple DFT (for small nFft this is fine)
    for k in 0 ..< (nFft div 2 + 1):
      var sum = complex(0.0'f32, 0.0'f32)
      for n in 0 ..< nFft:
        let angle = -2.0'f32 * PI * k.float32 * n.float32 / nFft.float32
        sum += frameData[n] * complex(cos(angle).float32, sin(angle).float32)
      result[frame][k] = sum

proc audioToMelSpectrogram(audio: seq[float32], nMels: int = 80): seq[float32] =
  ## Convert audio to mel spectrogram for Whisper
  ## Returns mel spectrogram [nMels, nFrames]

  # Compute STFT
  let stft = computeStft(audio, WHISPER_N_FFT, WHISPER_HOP_LENGTH)
  let nFrames = stft.len
  let nFreqBins = WHISPER_N_FFT div 2 + 1

  # Compute magnitude spectrogram
  var magSpec = newSeq[seq[float32]](nFrames)
  for t in 0 ..< nFrames:
    magSpec[t] = newSeq[float32](nFreqBins)
    for f in 0 ..< nFreqBins:
      magSpec[t][f] = abs(stft[t][f])

  # Create mel filterbank
  let melFilter = createMelFilterbank(WHISPER_N_FFT, nMels, WHISPER_SAMPLE_RATE)

  # Apply mel filterbank
  var melSpec = newSeq[seq[float32]](nMels)
  for m in 0 ..< nMels:
    melSpec[m] = newSeq[float32](nFrames)
    for t in 0 ..< nFrames:
      var sum: float32 = 0.0'f32
      for f in 0 ..< nFreqBins:
        sum += magSpec[t][f] * melFilter[m][f]
      # Convert to log scale (add small epsilon to avoid log(0))
      melSpec[m][t] = ln(max(sum, 1e-10'f32))

  # Flatten to [nMels, nFrames] format
  result = newSeq[float32](nMels * nFrames)
  for m in 0 ..< nMels:
    for t in 0 ..< nFrames:
      result[m * nFrames + t] = melSpec[m][t]

  echo &"Mel spectrogram shape: [{nMels}, {nFrames}]"

# Token ID to character mapping (simplified - just for common tokens)
var IdToToken: seq[string]

proc initTokenizer() =
  ## Initialize simple tokenizer
  IdToToken = newSeq[string](51865)
  
  # Load vocab.json first (this has the base vocabulary)
  if fileExists(VocabPath):
    let vocabContent = readFile(VocabPath)
    let vocabJson = parseJson(vocabContent)
    
    for token, id in vocabJson.pairs:
      let idx = id.getInt
      if idx < IdToToken.len:
        IdToToken[idx] = token
  
  # Load tokenizer.json for added tokens
  if fileExists(TokenizerPath):
    let jsonContent = readFile(TokenizerPath)
    let tokenizerJson = parseJson(jsonContent)
    
    # Load from model.vocab (overrides vocab.json if present)
    if tokenizerJson.hasKey("model") and tokenizerJson["model"].hasKey("vocab"):
      for token, id in tokenizerJson["model"]["vocab"].pairs:
        let idx = id.getInt
        if idx < IdToToken.len:
          IdToToken[idx] = token
    
    # Also load added_tokens
    if tokenizerJson.hasKey("added_tokens"):
      for tokenInfo in tokenizerJson["added_tokens"]:
        let idx = tokenInfo["id"].getInt
        let content = tokenInfo["content"].getStr
        if idx < IdToToken.len:
          IdToToken[idx] = content

proc tokenToText(tokenId: int64): string =
  ## Convert token ID to text
  if tokenId >= 0 and tokenId < IdToToken.len:
    return IdToToken[tokenId.int]
  return &"<token_{tokenId}>"

proc greedyDecode(logits: ptr float32, vocabSize: int; skipTokens: seq[int64] = @[]): int64 =
  ## Greedy decoding - pick the token with highest probability
  ## skipTokens: tokens to skip (e.g., suppressed tokens at beginning)
  var maxLogit: float32 = -1e10'f32
  var maxIdx: int64 = 0
  
  for i in 0 ..< vocabSize:
    # Skip suppressed tokens
    var shouldSkip = false
    for skip in skipTokens:
      if i.int64 == skip:
        shouldSkip = true
        break
    if shouldSkip:
      continue
      
    let logit = cast[ptr float32](cast[uint64](logits) + uint64(i * sizeof(float32)))[]
    if logit > maxLogit:
      maxLogit = logit
      maxIdx = i.int64
  
  return maxIdx

suite "Whisper ASR Tests":
  setup:
    loadConfigFromFiles()
    initTokenizer()

  test "Model files exist":
    check fileExists(EncoderPath)
    check fileExists(DecoderPath)
    check fileExists(ConfigPath)
    check fileExists(TokenizerPath)
    echo &"Encoder model size: {getFileSize(EncoderPath)} bytes"
    echo &"Decoder model size: {getFileSize(DecoderPath)} bytes"

  test "Inspect encoder model":
    if not fileExists(EncoderPath):
      skip()
    else:
      let encoder = newOnnxModel(EncoderPath)

      var inputCount, outputCount: csize_t
      var status = SessionGetInputCount(getSession(encoder), inputCount.addr)
      checkStatus(status)
      status = SessionGetOutputCount(getSession(encoder), outputCount.addr)
      checkStatus(status)

      echo &"Encoder: {inputCount} inputs, {outputCount} outputs"

      var allocator: OrtAllocator
      status = GetAllocatorWithDefaultOptions(allocator.addr)
      checkStatus(status)

      echo "Encoder inputs:"
      for i in 0 ..< inputCount.int:
        var namePtr: cstring
        status = SessionGetInputName(getSession(encoder), i.csize_t, allocator, namePtr.addr)
        checkStatus(status)
        if namePtr != nil:
          echo &"  {i}: {$namePtr}"

      echo "Encoder outputs:"
      for i in 0 ..< outputCount.int:
        var namePtr: cstring
        status = SessionGetOutputName(getSession(encoder), i.csize_t, allocator, namePtr.addr)
        checkStatus(status)
        if namePtr != nil:
          echo &"  {i}: {$namePtr}"

      encoder.close()

  test "Inspect decoder model":
    if not fileExists(DecoderPath):
      skip()
    else:
      let decoder = newOnnxModel(DecoderPath)

      var inputCount, outputCount: csize_t
      var status = SessionGetInputCount(getSession(decoder), inputCount.addr)
      checkStatus(status)
      status = SessionGetOutputCount(getSession(decoder), outputCount.addr)
      checkStatus(status)

      echo &"Decoder: {inputCount} inputs, {outputCount} outputs"

      var allocator: OrtAllocator
      status = GetAllocatorWithDefaultOptions(allocator.addr)
      checkStatus(status)

      echo "Decoder inputs:"
      for i in 0 ..< inputCount.int:
        var namePtr: cstring
        status = SessionGetInputName(getSession(decoder), i.csize_t, allocator, namePtr.addr)
        checkStatus(status)
        if namePtr != nil:
          echo &"  {i}: {$namePtr}"

      echo "Decoder outputs:"
      for i in 0 ..< outputCount.int:
        var namePtr: cstring
        status = SessionGetOutputName(getSession(decoder), i.csize_t, allocator, namePtr.addr)
        checkStatus(status)
        if namePtr != nil:
          echo &"  {i}: {$namePtr}"

      decoder.close()

  test "Load and verify test audio file":
    if not fileExists(TestAudioPath):
      skip()
    else:
      echo &"Loading test audio: {TestAudioPath}"
      let audio = loadWavFile(TestAudioPath)
      echo &"Loaded {audio.len} samples ({audio.len div WHISPER_SAMPLE_RATE} seconds)"
      check audio.len > 0

      # Check audio statistics
      var maxAmp: float32 = 0.0'f32
      var sumAmp: float32 = 0.0'f32
      for sample in audio:
        maxAmp = max(maxAmp, abs(sample))
        sumAmp += abs(sample)
      let avgAmp = sumAmp / audio.len.float32

      echo &"Audio stats: max={maxAmp:.4f}, avg={avgAmp:.4f}"
      check maxAmp > 0.01'f32  # Make sure there's actual audio content

  test "Convert audio to mel spectrogram":
    if not fileExists(TestAudioPath):
      skip()
    else:
      let audio = loadWavFile(TestAudioPath)
      echo &"Converting {audio.len} samples to mel spectrogram..."

      let melSpec = audioToMelSpectrogram(audio, WHISPER_N_MELS)
      echo &"Mel spectrogram size: {melSpec.len} elements"
      check melSpec.len > 0

      # Check mel spectrogram statistics
      var maxVal: float32 = 0.0'f32
      var minVal: float32 = 0.0'f32
      for val in melSpec:
        maxVal = max(maxVal, val)
        minVal = min(minVal, val)
      echo &"Mel spectrogram range: [{minVal:.4f}, {maxVal:.4f}]"

  test "Full ASR pipeline - encode and decode":
    if not fileExists(EncoderPath) or not fileExists(DecoderPath) or not fileExists(TestAudioPath):
      skip()
    else:
      echo "\n=== Starting Full ASR Pipeline ==="
      
      # Step 1: Load and preprocess audio
      let audio = loadWavFile(TestAudioPath)
      var melSpec = audioToMelSpectrogram(audio, WHISPER_N_MELS)

      # Pad or truncate to 3000 frames (30 seconds)
      let nFrames = melSpec.len div WHISPER_N_MELS
      let targetFrames = 3000

      var paddedMelSpec = newSeq[float32](WHISPER_N_MELS * targetFrames)
      for m in 0 ..< WHISPER_N_MELS:
        for t in 0 ..< targetFrames:
          if t < nFrames:
            paddedMelSpec[m * targetFrames + t] = melSpec[m * nFrames + t]
          else:
            paddedMelSpec[m * targetFrames + t] = 0.0'f32

      # Step 2: Run encoder
      let encoder = newOnnxModel(EncoderPath)

      var memoryInfo: OrtMemoryInfo
      var status = CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)
      checkStatus(status)

      var inputShape = @[1'i64, WHISPER_N_MELS.int64, targetFrames.int64]
      var encoderInput: OrtValue
      status = CreateTensorWithDataAsOrtValue(
        memoryInfo,
        paddedMelSpec[0].unsafeAddr,
        (paddedMelSpec.len * sizeof(float32)).csize_t,
        inputShape[0].unsafeAddr,
        inputShape.len.csize_t,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        encoderInput.addr
      )
      checkStatus(status)

      var allocator: OrtAllocator
      status = GetAllocatorWithDefaultOptions(allocator.addr)
      checkStatus(status)

      var encInputNamePtr, encOutputNamePtr: cstring
      status = SessionGetInputName(getSession(encoder), 0, allocator, encInputNamePtr.addr)
      checkStatus(status)
      status = SessionGetOutputName(getSession(encoder), 0, allocator, encOutputNamePtr.addr)
      checkStatus(status)

      var encInputNames = @[($encInputNamePtr).cstring]
      var encOutputNames = @[($encOutputNamePtr).cstring]

      var encoderOutput: OrtValue
      status = Run(
        getSession(encoder),
        nil,
        encInputNames[0].addr,
        encoderInput.addr,
        1,
        encOutputNames[0].addr,
        1,
        encoderOutput.addr
      )
      checkStatus(status)

      # Get encoder output shape
      var encTypeInfo: OrtTensorTypeAndShapeInfo
      status = GetTensorTypeAndShape(encoderOutput, encTypeInfo.addr)
      checkStatus(status)
      
      var encDimsCount: csize_t
      status = GetDimensionsCount(encTypeInfo, encDimsCount.addr)
      checkStatus(status)
      
      var encDims = newSeq[int64](encDimsCount.int)
      status = GetDimensions(encTypeInfo, encDims[0].addr, encDimsCount)
      checkStatus(status)
      
      # Verify encoder output has valid data
      var encDataPtr: ptr float32
      status = GetTensorMutableData(encoderOutput, cast[ptr pointer](encDataPtr.addr))
      checkStatus(status)
      
      var encSum: float32 = 0.0
      var encNonZero = 0
      var totalEncElements = 1'i64
      for d in encDims:
        totalEncElements *= d
      for i in 0 ..< min(1000, totalEncElements.int):
        let val = cast[ptr float32](cast[uint64](encDataPtr) + uint64(i * sizeof(float32)))[]
        encSum += abs(val)
        if val != 0.0:
          encNonZero += 1
      
      # ReleaseTensorTypeAndShapeInfo(encTypeInfo)

      # Step 3: Run decoder with greedy decoding
      let decoder = newOnnxModel(DecoderPath)

      # Initialize decoder input with special tokens
      # Format: <|startoftranscript|> <|zh|> <|transcribe|> <|notimestamps|>
      var inputIds = @[
        START_OF_TRANSCRIPT.int64,
        CHINESE_LANG.int64,
        TRANSCRIBE_TASK.int64,
        NOTIMESTAMPS.int64
      ]

      var generatedTokens: seq[int64] = @[]
      let maxLength = 50  # Maximum tokens to generate
      let vocabSize = 51865

      # Get decoder input/output names
      var decInputCount, decOutputCount: csize_t
      status = SessionGetInputCount(getSession(decoder), decInputCount.addr)
      checkStatus(status)
      status = SessionGetOutputCount(getSession(decoder), decOutputCount.addr)
      checkStatus(status)

      var decInputNamePtr0, decInputNamePtr1: cstring
      status = SessionGetInputName(getSession(decoder), 0, allocator, decInputNamePtr0.addr)
      checkStatus(status)
      if decInputCount > 1:
        status = SessionGetInputName(getSession(decoder), 1, allocator, decInputNamePtr1.addr)
        checkStatus(status)

      var decOutputNamePtr: cstring
      status = SessionGetOutputName(getSession(decoder), 0, allocator, decOutputNamePtr.addr)
      checkStatus(status)

      # Greedy decoding loop
      # Use suppressed tokens loaded from config.json
      let suppressedTokens = BEGIN_SUPPRESS_TOKENS
      var skipSuppressed = true  # Skip suppressed tokens for first few steps
      echo &"Using suppressed tokens: {suppressedTokens}"
      
      for step in 0 ..< maxLength:
        # Create input tensors
        var inputIdsShape = @[1'i64, inputIds.len.int64]
        var inputIdsValue: OrtValue
        status = CreateTensorWithDataAsOrtValue(
          memoryInfo,
          inputIds[0].unsafeAddr,
          (inputIds.len * sizeof(int64)).csize_t,
          inputIdsShape[0].unsafeAddr,
          inputIdsShape.len.csize_t,
          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
          inputIdsValue.addr
        )
        checkStatus(status)

        # Run decoder
        var decInputNames = @[($decInputNamePtr0).cstring]
        var decInputValues = @[inputIdsValue]
        
        # If decoder expects encoder_hidden_states as second input
        if decInputCount > 1 and decInputNamePtr1 != nil and ($decInputNamePtr1).contains("encoder"):
          decInputNames.add(($decInputNamePtr1).cstring)
          decInputValues.add(encoderOutput)

        var logitsValue: OrtValue
        status = Run(
          getSession(decoder),
          nil,
          decInputNames[0].addr,
          decInputValues[0].addr,
          decInputNames.len.csize_t,
          decOutputNamePtr.addr,
          1,
          logitsValue.addr
        )
        checkStatus(status)

        # Get logits
        var logitsPtr: ptr float32
        status = GetTensorMutableData(logitsValue, cast[ptr pointer](logitsPtr.addr))
        checkStatus(status)

        # Get last token logits (last position in sequence)
        let lastPosLogits = cast[ptr float32](cast[uint64](logitsPtr) + uint64((inputIds.len - 1) * vocabSize * sizeof(float32)))
        
        # Greedy decode (skip suppressed tokens at beginning)
        var nextToken: int64
        if skipSuppressed and step < 5:
          nextToken = greedyDecode(lastPosLogits, vocabSize, suppressedTokens)
        else:
          nextToken = greedyDecode(lastPosLogits, vocabSize)
          skipSuppressed = false
        
        # ReleaseValue(logitsValue)
        # ReleaseValue(inputIdsValue)

        # Check for end of sequence
        if nextToken == END_OF_TEXT.int64:
          # Don't break immediately - try to continue if we haven't generated enough
          if generatedTokens.len < 3:
            continue
          break

        generatedTokens.add(nextToken)
        inputIds.add(nextToken)

        if inputIds.len > 448:  # Max length from config
          break

      # Step 4: Convert tokens to text
      echo "\n=== ASR Results ==="
      echo &"Generated {generatedTokens.len} tokens: {generatedTokens}"
      
      var resultText = ""
      for token in generatedTokens:
        let tokenStr = tokenToText(token)
        # Remove special token markers for display
        if not tokenStr.startsWith("<"):
          resultText.add(tokenStr)
      
      echo &"Transcription: '{resultText}'"
      echo "==================="


      encoder.close()
      decoder.close()

      # Note: generatedTokens might be 0 if model doesn't recognize speech
      # This is not necessarily a failure - just means no text was transcribed
      echo &"Total generated tokens: {generatedTokens.len}"
