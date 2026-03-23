## kokoro_utils.nim
## Kokoro-82M TTS model specific utilities
## This is application-level code, not part of the core onnxruntime library
##
## Kokoro-82M uses a phoneme-based approach:
## - Input: Phoneme token IDs
## - Style vector from voices.bin file
## - Output: 24kHz audio

import onnx_rt
import onnx_rt/ort_bindings
import std/[strutils, tables, streams]

# Re-export types needed by users of this module
export InputTensor, OutputTensor, Model

#------------------------------------------------------------------------------
# Phoneme Vocabulary (from config.json)
#------------------------------------------------------------------------------

const KokoroVocab* = {
  ";": 1'i64, ":": 2'i64, ",": 3'i64, ".": 4'i64, "!": 5'i64, "?": 6'i64,
  "—": 9'i64, "…": 10'i64, "\"": 11'i64, "(": 12'i64, ")": 13'i64,
  " ": 16'i64,
  "ʣ": 18'i64, "ʥ": 19'i64, "ʦ": 20'i64, "ʨ": 21'i64,
  "A": 24'i64, "I": 25'i64, "O": 31'i64, "Q": 33'i64, "S": 35'i64,
  "T": 36'i64, "W": 39'i64, "Y": 41'i64,
  "a": 43'i64, "b": 44'i64, "c": 45'i64, "d": 46'i64, "e": 47'i64,
  "f": 48'i64, "h": 50'i64, "i": 51'i64, "j": 52'i64, "k": 53'i64,
  "l": 54'i64, "m": 55'i64, "n": 56'i64, "o": 57'i64, "p": 58'i64,
  "q": 59'i64, "r": 60'i64, "s": 61'i64, "t": 62'i64, "u": 63'i64,
  "v": 64'i64, "w": 65'i64, "x": 66'i64, "y": 67'i64, "z": 68'i64,
  "ɑ": 69'i64, "ɐ": 70'i64, "ɒ": 71'i64, "æ": 72'i64,
  "β": 75'i64, "ɔ": 76'i64, "ɕ": 77'i64, "ç": 78'i64,
  "ɖ": 80'i64, "ð": 81'i64, "ʤ": 82'i64, "ə": 83'i64, "ɚ": 85'i64,
  "ɛ": 86'i64, "ɜ": 87'i64, "ɟ": 90'i64, "ɡ": 92'i64,
  "ɥ": 99'i64, "ɨ": 101'i64, "ɪ": 102'i64, "ʝ": 103'i64,
  "ɯ": 110'i64, "ɰ": 111'i64, "ŋ": 112'i64, "ɳ": 113'i64,
  "ɲ": 114'i64, "ɴ": 115'i64, "ø": 116'i64,
  "ɸ": 118'i64, "θ": 119'i64, "œ": 120'i64,
  "ɹ": 123'i64, "ɾ": 125'i64, "ɻ": 126'i64,
  "ʁ": 128'i64, "ɽ": 129'i64, "ʂ": 130'i64, "ʃ": 131'i64,
  "ʈ": 132'i64, "ʧ": 133'i64, "ʊ": 135'i64, "ʋ": 136'i64,
  "ʌ": 138'i64, "ɣ": 139'i64, "ɤ": 140'i64,
  "χ": 142'i64, "ʎ": 143'i64, "ʒ": 147'i64, "ʔ": 148'i64,
  "ˈ": 156'i64, "ˌ": 157'i64, "ː": 158'i64, "ʰ": 162'i64,
  "ʲ": 164'i64, "↓": 169'i64, "→": 171'i64, "↗": 172'i64,
  "↘": 173'i64, "ᵻ": 177'i64,
}.toTable()

#------------------------------------------------------------------------------
# Voice Loading
#------------------------------------------------------------------------------

type
  VoiceData* = object
    ## Voice data loaded from voices.bin file
    ## Shape: (num_voices, max_length, 256)
    data*: seq[float32]
    numVoices*: int
    maxLength*: int

proc loadVoices*(path: string): VoiceData =
  ## Load voice data from binary file.
  ## The file contains float32 values in shape (num_voices, max_length, 256)
  
  let stream = newFileStream(path, fmRead)
  if stream == nil:
    raise newException(IOError, "Cannot open voice file: " & path)
  
  defer: stream.close()
  
  # Read all float32 values
  var values: seq[float32] = @[]
  while not stream.atEnd():
    var val: float32
    if stream.readData(val.addr, sizeof(float32)) == sizeof(float32):
      values.add(val)
  
  # Kokoro voices.bin has shape (num_voices, max_length, 256)
  # num_voices varies by file (usually 1 or multiple)
  # max_length is typically 510 or similar
  # Each style vector is 256-dimensional
  
  let totalElements = values.len
  let featureDim = 256
  let maxLen = totalElements div featureDim
  
  result.data = values
  result.numVoices = 1  # Default to 1 voice
  result.maxLength = maxLen

proc getStyle*(voices: VoiceData, tokenLen: int): seq[float32] =
  ## Get style vector for a given token length.
  ## Returns a 256-dimensional style vector.
  ##
  ## Parameters:
  ##   voices: Voice data loaded from file
  ##   tokenLen: Length of token sequence (used to index into voices)
  ##
  ## Returns:
  ##   256-dimensional style vector
  
  # Clamp token length to valid range
  let idx = min(tokenLen, voices.maxLength - 1)
  let offset = idx * 256
  
  result = newSeq[float32](256)
  for i in 0 ..< 256:
    if offset + i < voices.data.len:
      result[i] = voices.data[offset + i]
    else:
      result[i] = 0.0'f32

#------------------------------------------------------------------------------
# Text Processing
#------------------------------------------------------------------------------

proc phonemesToIds*(phonemes: string): seq[int64] =
  ## Convert phoneme string to token IDs.
  ## 
  ## Parameters:
  ##   phonemes: Space-separated phoneme string
  ##
  ## Returns:
  ##   Sequence of token IDs
  
  let parts = phonemes.splitWhitespace()
  for p in parts:
    if KokoroVocab.hasKey(p):
      result.add(KokoroVocab[p])
    else:
      # Unknown phoneme - skip or use space
      if KokoroVocab.hasKey(" "):
        result.add(KokoroVocab[" "])

proc prepareInputIds*(tokenIds: seq[int64]): seq[int64] =
  ## Prepare input IDs with padding.
  ## Adds pad token (0) at start and end.
  ##
  ## Parameters:
  ##   tokenIds: Raw token IDs
  ##
  ## Returns:
  ##   Padded token IDs: [0, ...tokenIds..., 0]
  
  result.add(0'i64)  # Start pad
  for id in tokenIds:
    result.add(id)
  result.add(0'i64)  # End pad

#------------------------------------------------------------------------------
# Kokoro TTS Inference
#------------------------------------------------------------------------------

proc runKokoroTTS*(
  model: Model,
  inputIds: seq[int64],
  style: seq[float32],
  speed: float32 = 1.0'f32
): OutputTensor =
  ## Run inference on Kokoro-82M TTS model.
  ##
  ## Parameters:
  ##   inputIds: Padded token IDs with shape (seq_len,)
  ##   style: Style vector with shape (256,)
  ##   speed: Speech speed multiplier (default: 1.0, <1=faster, >1=slower)
  ##
  ## Returns:
  ##   Output tensor with raw audio samples (float32, 24kHz)
  
  if inputIds.len == 0:
    raise newException(ValueError, "Input IDs cannot be empty")
  if style.len != 256:
    raise newException(ValueError, "Style vector must be 256-dimensional")
  
  let batchSize = 1'i64
  let seqLen = inputIds.len.int64
  
  var status: OrtStatusPtr
  
  # Create CPU memory info
  var memoryInfo: OrtMemoryInfo
  status = CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)
  checkStatus(status)
  
  # Prepare shapes
  var inputIdsShape = @[batchSize, seqLen]
  var styleShape = @[batchSize, 256'i64]
  var speedShape = @[batchSize]
  
  # Create input_ids tensor [batch, seq_len]
  var inputIdsOrtValue: OrtValue = nil
  let inputIdsDataSize = inputIds.len * sizeof(int64)
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    inputIds[0].unsafeAddr,
    inputIdsDataSize.csize_t,
    inputIdsShape[0].unsafeAddr,
    inputIdsShape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    inputIdsOrtValue.addr
  )
  checkStatus(status)
  
  # Create style tensor [batch, 256]
  var styleOrtValue: OrtValue = nil
  let styleDataSize = style.len * sizeof(float32)
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    style[0].unsafeAddr,
    styleDataSize.csize_t,
    styleShape[0].unsafeAddr,
    styleShape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    styleOrtValue.addr
  )
  checkStatus(status)
  
  # Create speed tensor [batch]
  var speedData = @[speed]
  var speedOrtValue: OrtValue = nil
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    speedData[0].unsafeAddr,
    sizeof(float32).csize_t,
    speedShape[0].unsafeAddr,
    speedShape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    speedOrtValue.addr
  )
  checkStatus(status)
  
  # Prepare input names and values
  # Note: HuggingFace Kokoro uses 'input_ids', 'style', 'speed'
  var inputNames: seq[cstring] = @[
    "input_ids".cstring,
    "style".cstring,
    "speed".cstring
  ]
  var inputs: seq[OrtValue] = @[
    inputIdsOrtValue,
    styleOrtValue,
    speedOrtValue
  ]
  
  # Run inference
  let outputName = "audio".cstring
  var outputOrtValue: OrtValue = nil
  status = Run(
    getSession(model.internal),
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
  
  # Copy data to Nim seq
  let floatPtr = cast[ptr UncheckedArray[float32]](outputDataPtr)
  var outputData = newSeq[float32](elemCount)
  for i in 0 ..< elemCount.int:
    outputData[i] = floatPtr[i]
  
  result = OutputTensor(
    data: outputData,
    shape: outputShape
  )

#------------------------------------------------------------------------------
# Audio Output Helpers
#------------------------------------------------------------------------------

proc toInt16Samples*(output: OutputTensor): seq[int16] =
  ## Convert float32 audio samples to int16 format for WAV file.
  ## Kokoro outputs 24kHz audio.
  result = newSeq[int16](output.data.len)
  for i in 0 ..< output.data.len:
    let sample = output.data[i]
    let clamped = max(-1.0'f32, min(1.0'f32, sample))
    result[i] = int16(clamped * 32767.0'f32)

proc sampleCount*(output: OutputTensor): int =
  result = output.data.len

const SampleRate* = 24000  ## Kokoro uses 24kHz sample rate
