## kokoro_sherpa_utils.nim
## Sherpa-ONNX Kokoro Multi-Lang TTS model utilities
## 
## This version supports kokoro-multi-lang-v1_0 from sherpa-onnx,
## which includes built-in text processing via lexicon and espeak-ng.
##
## Model download:
##   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
##   tar xf kokoro-multi-lang-v1_0.tar.bz2
##
## Supported languages: Chinese, English (and more)
## You can input raw text directly (no need for external phonemizer)

import onnx_rt
import onnx_rt/ort_bindings
import std/[strutils, tables, streams, unicode]

# Re-export types needed by users of this module
export InputTensor, OutputTensor, Model

#------------------------------------------------------------------------------
# Token Loading
#------------------------------------------------------------------------------

proc loadTokens*(path: string): Table[string, int64] =
  ## Load token to ID mapping from tokens.txt
  ## Format: token id (one per line, space-separated)
  
  let content = readFile(path)
  for line in content.splitLines:
    let parts = strutils.splitWhitespace(line.strip())
    if parts.len >= 2:
      let token = parts[0]
      let id = parseInt(parts[^1]).int64
      result[token] = id

#------------------------------------------------------------------------------
# Lexicon Loading (for Chinese and English)
#------------------------------------------------------------------------------

type
  LexiconEntry* = object
    word*: string        # Original word/character
    phonemes*: seq[string]  # Phoneme sequence

proc loadLexicon*(path: string): Table[string, seq[string]] =
  ## Load lexicon from file.
  ## Format: word phoneme1 phoneme2 ...
  ## Example: 你好 n i h ao
  ##          hello h ə l o ʊ
  
  let content = readFile(path)
  for line in content.splitLines:
    let parts = strutils.splitWhitespace(line.strip())
    if parts.len < 2:
      continue
    
    let word = parts[0]
    var phonemes: seq[string] = @[]
    
    for i in 1 ..< parts.len:
      phonemes.add(parts[i])
    
    result[word] = phonemes

#------------------------------------------------------------------------------
# Text Processing
#------------------------------------------------------------------------------

proc textToTokens*(
  text: string,
  lexiconZh: Table[string, seq[string]],
  lexiconEn: Table[string, seq[string]],
  tokens: Table[string, int64]
): seq[int64] =
  ## Convert text to token IDs using lexicons.
  ## 
  ## This is a simplified G2P (Grapheme-to-Phoneme) conversion.
  ## For production use, proper Chinese/English G2P is recommended.
  ##
  ## Parameters:
  ##   text: Input text (Chinese, English, or mixed)
  ##   lexiconZh: Chinese lexicon
  ##   lexiconEn: English lexicon
  ##   tokens: Token to ID mapping
  
  var phonemes: seq[string] = @[]
  
  # Process character by character/word by word
  var i = 0
  let chars = text.toRunes()
  
  while i < chars.len:
    let char = $chars[i]
    
    # Try Chinese lexicon first (for Chinese characters)
    if lexiconZh.hasKey(char):
      let entry = lexiconZh[char]
      for p in entry:
        phonemes.add(p)
      inc i
    
    # Try English word (lookahead for multi-character words)
    elif char[0] in {'a'..'z', 'A'..'Z'}:
      # Build English word
      var word = char.toLowerAscii()
      var j = i + 1
      while j < chars.len and ($chars[j])[0] in {'a'..'z', 'A'..'Z'}:
        word = word & ($chars[j]).toLowerAscii()
        inc j
      
      if lexiconEn.hasKey(word):
        let entry = lexiconEn[word]
        for p in entry:
          phonemes.add(p)
      else:
        # Unknown word - spell out letters
        for c in word:
          phonemes.add($c)
      
      i = j
    
    # Punctuation and space
    elif char == " ":
      phonemes.add(" ")
      inc i
    elif char == ".":
      phonemes.add(".")
      inc i
    elif char == ",":
      phonemes.add(",")
      inc i
    elif char == "?":
      phonemes.add("?")
      inc i
    elif char == "!":
      phonemes.add("!")
      inc i
    else:
      # Skip unknown characters
      inc i
  
  # Convert phonemes to token IDs
  for phoneme in phonemes:
    if tokens.hasKey(phoneme):
      result.add(tokens[phoneme])
    elif tokens.hasKey("<unk>"):
      result.add(tokens["<unk>"])

# Simple character-based tokenization for Chinese
proc chineseTextToTokens*(
  text: string,
  lexiconZh: Table[string, seq[string]],
  tokens: Table[string, int64]
): seq[int64] =
  ## Convert Chinese text to token IDs using lexicon.
  
  for char in text.runes:
    let charStr = $char
    if lexiconZh.hasKey(charStr):
      let phonemes = lexiconZh[charStr]
      for p in phonemes:
        if tokens.hasKey(p):
          result.add(tokens[p])

#------------------------------------------------------------------------------
# Voice Loading
#------------------------------------------------------------------------------

type
  VoiceData* = object
    ## Voice data loaded from voices.bin file
    data*: seq[float32]
    numVoices*: int
    maxLength*: int

proc loadVoices*(path: string): VoiceData =
  ## Load voice data from binary file.
  
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
  
  let totalElements = values.len
  let featureDim = 256
  let maxLen = totalElements div featureDim
  
  result.data = values
  result.numVoices = 1
  result.maxLength = maxLen

proc getStyle*(voices: VoiceData, tokenLen: int, speakerId: int = 0): seq[float32] =
  ## Get style vector for a given token length and speaker.
  
  let idx = min(tokenLen, voices.maxLength - 1)
  let offset = idx * 256
  
  result = newSeq[float32](256)
  for i in 0 ..< 256:
    if offset + i < voices.data.len:
      result[i] = voices.data[offset + i]
    else:
      result[i] = 0.0'f32

#------------------------------------------------------------------------------
# Kokoro Sherpa-ONNX Inference
#------------------------------------------------------------------------------

proc runKokoroSherpa*(
  model: Model,
  inputIds: seq[int64],
  style: seq[float32],
  speed: float32 = 1.0'f32
): OutputTensor =
  ## Run inference on Sherpa-ONNX Kokoro Multi-Lang model.
  ##
  ## Parameters:
  ##   inputIds: Token IDs with padding [0, ...tokens..., 0]
  ##   style: Style vector (256-dimensional)
  ##   speed: Speech speed (default: 1.0, <1=slower, >1=faster)
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
  
  # Create input_ids tensor
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
  
  # Create style tensor
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
  
  # Create speed tensor
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
  # Note: Sherpa-ONNX Kokoro uses 'tokens' not 'input_ids'
  var inputNames: seq[cstring] = @[
    "tokens".cstring,
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
    nil,
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
  ## Convert float32 audio samples to int16 format.
  result = newSeq[int16](output.data.len)
  for i in 0 ..< output.data.len:
    let sample = output.data[i]
    let clamped = max(-1.0'f32, min(1.0'f32, sample))
    result[i] = int16(clamped * 32767.0'f32)

proc sampleCount*(output: OutputTensor): int =
  result = output.data.len

const SampleRate* = 24000  ## Kokoro uses 24kHz sample rate

#------------------------------------------------------------------------------
# High-level API
#------------------------------------------------------------------------------

proc synthesize*(
  model: Model,
  text: string,
  lexiconZh: Table[string, seq[string]],
  lexiconEn: Table[string, seq[string]],
  tokens: Table[string, int64],
  voices: VoiceData,
  speakerId: int = 0,
  speed: float32 = 1.0'f32
): OutputTensor =
  ## High-level synthesis function.
  ## Converts text to speech in one step.
  ##
  ## Parameters:
  ##   text: Input text (Chinese, English, or mixed)
  ##   lexiconZh: Chinese lexicon
  ##   lexiconEn: English lexicon
  ##   tokens: Token mapping
  ##   voices: Voice data
  ##   speakerId: Speaker ID (default: 0)
  ##   speed: Speech speed (default: 1.0)
  
  # Convert text to token IDs
  let tokenIds = textToTokens(text, lexiconZh, lexiconEn, tokens)
  if tokenIds.len == 0:
    raise newException(ValueError, "No tokens generated from text")
  
  # Add padding
  var inputIds = @[0'i64]  # Start pad
  for id in tokenIds:
    inputIds.add(id)
  inputIds.add(0'i64)  # End pad
  
  # Get style
  let style = voices.getStyle(tokenIds.len, speakerId)
  
  # Run inference
  result = runKokoroSherpa(model, inputIds, style, speed)
