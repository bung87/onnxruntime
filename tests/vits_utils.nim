## vits_utils.nim
## Sherpa-ONNX VITS TTS model specific utilities
## This is application-level code, not part of the core onnxruntime library
##
## MeloTTS uses a lexicon-based approach:
## - Input: Chinese characters (汉字)
## - Lexicon maps characters to phonemes
## - Tokens are phonemes from tokens.txt

import onnxruntime
import onnxruntime/ort_bindings
import std/[strutils, tables, unicode]

# Re-export types needed by users of this module
export InputTensor, OutputTensor, Model

#------------------------------------------------------------------------------
# Lexicon Loading
#------------------------------------------------------------------------------

type
  LexiconEntry* = object
    ## Entry in the lexicon file
    ## Format: 汉字 phoneme1 phoneme2 ... tone1 tone2 ...
    characters*: string      # Original Chinese characters
    phonemes*: seq[string]   # Phoneme sequence
    tones*: seq[int]         # Tone sequence

proc loadLexicon*(path: string): Table[string, LexiconEntry] =
  ## Load lexicon from file.
  ## Format: 汉字 phoneme1 phoneme2 ... tone1 tone2 ...
  ## Example: 你好 n i h ao 3 3 3 3
  
  let content = readFile(path)
  for line in content.splitLines:
    let parts = strutils.splitWhitespace(line.strip())
    if parts.len < 3:
      continue
    
    # First part is the Chinese character(s)
    let characters = parts[0]
    
    # Find where phonemes end and tones begin (tones are single digits)
    var phonemes: seq[string] = @[]
    var tones: seq[int] = @[]
    
    for i in 1 ..< parts.len:
      let part = parts[i]
      # Check if it's a tone (single digit 0-5)
      if part.len == 1 and part[0] in {'0'..'5'}:
        tones.add(parseInt(part))
      else:
        phonemes.add(part)
    
    result[characters] = LexiconEntry(
      characters: characters,
      phonemes: phonemes,
      tones: tones
    )

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
# Text Processing for Chinese TTS
#------------------------------------------------------------------------------

proc charactersToPhonemesAndTones*(
  text: string, 
  lexicon: Table[string, LexiconEntry]
): tuple[phonemes: seq[string], tones: seq[int64]] =
  ## Convert Chinese characters to phoneme and tone sequences using lexicon.
  ## 
  ## Parameters:
  ##   text: Input Chinese text (e.g., "你好世界")
  ##   lexicon: Character to phoneme mapping
  ##
  ## Returns:
  ##   Tuple of (phoneme sequence, tone sequence)
  
  # Process each character
  for char in text.runes:
    let charStr = $char
    
    if lexicon.hasKey(charStr):
      let entry = lexicon[charStr]
      # Add phonemes for this character
      for phoneme in entry.phonemes:
        result.phonemes.add(phoneme)
      # Add tones (one per phoneme, or default to 0 if no tones)
      # Note: Convert from 1-5 (lexicon format) to 0-4 (model input format)
      if entry.tones.len > 0:
        for tone in entry.tones:
          # Convert 1-5 to 0-4 (subtract 1)
          result.tones.add(max(0, tone - 1).int64)
      else:
        # Default tone if not specified
        for _ in entry.phonemes:
          result.tones.add(0'i64)
    elif charStr == " " or charStr == "\n":
      # Skip whitespace
      discard
    else:
      # Character not in lexicon - could be punctuation or unknown
      result.phonemes.add(charStr)
      result.tones.add(0'i64)  # Default tone

proc phonemesToIds*(
  phonemes: seq[string], 
  tokens: Table[string, int64],
  addBlank: bool = true
): seq[int64] =
  ## Convert phoneme sequence to token IDs.
  ## 
  ## Parameters:
  ##   phonemes: Phoneme sequence
  ##   tokens: Token to ID mapping
  ##   addBlank: Whether to add blank tokens between phonemes (VITS uses this)
  
  let blankId = if tokens.hasKey("_"): tokens["_"] else: 0'i64
  
  for i, phoneme in phonemes:
    # Add blank before each phoneme (except the first)
    if addBlank and i > 0:
      result.add(blankId)
    
    if tokens.hasKey(phoneme):
      result.add(tokens[phoneme])
    elif tokens.hasKey("<unk>"):
      result.add(tokens["<unk>"])
    elif tokens.hasKey("UNK"):
      result.add(tokens["UNK"])
    else:
      # Use blank as fallback
      result.add(blankId)

proc textToTokenIds*(
  text: string, 
  lexicon: Table[string, LexiconEntry],
  tokens: Table[string, int64]
): seq[int64] =
  ## Convert Chinese text to token IDs.
  ## 
  ## This is the main entry point for text processing.
  ## It converts: 汉字 -> 音素(phonemes) -> token IDs
  ##
  ## Parameters:
  ##   text: Input Chinese text
  ##   lexicon: Character to phoneme mapping
  ##   tokens: Phoneme to ID mapping
  
  let (phonemes, tones) = charactersToPhonemesAndTones(text, lexicon)
  result = phonemesToIds(phonemes, tokens, addBlank = false)

#------------------------------------------------------------------------------
# VITS TTS Inference
#------------------------------------------------------------------------------

proc runVitsTTS*(
  model: Model,
  tokenIds: seq[int64],
  sid: int = 0,               # Speaker ID for multi-speaker models
  noiseScale: float32 = 0.667'f32,
  lengthScale: float32 = 1.0'f32,
  noiseScaleW: float32 = 0.8'f32
): OutputTensor =
  ## Run inference on a Sherpa-ONNX VITS TTS model.
  ##
  ## Parameters:
  ##   tokenIds: Token ID sequence from textToTokenIds
  ##   sid: Speaker ID for multi-speaker models (default: 0)
  ##   noiseScale: Noise scale for variance (default: 0.667)
  ##   lengthScale: Length scale for speed (default: 1.0, <1=faster, >1=slower)
  ##   noiseScaleW: Noise width (default: 0.8)
  ##   hasSpeakerId: Whether to include speaker ID input (default: false)
  ##
  ## Returns:
  ##   Output tensor with raw audio samples (float32)
  
  if tokenIds.len == 0:
    raise newException(ValueError, "Token IDs cannot be empty")
  
  let batchSize = 1'i64
  let seqLen = tokenIds.len.int64
  
  var status: OrtStatusPtr
  
  # Create CPU memory info
  var memoryInfo: OrtMemoryInfo
  status = CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)
  checkStatus(status)
  
  # Prepare shapes
  var inputShape = @[batchSize, seqLen]
  var lengthShape = @[batchSize]
  var scalarShape = @[1'i64]  # Shape for scalar values (1D tensor with single element)
  
  # Create input tensor (token IDs) - name: "input"
  var inputOrtValue: OrtValue = nil
  let inputDataSize = tokenIds.len * sizeof(int64)
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    tokenIds[0].unsafeAddr,
    inputDataSize.csize_t,
    inputShape[0].unsafeAddr,
    inputShape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    inputOrtValue.addr
  )
  checkStatus(status)
  
  # Create input_lengths tensor - name: "input_lengths"
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
  
  # Create individual scale tensors for AISHELL3 model
  # noise_scale
  var noiseScaleData = @[noiseScale]
  var noiseScaleOrtValue: OrtValue = nil
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    noiseScaleData[0].unsafeAddr,
    sizeof(float32).csize_t,
    scalarShape[0].unsafeAddr,
    scalarShape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    noiseScaleOrtValue.addr
  )
  checkStatus(status)
  
  # alpha (length_scale)
  var lengthScaleData = @[lengthScale]
  var lengthScaleOrtValue: OrtValue = nil
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    lengthScaleData[0].unsafeAddr,
    sizeof(float32).csize_t,
    scalarShape[0].unsafeAddr,
    scalarShape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    lengthScaleOrtValue.addr
  )
  checkStatus(status)
  
  # noise_scale_dur (noiseScaleW)
  var noiseScaleWData = @[noiseScaleW]
  var noiseScaleWOrtValue: OrtValue = nil
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    noiseScaleWData[0].unsafeAddr,
    sizeof(float32).csize_t,
    scalarShape[0].unsafeAddr,
    scalarShape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    noiseScaleWOrtValue.addr
  )
  checkStatus(status)
  
  # Create sid tensor (speaker ID) - name: "speaker"
  var sidData = @[sid.int64]
  var sidOrtValue: OrtValue = nil
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    sidData[0].unsafeAddr,
    sizeof(int64).csize_t,
    scalarShape[0].unsafeAddr,
    scalarShape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    sidOrtValue.addr
  )
  checkStatus(status)
  
  # Prepare input names and values
  # AISHELL3 icefall model uses different names:
  # - tokens (phoneme IDs)
  # - tokens_lens (sequence lengths)
  # - noise_scale, alpha, noise_scale_dur (scales)
  # - speaker (speaker ID)
  var inputNames: seq[cstring] = @[
    "tokens".cstring,        # phoneme IDs [batch, seq_len]
    "tokens_lens".cstring,   # sequence length [batch]
    "noise_scale".cstring,   # noise scale
    "alpha".cstring,         # length/speed scale
    "noise_scale_dur".cstring, # noise scale duration
    "speaker".cstring        # speaker ID (scalar)
  ]
  var inputs: seq[OrtValue] = @[
    inputOrtValue,       # tokens
    lengthOrtValue,      # tokens_lens
    noiseScaleOrtValue,  # noise_scale
    lengthScaleOrtValue, # alpha (length_scale)
    noiseScaleWOrtValue, # noise_scale_dur
    sidOrtValue          # speaker
  ]
  
  # Run inference
  # VITS output name is "y" not "output"
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
  result = newSeq[int16](output.data.len)
  for i in 0 ..< output.data.len:
    let sample = output.data[i]
    let clamped = max(-1.0'f32, min(1.0'f32, sample))
    result[i] = int16(clamped * 32767.0'f32)

proc sampleCount*(output: OutputTensor): int =
  result = output.data.len

proc sampleRate*(output: OutputTensor; defaultRate: int = 8000): int =
  ## Get the sample rate from output shape or return default.
  ## AISHELL3 uses 8000 Hz, MeloTTS uses 44100 Hz
  result = defaultRate
