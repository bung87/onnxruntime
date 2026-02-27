## whisper_utils.nim
## Whisper ASR model specific utilities
## This is application-level code, not part of the core onnxruntime library

import onnxruntime
import onnxruntime/ort_bindings
import std/[strutils, math, complex, tables]
import fftr

# Re-export types needed by users of this module
export InputTensor, OutputTensor, Model

#------------------------------------------------------------------------------
# Whisper Constants
#------------------------------------------------------------------------------

const
  WHISPER_SAMPLE_RATE* = 16000
  WHISPER_N_FFT* = 400
  WHISPER_N_MELS* = 80
  WHISPER_HOP_LENGTH* = 160
  WHISPER_CHUNK_LENGTH* = 30
  WHISPER_N_FRAMES* = 3000
  MEL_FMIN* = 0.0
  MEL_FMAX* = 8000.0

#------------------------------------------------------------------------------
# Whisper Model Type
#------------------------------------------------------------------------------

type
  WhisperModel* = object
    ## Wrapper for Whisper encoder-decoder model pair
    encoder*: Model
    decoder*: Model

proc loadWhisper*(encoderPath, decoderPath: string): WhisperModel =
  ## Load Whisper encoder and decoder models.
  result.encoder = loadModel(encoderPath)
  result.decoder = loadModel(decoderPath)

proc close*(whisper: WhisperModel) =
  ## Release both encoder and decoder models.
  whisper.encoder.close()
  whisper.decoder.close()

#------------------------------------------------------------------------------
# Audio Processing
#------------------------------------------------------------------------------

proc hertzToMel*(htk: float): float =
  2595.0 * log10(1.0 + htk / 700.0)

proc melToHertz*(mel: float): float =
  700.0 * (pow(10.0, mel / 2595.0) - 1.0)

proc createMelFilterbank*(nFft, nMels, sampleRate: int): seq[float32] =
  ## Create mel filterbank matching transformers' implementation
  ## Returns flat filterbank of shape (nFft//2 + 1, nMels)
  let numFreqBins = nFft div 2 + 1
  
  let melMin = hertzToMel(MEL_FMIN)
  let melMax = hertzToMel(min(MEL_FMAX, sampleRate.float / 2.0))
  
  var filterFreqs = newSeq[float](nMels + 2)
  for i in 0 ..< nMels + 2:
    filterFreqs[i] = melMin + (melMax - melMin) * i.float / (nMels + 1).float
    filterFreqs[i] = melToHertz(filterFreqs[i])
  
  var fftFreqs = newSeq[float](numFreqBins)
  for i in 0 ..< numFreqBins:
    fftFreqs[i] = (sampleRate.float / 2.0) * i.float / (numFreqBins - 1).float
  
  # Flat array: result[freqIdx * nMels + melIdx]
  result = newSeq[float32](numFreqBins * nMels)
  for i in 0 ..< numFreqBins:
    for j in 0 ..< nMels:
      let left = filterFreqs[j]
      let center = filterFreqs[j + 1]
      let right = filterFreqs[j + 2]
      let freq = fftFreqs[i]
      
      var value: float32 = 0.0
      if freq >= left and freq <= center:
        if center != left:
          value = ((freq - left) / (center - left)).float32
      elif freq > center and freq < right:
        if right != center:
          value = ((right - freq) / (right - center)).float32
      result[i * nMels + j] = value

proc loadWavFile*(path: string): seq[float32] =
  ## Load WAV file, properly handling chunks
  let file = open(path, fmRead)
  defer: file.close()
  
  var riffHeader: array[12, uint8]
  if file.readBytes(riffHeader, 0, 12) != 12:
    raise newException(IOError, "Failed to read RIFF header")
  if riffHeader[0] != uint8('R') or riffHeader[1] != uint8('I') or
     riffHeader[2] != uint8('F') or riffHeader[3] != uint8('F'):
    raise newException(IOError, "Not a valid RIFF file")
  if riffHeader[8] != uint8('W') or riffHeader[9] != uint8('A') or
     riffHeader[10] != uint8('V') or riffHeader[11] != uint8('E'):
    raise newException(IOError, "Not a valid WAV file")
  
  var numChannels: int
  var bitsPerSample: int
  var dataOffset: int = -1
  var dataSize: int
  
  while file.getFilePos() < file.getFileSize():
    var chunkId: array[4, uint8]
    if file.readBytes(chunkId, 0, 4) != 4:
      break
    var chunkSizeBytes: array[4, uint8]
    if file.readBytes(chunkSizeBytes, 0, 4) != 4:
      break
    let chunkSize = int(chunkSizeBytes[0]) or (int(chunkSizeBytes[1]) shl 8) or
                    (int(chunkSizeBytes[2]) shl 16) or (int(chunkSizeBytes[3]) shl 24)
    
    let chunkName = $char(chunkId[0]) & $char(chunkId[1]) & $char(chunkId[2]) & $char(chunkId[3])
    
    if chunkName == "fmt ":
      var fmtData: array[16, uint8]
      if file.readBytes(fmtData, 0, chunkSize) != chunkSize:
        raise newException(IOError, "Failed to read fmt chunk")
      numChannels = int(fmtData[2]) or (int(fmtData[3]) shl 8)
      bitsPerSample = int(fmtData[14]) or (int(fmtData[15]) shl 8)
    elif chunkName == "data":
      dataOffset = file.getFilePos()
      dataSize = chunkSize
      break
    else:
      file.setFilePos(chunkSize, fspCur)
  
  if dataOffset < 0:
    raise newException(IOError, "No data chunk found in WAV file")
  if bitsPerSample != 16:
    raise newException(IOError, "Only 16-bit PCM supported")
  
  file.setFilePos(dataOffset)
  let numSamples = dataSize div 2
  result = newSeq[float32](numSamples)
  var buffer: array[16384, uint8]  # 16KB buffer for better I/O throughput
  var sampleIdx = 0
  while sampleIdx < numSamples:
    let toRead = min(16384, (numSamples - sampleIdx) * 2)
    let bytesRead = file.readBytes(buffer, 0, toRead)
    if bytesRead == 0:
      break
    var i = 0
    while i < bytesRead and sampleIdx < numSamples:
      let sample = int16(buffer[i]) or (int16(buffer[i+1]) shl 8)
      result[sampleIdx] = sample.float32 / 32768.0'f32
      sampleIdx += 1
      i += 2
  
  if numChannels == 2:
    var monoSamples = newSeq[float32](numSamples div 2)
    for i in 0 ..< numSamples div 2:
      monoSamples[i] = (result[i*2] + result[i*2+1]) / 2.0'f32
    result = monoSamples

proc padOrTrimAudio*(audio: seq[float32], targetSamples: int): seq[float32] =
  ## Pad or trim audio to exactly targetSamples
  result = newSeq[float32](targetSamples)
  let samplesToCopy = min(audio.len, targetSamples)
  for i in 0 ..< samplesToCopy:
    result[i] = audio[i]

proc computeStft*(audio: seq[float32], nFft, hopLength: int): seq[seq[Complex[float32]]] =
  ## Compute STFT with center=True padding (reflect mode)
  let padLeft = nFft div 2
  let padRight = nFft div 2
  
  var paddedAudio = newSeq[float64](audio.len + padLeft + padRight)
  for i in 0 ..< audio.len + padLeft + padRight:
    if i < padLeft:
      let idx = padLeft - i
      paddedAudio[i] = if idx < audio.len: audio[idx].float64 else: 0.0
    elif i >= padLeft + audio.len:
      let idx = audio.len - 2 - (i - padLeft - audio.len)
      paddedAudio[i] = if idx >= 0: audio[idx].float64 else: 0.0
    else:
      paddedAudio[i] = audio[i - padLeft].float64
  
  let nFrames = (paddedAudio.len - nFft) div hopLength + 1
  result = newSeq[seq[Complex[float32]]](nFrames)
  
  var window = newSeq[float64](nFft)
  for i in 0 ..< nFft:
    window[i] = 0.5 - 0.5 * cos(2.0 * PI * i.float64 / (nFft - 1).float64)
  
  for frame in 0 ..< nFrames:
    result[frame] = newSeq[Complex[float32]](nFft div 2 + 1)
    let start = frame * hopLength
    var frameData = newSeq[Complex[float64]](nFft)
    for i in 0 ..< nFft:
      frameData[i] = complex(paddedAudio[start + i] * window[i], 0.0)
    
    let fftResult = fft(frameData, false)
    for k in 0 ..< (nFft div 2 + 1):
      result[frame][k] = complex(fftResult[k].re.float32, fftResult[k].im.float32)

proc computeWhisperMelSpectrogram*(audio: seq[float32]): seq[float32] =
  ## Compute mel spectrogram matching Whisper's implementation
  ## 1. STFT with hann window
  ## 2. Power spectrogram (|x|^2)
  ## 3. Apply mel filterbank
  ## 4. log10
  ## 5. Discard last frame (log_spec[:, :-1])
  ## 6. Normalize: max(log_spec, max - 8.0), then (log_spec + 4.0) / 4.0
  
  let stft = computeStft(audio, WHISPER_N_FFT, WHISPER_HOP_LENGTH)
  let nFramesTotal = stft.len
  let nFreqBins = WHISPER_N_FFT div 2 + 1
  
  # Compute power spectrogram: |x|^2 (flat array)
  # powerSpec[freqIdx * nFramesTotal + frameIdx]
  var powerSpec = newSeq[float64](nFreqBins * nFramesTotal)
  for f in 0 ..< nFreqBins:
    for t in 0 ..< nFramesTotal:
      let mag = abs(stft[t][f])
      powerSpec[f * nFramesTotal + t] = (mag * mag).float64
  
  # Apply mel filterbank
  let melFilter = createMelFilterbank(WHISPER_N_FFT, WHISPER_N_MELS, WHISPER_SAMPLE_RATE)
  # melSpec[melIdx * nFramesTotal + frameIdx]
  var melSpec = newSeq[float64](WHISPER_N_MELS * nFramesTotal)
  for m in 0 ..< WHISPER_N_MELS:
    for t in 0 ..< nFramesTotal:
      var sum: float64 = 0.0
      for f in 0 ..< nFreqBins:
        sum += powerSpec[f * nFramesTotal + t] * melFilter[f * WHISPER_N_MELS + m].float64
      melSpec[m * nFramesTotal + t] = max(sum, 1e-10)
  
  # log10
  for m in 0 ..< WHISPER_N_MELS:
    for t in 0 ..< nFramesTotal:
      melSpec[m * nFramesTotal + t] = log10(melSpec[m * nFramesTotal + t])
  
  # Discard last frame: log_spec[:, :-1]
  let nFrames = nFramesTotal - 1
  
  # Whisper normalization
  var globalMax = -1e300
  for m in 0 ..< WHISPER_N_MELS:
    for t in 0 ..< nFrames:
      globalMax = max(globalMax, melSpec[m * nFramesTotal + t])
  
  result = newSeq[float32](WHISPER_N_MELS * nFrames)
  for m in 0 ..< WHISPER_N_MELS:
    for t in 0 ..< nFrames:
      var val = max(melSpec[m * nFramesTotal + t], globalMax - 8.0)
      val = (val + 4.0) / 4.0
      result[m * nFrames + t] = val.float32

#------------------------------------------------------------------------------
# Encoder/Decoder Inference
#------------------------------------------------------------------------------

proc runEncoder*(whisper: WhisperModel, melSpectrogram: seq[float32]): OrtValue =
  ## Run Whisper encoder on mel spectrogram.
  ## Returns encoder output as OrtValue that must be released by the caller
  ## using `ReleaseValue()` when no longer needed to prevent memory leaks.
  var status: OrtStatusPtr

  var memoryInfo: OrtMemoryInfo
  status = CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)
  checkStatus(status)

  var inputShape = @[1'i64, WHISPER_N_MELS.int64, 3000'i64]
  var encoderInput: OrtValue
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    melSpectrogram[0].unsafeAddr,
    (melSpectrogram.len * sizeof(float32)).csize_t,
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
  status = SessionGetInputName(getSession(whisper.encoder.internal), 0, allocator, encInputNamePtr.addr)
  checkStatus(status)
  status = SessionGetOutputName(getSession(whisper.encoder.internal), 0, allocator, encOutputNamePtr.addr)
  checkStatus(status)

  var encInputNames = @[($encInputNamePtr).cstring]
  var encOutputNames = @[($encOutputNamePtr).cstring]

  var encoderOutput: OrtValue
  status = Run(
    getSession(whisper.encoder.internal),
    nil,
    encInputNames[0].addr,
    encoderInput.addr,
    1,
    encOutputNames[0].addr,
    1,
    encoderOutput.addr
  )
  checkStatus(status)

  result = encoderOutput

proc runDecoderStep*(
  whisper: WhisperModel,
  inputIds: seq[int64],
  encoderOutput: OrtValue,
  vocabSize: int = 51865
): int64 =
  ## Run one decoder step and return next token ID.
  var status: OrtStatusPtr

  var memoryInfo: OrtMemoryInfo
  status = CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)
  checkStatus(status)

  var allocator: OrtAllocator
  status = GetAllocatorWithDefaultOptions(allocator.addr)
  checkStatus(status)

  var decInputCount, decOutputCount: csize_t
  status = SessionGetInputCount(getSession(whisper.decoder.internal), decInputCount.addr)
  checkStatus(status)
  status = SessionGetOutputCount(getSession(whisper.decoder.internal), decOutputCount.addr)
  checkStatus(status)

  var decInputNamePtr0, decInputNamePtr1: cstring
  status = SessionGetInputName(getSession(whisper.decoder.internal), 0, allocator, decInputNamePtr0.addr)
  checkStatus(status)
  if decInputCount > 1:
    status = SessionGetInputName(getSession(whisper.decoder.internal), 1, allocator, decInputNamePtr1.addr)
    checkStatus(status)

  var decOutputNamePtr: cstring
  status = SessionGetOutputName(getSession(whisper.decoder.internal), 0, allocator, decOutputNamePtr.addr)
  checkStatus(status)

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

  var decInputNames = @[($decInputNamePtr0).cstring]
  var decInputValues = @[inputIdsValue]

  if decInputCount > 1 and decInputNamePtr1 != nil and ($decInputNamePtr1).contains("encoder"):
    decInputNames.add(($decInputNamePtr1).cstring)
    decInputValues.add(encoderOutput)

  var logitsValue: OrtValue
  status = Run(
    getSession(whisper.decoder.internal),
    nil,
    decInputNames[0].addr,
    decInputValues[0].addr,
    decInputNames.len.csize_t,
    decOutputNamePtr.addr,
    1,
    logitsValue.addr
  )
  checkStatus(status)

  var logitsPtr: ptr float32
  status = GetTensorMutableData(logitsValue, cast[ptr pointer](logitsPtr.addr))
  checkStatus(status)

  let lastPosLogits = cast[ptr float32](cast[uint64](logitsPtr) + uint64((inputIds.len - 1) * vocabSize * sizeof(float32)))

  var maxLogit = -Inf.float32
  var nextToken: int64 = 0
  for i in 0 ..< vocabSize:
    let logit = (cast[ptr UncheckedArray[float32]](lastPosLogits))[i]
    if logit > maxLogit:
      maxLogit = logit
      nextToken = i.int64

  result = nextToken

#------------------------------------------------------------------------------
# Token Decoding (Whisper Byte-Level BPE)
#------------------------------------------------------------------------------

# Byte-to-Unicode mapping for GPT-2/Whisper tokenizer
var ByteToUnicode: seq[string]
var UnicodeToByte: Table[string, byte]
var ByteToUnicodeInitialized = false

proc initByteToUnicode() =
  ## Initialize ByteToUnicode mapping for proper token decoding
  if ByteToUnicodeInitialized:
    return
  
  ByteToUnicode = newSeq[string](256)
  UnicodeToByte = initTable[string, byte]()
  
  # Printable ASCII characters (33-126) map to themselves
  for b in 33 .. 126:
    ByteToUnicode[b] = $chr(b)
  
  # Latin-1 supplement characters
  let latin1Chars = [
    (161, "¡"), (162, "¢"), (163, "£"), (164, "¤"), (165, "¥"),
    (166, "¦"), (167, "§"), (168, "¨"), (169, "©"), (170, "ª"),
    (171, "«"), (172, "¬"), (174, "®"), (175, "¯"), (176, "°"),
    (177, "±"), (178, "²"), (179, "³"), (180, "´"), (181, "µ"),
    (182, "¶"), (183, "·"), (184, "¸"), (185, "¹"), (186, "º"),
    (187, "»"), (188, "¼"), (189, "½"), (190, "¾"), (191, "¿"),
    (192, "À"), (193, "Á"), (194, "Â"), (195, "Ã"), (196, "Ä"),
    (197, "Å"), (198, "Æ"), (199, "Ç"), (200, "È"), (201, "É"),
    (202, "Ê"), (203, "Ë"), (204, "Ì"), (205, "Í"), (206, "Î"),
    (207, "Ï"), (208, "Ð"), (209, "Ñ"), (210, "Ò"), (211, "Ó"),
    (212, "Ô"), (213, "Õ"), (214, "Ö"), (215, "×"), (216, "Ø"),
    (217, "Ù"), (218, "Ú"), (219, "Û"), (220, "Ü"), (221, "Ý"),
    (222, "Þ"), (223, "ß"), (224, "à"), (225, "á"), (226, "â"),
    (227, "ã"), (228, "ä"), (229, "å"), (230, "æ"), (231, "ç"),
    (232, "è"), (233, "é"), (234, "ê"), (235, "ë"), (236, "ì"),
    (237, "í"), (238, "î"), (239, "ï"), (240, "ð"), (241, "ñ"),
    (242, "ò"), (243, "ó"), (244, "ô"), (245, "õ"), (246, "ö"),
    (247, "÷"), (248, "ø"), (249, "ù"), (250, "ú"), (251, "û"),
    (252, "ü"), (253, "ý"), (254, "þ"), (255, "ÿ")
  ]
  
  for (byteVal, charStr) in latin1Chars:
    ByteToUnicode[byteVal] = charStr
  
  # Remaining bytes map to Unicode characters starting at 0x0100
  var nextUnicode = 0x0100
  for b in 0 .. 255:
    if ByteToUnicode[b].len == 0:
      if nextUnicode <= 0x7F:
        ByteToUnicode[b] = $chr(nextUnicode)
      elif nextUnicode <= 0x7FF:
        var s = newString(2)
        s[0] = chr(0xC0 or ((nextUnicode shr 6) and 0x1F))
        s[1] = chr(0x80 or (nextUnicode and 0x3F))
        ByteToUnicode[b] = s
      elif nextUnicode <= 0xFFFF:
        var s = newString(3)
        s[0] = chr(0xE0 or ((nextUnicode shr 12) and 0x0F))
        s[1] = chr(0x80 or ((nextUnicode shr 6) and 0x3F))
        s[2] = chr(0x80 or (nextUnicode and 0x3F))
        ByteToUnicode[b] = s
      else:
        var s = newString(4)
        s[0] = chr(0xF0 or ((nextUnicode shr 18) and 0x07))
        s[1] = chr(0x80 or ((nextUnicode shr 12) and 0x3F))
        s[2] = chr(0x80 or ((nextUnicode shr 6) and 0x3F))
        s[3] = chr(0x80 or (nextUnicode and 0x3F))
        ByteToUnicode[b] = s
      nextUnicode.inc
  
  # Build reverse mapping
  for b in 0 .. 255:
    UnicodeToByte[ByteToUnicode[b]] = b.byte
  
  ByteToUnicodeInitialized = true

proc decodeTokenToBytes(token: string): seq[byte] =
  ## Decode a token string back to bytes using ByteToUnicode mapping
  if token.len == 0:
    return @[]
  
  if not ByteToUnicodeInitialized:
    initByteToUnicode()
  
  result = @[]
  var i = 0
  while i < token.len:
    var charStr: string
    let c = token[i]
    let ordC = ord(c)
    
    # Parse UTF-8 character
    if ordC < 0x80:
      charStr = $c
      i += 1
    elif (ordC and 0xE0) == 0xC0 and i + 1 < token.len:
      charStr = token[i .. i+1]
      i += 2
    elif (ordC and 0xF0) == 0xE0 and i + 2 < token.len:
      charStr = token[i .. i+2]
      i += 3
    elif (ordC and 0xF8) == 0xF0 and i + 3 < token.len:
      charStr = token[i .. i+3]
      i += 4
    else:
      charStr = $c
      i += 1
    
    # Look up byte value
    if UnicodeToByte.hasKey(charStr):
      result.add(UnicodeToByte[charStr])
    else:
      # If not in mapping, just use the character directly
      for b in charStr:
        result.add(ord(b).byte)

proc isCompleteUtf8(bytes: seq[byte]): tuple[complete: bool, consumed: int] =
  ## Check if bytes form a complete UTF-8 sequence
  if bytes.len == 0:
    return (true, 0)
  
  let first = bytes[0].int
  if first < 0x80:
    return (true, 1)
  elif (first and 0xE0) == 0xC0:
    if bytes.len < 2 or (bytes[1].int and 0xC0) != 0x80:
      return (false, 0)
    return (true, 2)
  elif (first and 0xF0) == 0xE0:
    if bytes.len < 3 or (bytes[1].int and 0xC0) != 0x80 or (bytes[2].int and 0xC0) != 0x80:
      return (false, 0)
    return (true, 3)
  elif (first and 0xF8) == 0xF0:
    if bytes.len < 4 or (bytes[1].int and 0xC0) != 0x80 or (bytes[2].int and 0xC0) != 0x80 or (bytes[3].int and 0xC0) != 0x80:
      return (false, 0)
    return (true, 4)
  else:
    return (true, 1)

proc decodeTokensToText*(tokenIds: seq[int64], idToToken: seq[string]): string =
  ## Decode token IDs to text using the provided id-to-token mapping.
  ## Handles ByteToUnicode mapping for proper UTF-8 output.
  result = ""
  var pendingBytes: seq[byte] = @[]
  
  for tokenId in tokenIds:
    if tokenId < 0 or tokenId >= idToToken.len:
      continue
    
    let token = idToToken[tokenId.int]
    if token.len == 0:
      continue
    
    # Skip special tokens
    if token.startsWith("<|") and token.endsWith("|>"):
      continue
    
    # Decode token to bytes
    let tokenBytes = decodeTokenToBytes(token)
    for b in tokenBytes:
      pendingBytes.add(b)
    
    # Try to form complete UTF-8 characters
    while true:
      let (complete, consumed) = isCompleteUtf8(pendingBytes)
      if complete and consumed > 0:
        var validSeq = newString(consumed)
        for i in 0 ..< consumed:
          validSeq[i] = chr(pendingBytes[i].int)
        result.add(validSeq)
        pendingBytes = pendingBytes[consumed .. ^1]
      else:
        break
  
  # Handle any remaining bytes
  if pendingBytes.len > 0:
    for b in pendingBytes:
      if b < 128:
        result.add(chr(b.int))
