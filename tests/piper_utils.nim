## piper_utils.nim
## Piper TTS model specific utilities
## This is application-level code, not part of the core onnxruntime library

import onnxruntime
import onnxruntime/ort_bindings

# Re-export types needed by users of this module
export InputTensor, OutputTensor, Model

#------------------------------------------------------------------------------
# Piper TTS Inference
#------------------------------------------------------------------------------

proc runPiper*(
  model: Model,
  phonemeIds: seq[int64],
  noiseScale: float32 = 0.667'f32,
  lengthScale: float32 = 1.0'f32,
  noiseW: float32 = 0.8'f32,
  speakerId: int = 0,
  hasSpeakerId: bool = false
): OutputTensor =
  ## Run inference on a Piper TTS model.
  ##
  ## Parameters:
  ##   phonemeIds: Phoneme ID sequence
  ##   noiseScale: Noise scale for variance (default: 0.667)
  ##   lengthScale: Length scale for speed (default: 1.0)
  ##   noiseW: Noise width (default: 0.8)
  ##   speakerId: Speaker ID for multi-speaker models (default: 0)
  ##   hasSpeakerId: Whether to include speaker ID input (default: false)
  ##
  ## Returns:
  ##   Output tensor with raw audio samples
  ##
  ## Raises:
  ##   OnnxRuntimeError: If inference fails
  
  let batchSize = 1'i64
  let seqLen = phonemeIds.len.int64
  
  var status: OrtStatusPtr
  
  # Create CPU memory info
  var memoryInfo: OrtMemoryInfo
  status = CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)
  checkStatus(status)
  
  # Prepare shapes
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
  var scalesData = @[noiseScale, lengthScale, noiseW]
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
  var inputNames: seq[cstring] = @["input".cstring, "input_lengths".cstring, "scales".cstring]
  var inputs: seq[OrtValue] = @[inputOrtValue, lengthOrtValue, scalesOrtValue]
  
  # Add speaker ID if multi-speaker model
  var sidOrtValue: OrtValue = nil
  if hasSpeakerId:
    var sidData = @[speakerId.int64]
    status = CreateTensorWithDataAsOrtValue(
      memoryInfo,
      sidData[0].unsafeAddr,
      sizeof(int64).csize_t,
      lengthShape[0].unsafeAddr,
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
  ##
  ## Parameters:
  ##   output: Raw model output with float32 samples in [-1, 1] range
  ##
  ## Returns:
  ##   Audio samples converted to int16 format
  result = newSeq[int16](output.data.len)
  for i in 0 ..< output.data.len:
    let sample = output.data[i]
    let clamped = max(-1.0'f32, min(1.0'f32, sample))
    result[i] = int16(clamped * 32767.0'f32)

proc sampleCount*(output: OutputTensor): int =
  ## Get the number of audio samples.
  result = output.data.len
