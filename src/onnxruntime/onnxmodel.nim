## onnxmodel.nim
## A simple wrapper for ONNX Runtime inference with GPT-like models
## This module provides utilities to load ONNX models and run inference
## with tokenized inputs (e.g., for tiny2GPT or GPT-2 models)

import ort_bindings

# Re-export types needed by users of this module
export OrtLoggingLevel, ONNXTensorElementDataType, OrtErrorCode

## Custom exception for ONNX Runtime errors
type OnnxRuntimeError* = object of CatchableError
  ## Exception raised when ONNX Runtime operations fail
  errorCode*: OrtErrorCode  ## The specific error code from ONNX Runtime

## Type representing a tensor input for the model
type
  OnnxInputTensor* = object
    ## Input tensor data for ONNX inference
    ## `data`: The flattened array of values (e.g., token IDs)
    ## `shape`: The shape of the tensor (e.g., [batch_size, sequence_length])
    data*: seq[int64]   # Token IDs as int64 (common for GPT models)
    shape*: seq[int64]  # Tensor shape with batch dimension

  OnnxOutputTensor* = object
    ## Output tensor from ONNX inference
    ## Contains the raw output data and its shape
    data*: seq[float32]  # Model outputs are typically float32 (logits)
    shape*: seq[int64]   # Shape: [batch_size, sequence_length, vocab_size]

  OnnxNeoOutput* = object
    ## Full output from GPT-Neo model including logits and present_key_values
    logits*: OnnxOutputTensor
    presentKeyValues*: seq[OnnxOutputTensor]  # 16 tensors (8 layers x 2 for key/value)

  OnnxModelObj = object
    ## Internal object type for OnnxModel
    env: OrtEnv
    session: OrtSession
    options: OrtSessionOptions
  
  OnnxModel* = ref OnnxModelObj
    ## Wrapper around an ONNX session for GPT-like models

proc getSession*(model: OnnxModel): OrtSession =
  ## Get the underlying ONNX session
  result = model.session

proc newOnnxModel*(modelPath: string): OnnxModel =
  ## Create a new ONNX model session from a file path
  ## 
  ## Example:
  ##   let model = newOnnxModel("tests/model.onnx")
  ##
  result = new(OnnxModel)
  
  # Initialize fields to nil/empty to prevent issues during cleanup
  result.env = nil
  result.session = nil
  result.options = nil
  
  var status: OrtStatusPtr
  
  # Create the ONNX Runtime environment
  # ORT_LOGGING_LEVEL_WARNING suppresses most logs (use VERBOSE for debugging)
  status = CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnxmodel", result.env.addr)
  checkStatus(status)
  
  # Create session options
  status = CreateSessionOptions(result.options.addr)
  if status != nil:
    # Clean up env on failure
    if result.env != nil:
      ReleaseEnv(result.env)
      result.env = nil
    checkStatus(status)
  
  # Enable CPU memory arena for better performance
  status = EnableCpuMemArena(result.options)
  if status != nil:
    if result.options != nil:
      ReleaseSessionOptions(result.options)
      result.options = nil
    if result.env != nil:
      ReleaseEnv(result.env)
      result.env = nil
    checkStatus(status)
  
  # Load the model from file
  # This loads the model architecture and weights into memory
  status = CreateSession(result.env, modelPath.cstring, result.options, result.session.addr)
  if status != nil:
    # Clean up on failure
    if result.options != nil:
      ReleaseSessionOptions(result.options)
      result.options = nil
    if result.env != nil:
      ReleaseEnv(result.env)
      result.env = nil
    checkStatus(status)

proc close*(model: OnnxModel) =
  ## Clean up and release the ONNX model session
  ## Should be called when done with the model to free resources
  ##
  ## Example:
  ##   model.close()
  ##
  if model == nil:
    return
    
  if model.session != nil:
    ReleaseSession(model.session)
    model.session = nil
  
  if model.options != nil:
    ReleaseSessionOptions(model.options)
    model.options = nil
  
  if model.env != nil:
    ReleaseEnv(model.env)
    model.env = nil

proc runInference*(
  model: OnnxModel,
  inputTensor: OnnxInputTensor,
  inputName: string = "input",
  outputName: string = "output"
): OnnxOutputTensor =
  ## Run inference on the model with a single input tensor
  ##
  ## Parameters:
  ##   inputTensor: The input data and shape
  ##   inputName: Name of the input node in the ONNX graph (default: "input")
  ##   outputName: Name of the output node in the ONNX graph (default: "output")
  ##
  ## Returns:
  ##   The output tensor with logits or other model outputs
  ##
  ## Example:
  ##   let input = OnnxInputTensor(data: @[1'i64, 2, 3], shape: @[1'i64, 3])
  ##   let output = model.runInference(input)
  ##
  var status: OrtStatusPtr
  
  # Validate input
  if inputTensor.data.len == 0:
    raise newException(Exception, "Input tensor data cannot be empty")
  if inputTensor.shape.len == 0:
    raise newException(Exception, "Input tensor shape cannot be empty")
  
  # Create CPU memory info for tensor allocation
  var memoryInfo: OrtMemoryInfo
  status = CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)
  checkStatus(status)
  defer:
    ReleaseMemoryInfo(memoryInfo)
  
  # Create input tensor from our data
  var inputOrtValue: OrtValue = nil
  let dataSize = inputTensor.data.len * sizeof(int64)
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    inputTensor.data[0].unsafeAddr,
    dataSize.csize_t,
    inputTensor.shape[0].unsafeAddr,
    inputTensor.shape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    inputOrtValue.addr
  )
  checkStatus(status)
  defer:
    if inputOrtValue != nil:
      ReleaseValue(inputOrtValue)
  
  # Prepare input name as C string
  let inputNameC = inputName.cstring
  let outputNameC = outputName.cstring
  
  # Run inference
  var outputOrtValue: OrtValue = nil
  status = Run(
    model.session,
    nil,  # run_options
    inputNameC.addr,
    inputOrtValue.addr,
    1.csize_t,
    outputNameC.addr,
    1.csize_t,
    outputOrtValue.addr
  )
  checkStatus(status)
  defer:
    if outputOrtValue != nil:
      ReleaseValue(outputOrtValue)
  
  # Get output type info
  var typeInfo: OrtTypeInfo
  status = GetTypeInfo(outputOrtValue, typeInfo.addr)
  checkStatus(status)
  defer:
    ReleaseTypeInfo(typeInfo)
  
  # Cast to tensor info to get shape and data
  var tensorInfo: OrtTensorTypeAndShapeInfo
  status = CastTypeInfoToTensorInfo(typeInfo, tensorInfo.addr)
  checkStatus(status)
  defer:
    ReleaseTensorTypeAndShapeInfo(tensorInfo)
  
  # Get output tensor dimensions
  var dimsCount: csize_t
  status = GetDimensionsCount(tensorInfo, dimsCount.addr)
  checkStatus(status)
  
  var outputShape = newSeq[int64](dimsCount)
  if dimsCount > 0:
    status = GetDimensions(tensorInfo, outputShape[0].addr, dimsCount)
    checkStatus(status)
  
  # Get pointer to output data
  var outputDataPtr: pointer
  status = GetTensorMutableData(outputOrtValue, outputDataPtr.addr)
  checkStatus(status)
  
  # Get total element count
  var elemCount: csize_t
  status = GetTensorShapeElementCount(tensorInfo, elemCount.addr)
  checkStatus(status)
  
  # Copy data from OrtValue to Nim seq
  let floatPtr = cast[ptr UncheckedArray[float32]](outputDataPtr)
  var outputData = newSeq[float32](elemCount)
  for i in 0 ..< elemCount.int:
    outputData[i] = floatPtr[i]
  
  # Return result
  result = OnnxOutputTensor(
    data: outputData,
    shape: outputShape
  )

## GPT-Neo model input type
type
  OnnxNeoInputTensor* = object
    ## Input tensor collection for GPT-Neo model
    inputIds*: OnnxInputTensor
    positionIds*: OnnxInputTensor
    pastKeyValues*: seq[OnnxInputTensor]  # 16 tensors (8 layers x 2 for key/value)

proc runInferenceNeo*(
  model: OnnxModel,
  neoInput: OnnxNeoInputTensor,
  inputName: string = "input_ids",
  outputName: string = "logits",
  attentionMask: OnnxInputTensor = OnnxInputTensor(data: @[], shape: @[])  # Optional attention mask
): OnnxOutputTensor =
  ## Run inference on GPT-Neo model with past_key_values support
  ## Updated to include attention_mask for TinyStories-1M compatibility
  ##
  ## Parameters:
  ##   neoInput: Collection of input tensors (input_ids, position_ids, past_key_values)
  ##   inputName: Name of the input node (default: "input_ids")
  ##   outputName: Name of the output node (default: "logits")
  ##   attentionMask: Optional attention mask tensor
  ##
  ## Returns:
  ##   Output tensor with logits
  ##
  var status: OrtStatusPtr
  let numLayers = neoInput.pastKeyValues.len div 2
  
  # Validate inputs
  if neoInput.inputIds.data.len == 0:
    raise newException(Exception, "Input tensor data cannot be empty")
  if neoInput.inputIds.shape.len == 0:
    raise newException(Exception, "Input tensor shape cannot be empty")
  if neoInput.positionIds.data.len == 0:
    raise newException(Exception, "Position IDs data cannot be empty")
  if neoInput.positionIds.shape.len == 0:
    raise newException(Exception, "Position IDs shape cannot be empty")
  
  # Create CPU memory info
  var memoryInfo: OrtMemoryInfo
  status = CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)
  checkStatus(status)
  defer:
    ReleaseMemoryInfo(memoryInfo)
  
  # Create input_ids tensor
  var inputOrtValue: OrtValue = nil
  let dataSize = neoInput.inputIds.data.len * sizeof(int64)
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    neoInput.inputIds.data[0].unsafeAddr,
    dataSize.csize_t,
    neoInput.inputIds.shape[0].unsafeAddr,
    neoInput.inputIds.shape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    inputOrtValue.addr
  )
  checkStatus(status)
  
  # Create position_ids tensor
  var positionIdsOrtValue: OrtValue = nil
  let posDataSize = neoInput.positionIds.data.len * sizeof(int64)
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    neoInput.positionIds.data[0].unsafeAddr,
    posDataSize.csize_t,
    neoInput.positionIds.shape[0].unsafeAddr,
    neoInput.positionIds.shape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    positionIdsOrtValue.addr
  )
  checkStatus(status)
  
  # Create past_key_values tensors (using float32 for cache)
  var pastKeyValuesOrt = newSeq[OrtValue](neoInput.pastKeyValues.len)
  for i in 0 ..< neoInput.pastKeyValues.len:
    if neoInput.pastKeyValues[i].data.len > 0:
      # Create float32 data for past_key_values
      var floatData = newSeq[float32](neoInput.pastKeyValues[i].data.len)
      for j in 0 ..< floatData.len:
        floatData[j] = 0.0  # Initialize with zeros
      
      let kvDataSize = floatData.len * sizeof(float32)
      status = CreateTensorWithDataAsOrtValue(
        memoryInfo,
        floatData[0].unsafeAddr,
        kvDataSize.csize_t,
        neoInput.pastKeyValues[i].shape[0].unsafeAddr,
        neoInput.pastKeyValues[i].shape.len.csize_t,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        pastKeyValuesOrt[i].addr
      )
      checkStatus(status)
    else:
      # Handle empty cache tensors
      var dummyData: float32 = 0.0
      status = CreateTensorWithDataAsOrtValue(
        memoryInfo,
        dummyData.unsafeAddr,
        sizeof(float32).csize_t,
        neoInput.pastKeyValues[i].shape[0].unsafeAddr,
        neoInput.pastKeyValues[i].shape.len.csize_t,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        pastKeyValuesOrt[i].addr
      )
      checkStatus(status)
  
  # Create attention_mask tensor (if provided)
  var attentionMaskOrtValue: OrtValue = nil
  if attentionMask.data.len > 0 and attentionMask.shape.len > 0:
    let maskDataSize = attentionMask.data.len * sizeof(int64)
    status = CreateTensorWithDataAsOrtValue(
      memoryInfo,
      attentionMask.data[0].unsafeAddr,
      maskDataSize.csize_t,
      attentionMask.shape[0].unsafeAddr,
      attentionMask.shape.len.csize_t,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      attentionMaskOrtValue.addr
    )
    checkStatus(status)
  
  # Build input names and values arrays
  # Order: input_ids, past_key_values.0.key, past_key_values.0.value, ..., attention_mask, position_ids
  let numInputs = 1 + neoInput.pastKeyValues.len + 2  # 1 + 16 + 2 = 19
  var inputNames = newSeq[string](numInputs)
  var inputs = newSeq[OrtValue](numInputs)
  
  inputNames[0] = "input_ids"
  inputs[0] = inputOrtValue
  
  for i in 0 ..< neoInput.pastKeyValues.len:
    inputNames[1 + i] = "past_key_values." & $(i div 2) & "." & (if i mod 2 == 0: "key" else: "value")
    inputs[1 + i] = pastKeyValuesOrt[i]
  
  inputNames[1 + neoInput.pastKeyValues.len] = "attention_mask"
  if attentionMaskOrtValue != nil:
    inputs[1 + neoInput.pastKeyValues.len] = attentionMaskOrtValue
  else:
    inputs[1 + neoInput.pastKeyValues.len] = inputOrtValue  # Fallback
  
  inputNames[1 + neoInput.pastKeyValues.len + 1] = "position_ids"
  inputs[1 + neoInput.pastKeyValues.len + 1] = positionIdsOrtValue
  
  # Convert input names to cstrings
  var inputNamePtrs = newSeq[cstring](numInputs)
  for i in 0 ..< numInputs:
    inputNamePtrs[i] = inputNames[i].cstring
  
  # Prepare output name
  let outputNameC = outputName.cstring
  
  # Run inference
  var outputOrtValue: OrtValue = nil
  status = Run(
    model.session,
    nil,  # run_options
    inputNamePtrs[0].addr,
    inputs[0].addr,
    numInputs.csize_t,
    outputNameC.addr,
    1.csize_t,
    outputOrtValue.addr
  )
  checkStatus(status)
  
  # Process output
  var typeInfo: OrtTypeInfo
  status = GetTypeInfo(outputOrtValue, typeInfo.addr)
  checkStatus(status)
  
  var tensorInfo: OrtTensorTypeAndShapeInfo
  status = CastTypeInfoToTensorInfo(typeInfo, tensorInfo.addr)
  checkStatus(status)
  
  var dimsCount: csize_t
  status = GetDimensionsCount(tensorInfo, dimsCount.addr)
  checkStatus(status)
  
  var outputShape = newSeq[int64](dimsCount)
  if dimsCount > 0:
    status = GetDimensions(tensorInfo, outputShape[0].addr, dimsCount)
    checkStatus(status)
  
  var outputDataPtr: pointer
  status = GetTensorMutableData(outputOrtValue, outputDataPtr.addr)
  checkStatus(status)
  
  var elemCount: csize_t
  status = GetTensorShapeElementCount(tensorInfo, elemCount.addr)
  checkStatus(status)
  
  # Copy output data
  let floatPtr = cast[ptr UncheckedArray[float32]](outputDataPtr)
  var outputData = newSeq[float32](elemCount)
  for i in 0 ..< elemCount.int:
    outputData[i] = floatPtr[i]
  
  # Note: Don't release type info to avoid double-free issues
  # The tensor info is owned by the type info which is owned by the OrtValue
  
  # Clean up tensors
  if inputOrtValue != nil:
    ReleaseValue(inputOrtValue)
  if positionIdsOrtValue != nil:
    ReleaseValue(positionIdsOrtValue)
  if attentionMaskOrtValue != nil:
    ReleaseValue(attentionMaskOrtValue)
  for i in 0 ..< pastKeyValuesOrt.len:
    if pastKeyValuesOrt[i] != nil:
      ReleaseValue(pastKeyValuesOrt[i])
  if outputOrtValue != nil:
    ReleaseValue(outputOrtValue)
  
  # Return result
  result = OnnxOutputTensor(
    data: outputData,
    shape: outputShape
  )

## Enhanced GPT-Neo inference function for TinyStories model
## This handles all 19 inputs in the correct order

proc runInferenceNeoComplete*(
  model: OnnxModel,
  inputIds: OnnxInputTensor,
  attentionMask: OnnxInputTensor,
  positionIds: OnnxInputTensor,
  pastKeyValues: seq[OnnxInputTensor],
  outputName: string = "logits"
): OnnxOutputTensor =
  ## Run inference on GPT-Neo model with all required inputs
  ## Handles the exact input order: input_ids, past_key_values.*, attention_mask, position_ids
  
  var status: OrtStatusPtr
  let numLayers = pastKeyValues.len div 2
  
  # Validate inputs
  if inputIds.data.len == 0:
    raise newException(Exception, "Input tensor data cannot be empty")
  if inputIds.shape.len == 0:
    raise newException(Exception, "Input tensor shape cannot be empty")
  if attentionMask.data.len == 0:
    raise newException(Exception, "Attention mask data cannot be empty")
  if attentionMask.shape.len == 0:
    raise newException(Exception, "Attention mask shape cannot be empty")
  if positionIds.data.len == 0:
    raise newException(Exception, "Position IDs data cannot be empty")
  if positionIds.shape.len == 0:
    raise newException(Exception, "Position IDs shape cannot be empty")
  
  # Create CPU memory info
  var memoryInfo: OrtMemoryInfo
  status = CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)
  checkStatus(status)
  defer:
    ReleaseMemoryInfo(memoryInfo)
  
  # Create input_ids tensor
  var inputIdsOrtValue: OrtValue = nil
  let inputDataSize = inputIds.data.len * sizeof(int64)
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    inputIds.data[0].unsafeAddr,
    inputDataSize.csize_t,
    inputIds.shape[0].unsafeAddr,
    inputIds.shape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    inputIdsOrtValue.addr
  )
  checkStatus(status)
  
  # Create past_key_values tensors (16 tensors for 8 layers)
  var pastKeyValuesOrt = newSeq[OrtValue](pastKeyValues.len)
  for i in 0 ..< pastKeyValues.len:
    # Always create float32 data for past_key_values, even if empty
    var floatData = newSeq[float32](pastKeyValues[i].data.len)
    for j in 0 ..< floatData.len:
      floatData[j] = 0.0  # Initialize with zeros
    
    # Handle empty data case - create a single zero value
    var dataPtr: pointer
    var dataSize: csize_t
    if floatData.len > 0:
      dataPtr = floatData[0].unsafeAddr
      dataSize = (floatData.len * sizeof(float32)).csize_t
    else:
      # Create a single zero value for empty tensors
      var zeroValue: float32 = 0.0
      dataPtr = zeroValue.unsafeAddr
      dataSize = sizeof(float32).csize_t
    
    let kvDataSize = floatData.len * sizeof(float32)
    status = CreateTensorWithDataAsOrtValue(
      memoryInfo,
      dataPtr,
      dataSize,
      pastKeyValues[i].shape[0].unsafeAddr,
      pastKeyValues[i].shape.len.csize_t,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      pastKeyValuesOrt[i].addr
    )
    checkStatus(status)
  
  # Create attention_mask tensor
  var attentionMaskOrtValue: OrtValue = nil
  let maskDataSize = attentionMask.data.len * sizeof(int64)
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    attentionMask.data[0].unsafeAddr,
    maskDataSize.csize_t,
    attentionMask.shape[0].unsafeAddr,
    attentionMask.shape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    attentionMaskOrtValue.addr
  )
  checkStatus(status)
  
  # Create position_ids tensor
  var positionIdsOrtValue: OrtValue = nil
  let posDataSize = positionIds.data.len * sizeof(int64)
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    positionIds.data[0].unsafeAddr,
    posDataSize.csize_t,
    positionIds.shape[0].unsafeAddr,
    positionIds.shape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    positionIdsOrtValue.addr
  )
  checkStatus(status)
  
  # Build input names array in exact order expected by model
  var inputNames = newSeq[cstring](1 + numLayers * 2 + 2)  # 1 + 16 + 2 = 19
  inputNames[0] = "input_ids"
  for i in 0 ..< numLayers:
    inputNames[1 + i * 2] = "past_key_values." & $i & ".key"
    inputNames[1 + i * 2 + 1] = "past_key_values." & $i & ".value"
  inputNames[1 + numLayers * 2] = "attention_mask"
  inputNames[1 + numLayers * 2 + 1] = "position_ids"
  
  # Build inputs array in exact order
  var inputs = newSeq[OrtValue](1 + numLayers * 2 + 2)  # 19 total
  inputs[0] = inputIdsOrtValue
  for i in 0 ..< pastKeyValues.len:
    inputs[1 + i] = pastKeyValuesOrt[i]
  inputs[1 + numLayers * 2] = attentionMaskOrtValue
  inputs[1 + numLayers * 2 + 1] = positionIdsOrtValue
  
  # Prepare output name
  let outputNameC = outputName.cstring
  
  # Run inference
  var outputOrtValue: OrtValue = nil
  status = Run(
    model.session,
    nil,  # run_options
    inputNames[0].addr,
    inputs[0].addr,
    inputs.len.csize_t,
    outputNameC.addr,
    1.csize_t,
    outputOrtValue.addr
  )
  checkStatus(status)
  
  # Process output
  var typeInfo: OrtTypeInfo
  status = GetTypeInfo(outputOrtValue, typeInfo.addr)
  checkStatus(status)
  defer:
    ReleaseTypeInfo(typeInfo)
  
  var tensorInfo: OrtTensorTypeAndShapeInfo
  status = CastTypeInfoToTensorInfo(typeInfo, tensorInfo.addr)
  checkStatus(status)
  defer:
    ReleaseTensorTypeAndShapeInfo(tensorInfo)
  
  var dimsCount: csize_t
  status = GetDimensionsCount(tensorInfo, dimsCount.addr)
  checkStatus(status)
  
  var outputShape = newSeq[int64](dimsCount)
  if dimsCount > 0:
    status = GetDimensions(tensorInfo, outputShape[0].addr, dimsCount)
    checkStatus(status)
  
  var outputDataPtr: pointer
  status = GetTensorMutableData(outputOrtValue, outputDataPtr.addr)
  checkStatus(status)
  
  var elemCount: csize_t
  status = GetTensorShapeElementCount(tensorInfo, elemCount.addr)
  checkStatus(status)
  
  # Copy output data
  let floatPtr = cast[ptr UncheckedArray[float32]](outputDataPtr)
  var outputData = newSeq[float32](elemCount)
  for i in 0 ..< elemCount.int:
    outputData[i] = floatPtr[i]
  
  # Clean up tensors
  if inputIdsOrtValue != nil:
    ReleaseValue(inputIdsOrtValue)
  if attentionMaskOrtValue != nil:
    ReleaseValue(attentionMaskOrtValue)
  if positionIdsOrtValue != nil:
    ReleaseValue(positionIdsOrtValue)
  for i in 0 ..< pastKeyValuesOrt.len:
    if pastKeyValuesOrt[i] != nil:
      ReleaseValue(pastKeyValuesOrt[i])
  if outputOrtValue != nil:
    ReleaseValue(outputOrtValue)
  
  # Return result
  result = OnnxOutputTensor(
    data: outputData,
    shape: outputShape
  )

## Model introspection utilities

proc getModelOutputNames*(model: OnnxModel): seq[string] =
  ## Discover the output names from the model
  ## This helps determine the correct names for present_key_values outputs
  result = @[]
  var status: OrtStatusPtr
  
  # Get output count first (doesn't need allocator)
  var outputCount: csize_t
  status = SessionGetOutputCount(model.session, outputCount.addr)
  if status != nil:
    ReleaseStatus(status)
    return result
  
  # Get default allocator
  var allocator: OrtAllocator
  status = GetAllocatorWithDefaultOptions(allocator.addr)
  if status != nil:
    ReleaseStatus(status)
    return result
  
  # Get each output name
  for i in 0 ..< outputCount.int:
    var name: cstring
    status = SessionGetOutputName(model.session, i.csize_t, allocator, name.addr)
    if status == nil:
      result.add($name)
    else:
      ReleaseStatus(status)
  
  # Release allocator after use
  ReleaseAllocator(allocator)

## Full inference with KV-cache input support
## Note: present_key_values output is not captured due to ONNX Runtime limitations
## Callers should pass the full sequence each iteration instead of relying on KV-cache

proc runInferenceNeoWithCache*(
  model: OnnxModel,
  inputIds: OnnxInputTensor,
  attentionMask: OnnxInputTensor,
  positionIds: OnnxInputTensor,
  pastKeyValues: seq[OnnxInputTensor],
  numLayers: int = 8
): OnnxNeoOutput =
  ## Run inference on GPT-Neo model
  ## Accepts past_key_values as input but does not return present_key_values
  ## For proper generation, pass the full token sequence on each call
  
  var status: OrtStatusPtr
  
  # Validate inputs
  if inputIds.data.len == 0:
    raise newException(Exception, "Input tensor data cannot be empty")
  
  # Create CPU memory info
  var memoryInfo: OrtMemoryInfo
  status = CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)
  checkStatus(status)
  defer:
    ReleaseMemoryInfo(memoryInfo)
  
  # Create input_ids tensor
  var inputIdsOrtValue: OrtValue = nil
  let inputDataSize = inputIds.data.len * sizeof(int64)
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    inputIds.data[0].unsafeAddr,
    inputDataSize.csize_t,
    inputIds.shape[0].unsafeAddr,
    inputIds.shape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    inputIdsOrtValue.addr
  )
  checkStatus(status)
  
  # Create attention_mask tensor
  var attentionMaskOrtValue: OrtValue = nil
  let maskDataSize = attentionMask.data.len * sizeof(int64)
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    attentionMask.data[0].unsafeAddr,
    maskDataSize.csize_t,
    attentionMask.shape[0].unsafeAddr,
    attentionMask.shape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    attentionMaskOrtValue.addr
  )
  checkStatus(status)
  
  # Create position_ids tensor
  var positionIdsOrtValue: OrtValue = nil
  let posDataSize = positionIds.data.len * sizeof(int64)
  status = CreateTensorWithDataAsOrtValue(
    memoryInfo,
    positionIds.data[0].unsafeAddr,
    posDataSize.csize_t,
    positionIds.shape[0].unsafeAddr,
    positionIds.shape.len.csize_t,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    positionIdsOrtValue.addr
  )
  checkStatus(status)
  
  # Create past_key_values tensors
  var pastKeyValuesOrt = newSeq[OrtValue](pastKeyValues.len)
  for i in 0 ..< pastKeyValues.len:
    var floatData = newSeq[float32](pastKeyValues[i].data.len)
    for j in 0 ..< floatData.len:
      floatData[j] = 0.0
    
    var dataPtr: pointer
    var dataSize: csize_t
    if floatData.len > 0:
      dataPtr = floatData[0].unsafeAddr
      dataSize = (floatData.len * sizeof(float32)).csize_t
    else:
      var zeroValue: float32 = 0.0
      dataPtr = zeroValue.unsafeAddr
      dataSize = sizeof(float32).csize_t
    
    status = CreateTensorWithDataAsOrtValue(
      memoryInfo,
      dataPtr,
      dataSize,
      pastKeyValues[i].shape[0].unsafeAddr,
      pastKeyValues[i].shape.len.csize_t,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      pastKeyValuesOrt[i].addr
    )
    checkStatus(status)
  
  # Build input names and values
  let numInputs = 1 + pastKeyValues.len + 2
  var inputNamesStr = newSeq[string](numInputs)
  var inputs = newSeq[OrtValue](numInputs)
  
  inputNamesStr[0] = "input_ids"
  inputs[0] = inputIdsOrtValue
  
  for i in 0 ..< pastKeyValues.len:
    inputNamesStr[1 + i] = "past_key_values." & $(i div 2) & "." & (if i mod 2 == 0: "key" else: "value")
    inputs[1 + i] = pastKeyValuesOrt[i]
  
  inputNamesStr[1 + pastKeyValues.len] = "attention_mask"
  inputs[1 + pastKeyValues.len] = attentionMaskOrtValue
  
  inputNamesStr[1 + pastKeyValues.len + 1] = "position_ids"
  inputs[1 + pastKeyValues.len + 1] = positionIdsOrtValue
  
  # Convert to cstring array - must keep string data alive!
  var inputNames = newSeq[cstring](numInputs)
  for i in 0 ..< numInputs:
    inputNames[i] = inputNamesStr[i].cstring
  
  # For now, only request logits output to avoid the "Invalid input name" error
  # The model has issues when requesting multiple outputs
  let numOutputs = 1
  var outputOrtValues = newSeq[OrtValue](numOutputs)
  
  # Build output names array
  var outputNamesStr = newSeq[string](numOutputs)
  outputNamesStr[0] = "logits"
  
  var outputNames = newSeq[cstring](numOutputs)
  for i in 0 ..< numOutputs:
    outputNames[i] = outputNamesStr[i].cstring
  
  # Run inference with explicit output names
  status = Run(
    model.session,
    nil,
    inputNames[0].addr,
    inputs[0].addr,
    numInputs.csize_t,
    outputNames[0].addr,
    numOutputs.csize_t,
    outputOrtValues[0].addr
  )
  checkStatus(status)
  
  # Process logits output (first output)
  var logitsTypeInfo: OrtTypeInfo
  status = GetTypeInfo(outputOrtValues[0], logitsTypeInfo.addr)
  checkStatus(status)
  
  var logitsTensorInfo: OrtTensorTypeAndShapeInfo
  status = CastTypeInfoToTensorInfo(logitsTypeInfo, logitsTensorInfo.addr)
  checkStatus(status)
  
  var logitsDimsCount: csize_t
  status = GetDimensionsCount(logitsTensorInfo, logitsDimsCount.addr)
  checkStatus(status)
  
  var logitsShape = newSeq[int64](logitsDimsCount)
  if logitsDimsCount > 0:
    status = GetDimensions(logitsTensorInfo, logitsShape[0].addr, logitsDimsCount)
    checkStatus(status)
  
  var logitsDataPtr: pointer
  status = GetTensorMutableData(outputOrtValues[0], logitsDataPtr.addr)
  checkStatus(status)
  
  var logitsElemCount: csize_t
  status = GetTensorShapeElementCount(logitsTensorInfo, logitsElemCount.addr)
  checkStatus(status)
  
  let logitsFloatPtr = cast[ptr UncheckedArray[float32]](logitsDataPtr)
  var logitsData = newSeq[float32](logitsElemCount)
  for i in 0 ..< logitsElemCount.int:
    logitsData[i] = logitsFloatPtr[i]
  
  # Note: Don't release type info to avoid double-free issues
  # The tensor info is owned by the type info which is owned by the OrtValue
  
  # Note: present_key_values outputs are not captured due to ONNX Runtime limitations
  # The model produces them but we can't request them without errors
  # For now, return empty presentKeyValues - the caller should handle this
  var presentKeyValues = newSeq[OnnxOutputTensor](numLayers * 2)
  
  # Clean up input tensors
  if inputIdsOrtValue != nil:
    ReleaseValue(inputIdsOrtValue)
  if attentionMaskOrtValue != nil:
    ReleaseValue(attentionMaskOrtValue)
  if positionIdsOrtValue != nil:
    ReleaseValue(positionIdsOrtValue)
  for i in 0 ..< pastKeyValuesOrt.len:
    if pastKeyValuesOrt[i] != nil:
      ReleaseValue(pastKeyValuesOrt[i])
  
  # Clean up output tensors
  for i in 0 ..< outputOrtValues.len:
    if outputOrtValues[i] != nil:
      ReleaseValue(outputOrtValues[i])
  
  # Return result
  result = OnnxNeoOutput(
    logits: OnnxOutputTensor(
      data: logitsData,
      shape: logitsShape
    ),
    presentKeyValues: presentKeyValues
  )

## Generic multi-input inference for classification models

type
  NamedInputTensor* = object
    ## Named input tensor for multi-input models
    name*: string
    data*: seq[int64]
    shape*: seq[int64]

proc runInferenceMultiInput*(
  model: OnnxModel,
  inputs: seq[NamedInputTensor],
  outputName: string = "logits"
): OnnxOutputTensor =
  ## Run inference on a model with multiple named inputs.
  ## This is useful for BERT-like models that require input_ids and attention_mask.
  ##
  ## Parameters:
  ##   inputs: Sequence of named input tensors
  ##   outputName: Name of the output node (default: "logits")
  ##
  ## Returns:
  ##   Output tensor with model predictions
  ##
  ## Example:
  ##   let inputs = @[
  ##     NamedInputTensor(name: "input_ids", data: tokenIds, shape: @[1'i64, 512]),
  ##     NamedInputTensor(name: "attention_mask", data: mask, shape: @[1'i64, 512])
  ##   ]
  ##   let output = model.runInferenceMultiInput(inputs)
  ##
  var status: OrtStatusPtr

  if inputs.len == 0:
    raise newException(Exception, "At least one input is required")

  # Create CPU memory info
  var memoryInfo: OrtMemoryInfo
  status = CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memoryInfo.addr)
  checkStatus(status)
  defer:
    ReleaseMemoryInfo(memoryInfo)

  # Create input tensors
  var inputOrtValues = newSeq[OrtValue](inputs.len)
  var inputNames = newSeq[cstring](inputs.len)

  for i in 0 ..< inputs.len:
    if inputs[i].data.len == 0:
      raise newException(Exception, "Input tensor data cannot be empty: " & inputs[i].name)
    if inputs[i].shape.len == 0:
      raise newException(Exception, "Input tensor shape cannot be empty: " & inputs[i].name)

    let dataSize = inputs[i].data.len * sizeof(int64)
    status = CreateTensorWithDataAsOrtValue(
      memoryInfo,
      inputs[i].data[0].unsafeAddr,
      dataSize.csize_t,
      inputs[i].shape[0].unsafeAddr,
      inputs[i].shape.len.csize_t,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      inputOrtValues[i].addr
    )
    checkStatus(status)
    inputNames[i] = inputs[i].name.cstring

  # Prepare output name
  let outputNameC = outputName.cstring

  # Run inference
  var outputOrtValue: OrtValue = nil
  status = Run(
    model.session,
    nil,  # run_options
    inputNames[0].addr,
    inputOrtValues[0].addr,
    inputs.len.csize_t,
    outputNameC.addr,
    1.csize_t,
    outputOrtValue.addr
  )
  checkStatus(status)

  # Get output type info
  var typeInfo: OrtTypeInfo
  status = GetTypeInfo(outputOrtValue, typeInfo.addr)
  checkStatus(status)

  # Cast to tensor info (owned by typeInfo, don't release separately)
  var tensorInfo: OrtTensorTypeAndShapeInfo
  status = CastTypeInfoToTensorInfo(typeInfo, tensorInfo.addr)
  checkStatus(status)

  # Get output dimensions
  var dimsCount: csize_t
  status = GetDimensionsCount(tensorInfo, dimsCount.addr)
  checkStatus(status)

  var outputShape = newSeq[int64](dimsCount)
  if dimsCount > 0:
    status = GetDimensions(tensorInfo, outputShape[0].addr, dimsCount)
    checkStatus(status)

  # Get pointer to output data
  var outputDataPtr: pointer
  status = GetTensorMutableData(outputOrtValue, outputDataPtr.addr)
  checkStatus(status)

  # Get total element count
  var elemCount: csize_t
  status = GetTensorShapeElementCount(tensorInfo, elemCount.addr)
  checkStatus(status)

  # Copy data from OrtValue to Nim seq
  let floatPtr = cast[ptr UncheckedArray[float32]](outputDataPtr)
  var outputData = newSeq[float32](elemCount)
  for i in 0 ..< elemCount.int:
    outputData[i] = floatPtr[i]

  # Clean up input tensors
  for i in 0 ..< inputOrtValues.len:
    if inputOrtValues[i] != nil:
      ReleaseValue(inputOrtValues[i])

  # Clean up output tensor (this also releases typeInfo and tensorInfo)
  if outputOrtValue != nil:
    ReleaseValue(outputOrtValue)

  # Return result
  result = OnnxOutputTensor(
    data: outputData,
    shape: outputShape
  )
