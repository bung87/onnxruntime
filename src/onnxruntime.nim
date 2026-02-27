## onnxruntime - High-level ONNX Runtime wrapper for Nim
##
## This module provides a high-level, user-friendly interface for loading and running
## ONNX models. All low-level error handling is managed internally - users don't need
## to call `checkStatus` manually.
##
## Example usage:
##
##   import onnxruntime
##
##   # Load the model
##   let model = loadModel("path/to/model.onnx")
##
##   # Create input tensor
##   let input = newInputTensor(@[1'i64, 2, 3, 4], shape = @[1'i64, 4])
##
##   # Run inference
##   let output = model.run(input)
##
##   echo output.shape   # [1, 4, vocab_size]
##   echo output.data    # Raw logits

import onnxruntime/[onnxmodel, ort_bindings]

# Re-export types that users need
export OnnxInputTensor, OnnxOutputTensor
export OnnxRuntimeError, OrtLoggingLevel, ONNXTensorElementDataType, OrtErrorCode

# Re-export low-level API for backward compatibility and advanced use
export OnnxModel, newOnnxModel, close, runInference
export runInferenceNeo, runInferenceNeoComplete, runInferenceNeoWithCache
export getModelOutputNames, checkStatus, getSession

## High-level Model type that wraps the low-level OnnxModel
type
  Model* = ref object
    ## High-level wrapper for an ONNX model session.
    ## Provides a clean API for inference without manual error handling.
    internal*: OnnxModel  ## Internal low-level model (exposed for advanced use)

## Tensor type aliases for cleaner API
type
  InputTensor* = OnnxInputTensor
    ## Input tensor for model inference
  OutputTensor* = OnnxOutputTensor
    ## Output tensor from model inference

#------------------------------------------------------------------------------
# Model Loading
#------------------------------------------------------------------------------

proc loadModel*(path: string): Model =
  ## Load an ONNX model from a file path.
  ##
  ## Parameters:
  ##   path: Path to the .onnx model file
  ##
  ## Returns:
  ##   A Model object ready for inference
  ##
  ## Raises:
  ##   OnnxRuntimeError: If the model fails to load
  ##
  ## Example:
  ##   let model = loadModel("models/gpt2.onnx")
  result = Model(internal: newOnnxModel(path))

proc close*(model: Model) =
  ## Release the model and free associated resources.
  ##
  ## Example:
  ##   model.close()
  if model != nil and model.internal != nil:
    model.internal.close()

#------------------------------------------------------------------------------
# Tensor Creation Helpers
#------------------------------------------------------------------------------

proc newInputTensor*(data: seq[int64], shape: seq[int64]): InputTensor =
  ## Create a new input tensor with the given data and shape.
  ##
  ## Parameters:
  ##   data: Flattened array of int64 values (e.g., token IDs)
  ##   shape: Tensor shape (e.g., @[batch_size, sequence_length])
  ##
  ## Returns:
  ##   An InputTensor ready for model inference
  ##
  ## Example:
  ##   let input = newInputTensor(@[1'i64, 2, 3], shape = @[1'i64, 3])
  result = InputTensor(data: data, shape: shape)

proc newInputTensor*(data: seq[float32], shape: seq[int64]): InputTensor =
  ## Create a new input tensor with float32 data.
  ## Note: Data is converted to int64 format internally for compatibility.
  ##
  ## Parameters:
  ##   data: Flattened array of float32 values
  ##   shape: Tensor shape
  ##
  ## Returns:
  ##   An InputTensor with data converted to int64
  var intData = newSeq[int64](data.len)
  for i in 0 ..< data.len:
    intData[i] = data[i].int64
  result = OnnxInputTensor(data: intData, shape: shape)

proc newOutputTensor*(data: seq[float32], shape: seq[int64]): OutputTensor =
  ## Create a new output tensor (mainly for testing/debugging).
  ##
  ## Parameters:
  ##   data: Flattened array of float32 values
  ##   shape: Tensor shape
  ##
  ## Returns:
  ##   An OutputTensor
  result = OnnxOutputTensor(data: data, shape: shape)

#------------------------------------------------------------------------------
# Basic Inference
#------------------------------------------------------------------------------

proc run*(model: Model, input: InputTensor, inputName = "input", outputName = "output"): OutputTensor =
  ## Run inference on the model with a single input tensor.
  ##
  ## Parameters:
  ##   input: The input tensor data and shape
  ##   inputName: Name of the input node in the ONNX graph (default: "input")
  ##   outputName: Name of the output node in the ONNX graph (default: "output")
  ##
  ## Returns:
  ##   The output tensor with model predictions
  ##
  ## Raises:
  ##   OnnxRuntimeError: If inference fails
  ##
  ## Example:
  ##   let input = newInputTensor(@[1'i64, 2, 3], shape = @[1'i64, 3])
  ##   let output = model.run(input)
  result = runInference(model.internal, input, inputName, outputName)

#------------------------------------------------------------------------------
# Model Introspection
#------------------------------------------------------------------------------

proc getOutputNames*(model: Model): seq[string] =
  ## Get the names of all output nodes in the model.
  ##
  ## Returns:
  ##   Sequence of output node names
  ##
  ## Example:
  ##   let outputs = model.getOutputNames()
  ##   echo "Model outputs: ", outputs
  result = getModelOutputNames(model.internal)

#------------------------------------------------------------------------------
# Shape and Data Access Helpers
#------------------------------------------------------------------------------

proc batchSize*(tensor: OutputTensor): int64 =
  ## Get the batch size from the output tensor shape.
  ## Returns 0 if shape is empty.
  if tensor.shape.len > 0:
    result = tensor.shape[0]
  else:
    result = 0

proc seqLen*(tensor: OutputTensor): int64 =
  ## Get the sequence length from the output tensor shape.
  ## Returns 0 if shape has fewer than 2 dimensions.
  if tensor.shape.len > 1:
    result = tensor.shape[1]
  else:
    result = 0

proc featureCount*(tensor: OutputTensor): int64 =
  ## Get the feature count from the output tensor shape (typically the last dimension).
  ## Returns 0 if shape is empty.
  if tensor.shape.len > 0:
    result = tensor.shape[tensor.shape.len - 1]
  else:
    result = 0
