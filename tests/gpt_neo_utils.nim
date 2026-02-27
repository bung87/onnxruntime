## gpt_neo_utils.nim
## GPT-Neo model specific utilities for text generation
## This is application-level code, not part of the core onnxruntime library

import onnxruntime
import onnxruntime/onnxmodel

# Re-export types needed by users of this module
export InputTensor, OutputTensor, Model, OnnxNeoOutput

#------------------------------------------------------------------------------
# GPT-Neo Model Inference
#------------------------------------------------------------------------------

proc runNeo*(
  model: Model,
  inputIds: InputTensor,
  attentionMask: InputTensor,
  positionIds: InputTensor,
  pastKeyValues: seq[InputTensor] = @[],
  outputName = "logits"
): OutputTensor =
  ## Run inference on a GPT-Neo model with all required inputs.
  ##
  ## Parameters:
  ##   inputIds: Input token IDs tensor
  ##   attentionMask: Attention mask tensor (1s for real tokens, 0s for padding)
  ##   positionIds: Position IDs tensor (0, 1, 2, ... for each token)
  ##   pastKeyValues: Optional past key-value cache tensors for efficient generation
  ##   outputName: Name of the output node (default: "logits")
  ##
  ## Returns:
  ##   Output tensor with logits
  ##
  ## Raises:
  ##   OnnxRuntimeError: If inference fails
  result = runInferenceNeoComplete(
    model.internal, inputIds, attentionMask, positionIds, pastKeyValues, outputName
  )

proc runNeoWithCache*(
  model: Model,
  inputIds: InputTensor,
  attentionMask: InputTensor,
  positionIds: InputTensor,
  pastKeyValues: seq[InputTensor] = @[],
  numLayers = 8
): OnnxNeoOutput =
  ## Run inference on a GPT-Neo model with KV-cache support.
  ##
  ## Note: present_key_values output may not be fully supported by all models.
  ## For proper generation, consider passing the full token sequence each iteration.
  ##
  ## Parameters:
  ##   inputIds: Input token IDs tensor
  ##   attentionMask: Attention mask tensor
  ##   positionIds: Position IDs tensor
  ##   pastKeyValues: Past key-value cache tensors
  ##   numLayers: Number of transformer layers (default: 8)
  ##
  ## Returns:
  ##   OnnxNeoOutput containing logits and present_key_values
  ##
  ## Raises:
  ##   OnnxRuntimeError: If inference fails
  result = runInferenceNeoWithCache(
    model.internal, inputIds, attentionMask, positionIds, pastKeyValues, numLayers
  )

#------------------------------------------------------------------------------
# Utility Functions
#------------------------------------------------------------------------------

proc createAttentionMask*(seqLen: int, batchSize = 1): InputTensor =
  ## Create an attention mask tensor filled with 1s.
  ##
  ## Parameters:
  ##   seqLen: Sequence length
  ##   batchSize: Batch size (default: 1)
  ##
  ## Returns:
  ##   Attention mask tensor of shape [batchSize, seqLen]
  var data = newSeq[int64](seqLen)
  for i in 0 ..< seqLen:
    data[i] = 1'i64
  result = newInputTensor(data, shape = @[batchSize.int64, seqLen.int64])

proc createPositionIds*(seqLen: int, batchSize = 1): InputTensor =
  ## Create position IDs tensor (0, 1, 2, ...).
  ##
  ## Parameters:
  ##   seqLen: Sequence length
  ##   batchSize: Batch size (default: 1)
  ##
  ## Returns:
  ##   Position IDs tensor of shape [batchSize, seqLen]
  var data = newSeq[int64](seqLen)
  for i in 0 ..< seqLen:
    data[i] = i.int64
  result = newInputTensor(data, shape = @[batchSize.int64, seqLen.int64])

proc createEmptyPastKeyValues*(
  numLayers, numHeads, headDim: int,
  batchSize = 1,
  seqLen = 0
): seq[InputTensor] =
  ## Create empty past_key_values tensors for GPT-Neo models.
  ##
  ## Parameters:
  ##   numLayers: Number of transformer layers
  ##   numHeads: Number of attention heads
  ##   headDim: Dimension of each attention head
  ##   batchSize: Batch size (default: 1)
  ##   seqLen: Sequence length (default: 0 for empty cache)
  ##
  ## Returns:
  ##   Sequence of empty key/value tensors for all layers
  result = newSeq[InputTensor](numLayers * 2)
  for i in 0 ..< numLayers * 2:
    result[i] = InputTensor(
      data: newSeq[int64](),
      shape: @[batchSize.int64, numHeads.int64, seqLen.int64, headDim.int64]
    )

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

proc vocabSize*(tensor: OutputTensor): int64 =
  ## Get the vocabulary size from the output tensor shape.
  ## Returns 0 if shape has fewer than 3 dimensions.
  if tensor.shape.len > 2:
    result = tensor.shape[2]
  else:
    result = 0

proc getLogitsForPosition*(tensor: OutputTensor, position: int): seq[float32] =
  ## Extract logits for a specific position in the sequence.
  ##
  ## Parameters:
  ##   position: Position index (0-based)
  ##
  ## Returns:
  ##   Logits vector for the specified position
  let vSize = tensor.vocabSize.int
  let seqLength = tensor.seqLen.int
  if position < 0 or position >= seqLength:
    raise newException(IndexDefect, "Position " & $position & " out of range (0-" & $(seqLength-1) & ")")
  
  result = newSeq[float32](vSize)
  let startIdx = position * vSize
  for i in 0 ..< vSize:
    result[i] = tensor.data[startIdx + i]

proc getLastLogits*(tensor: OutputTensor): seq[float32] =
  ## Extract logits for the last position in the sequence.
  ## This is commonly used for next-token prediction.
  ##
  ## Returns:
  ##   Logits vector for the last position
  let lastPos = tensor.seqLen.int - 1
  if lastPos < 0:
    raise newException(IndexDefect, "Cannot get last logits from empty sequence")
  result = tensor.getLogitsForPosition(lastPos)
