## onnx - ONNX Runtime wrapper for Nim
## 
## This module provides a high-level interface for loading and running
## ONNX models, specifically designed for GPT-like language models.
##
## It directly binds to the ONNX Runtime C library installed on your system
## (e.g., via Homebrew on macOS: `brew install onnxruntime`)
##
## Example usage:
##   
##   import onnx
##   
##   # Load the model
##   let model = newOnnxModel("path/to/model.onnx")
##   
##   # Create dummy input tokens [1, 2, 3, 4]
##   # In real use, convert text to tokens using a tokenizer
##   let input = createDummyInput(seqLen = 4, batchSize = 1)
##   
##   # Run inference
##   let output = runInference(model, input)
##   
##   echo output.shape  # [1, 4, vocab_size]
##   echo output.data   # Raw logits

import onnxruntime/onnxmodel
export onnxmodel
