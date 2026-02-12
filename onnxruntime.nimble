# Package
version       = "0.1.0"
author        = "bung87"
description   = "ONNX Runtime wrapper for Nim - High-level interface for loading and running ONNX models"
license       = "MIT"
srcDir        = "src"

# Dependencies
requires "nim >= 2.0.0"

# System dependency note:
# This package requires ONNX Runtime C library to be installed on your system.
# On macOS: brew install onnxruntime
# On Ubuntu/Debian: apt-get install libonnxruntime-dev
# On other systems, see: https://onnxruntime.ai/docs/install/
