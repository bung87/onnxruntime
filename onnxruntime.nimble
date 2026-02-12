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
import os

# Tasks
task download_vits_melo, "Download VITS Melo TTS model (Chinese + English)":
  ## Downloads the vits-melo-tts-zh_en model from sherpa-onnx releases
  let testDataDir = "tests/testdata"
  let modelDir = testDataDir / "vits-melo-tts-zh_en"
  let tempDir = getTempDir()
  let tarFile = tempDir / "vits-melo-tts-zh_en.tar.bz2"
  let modelUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2"

  # Create testdata directory if it doesn't exist
  if not dirExists(testDataDir):
    echo "Creating directory: " & testDataDir
    mkDir(testDataDir)

  # Check if model already exists
  if fileExists(modelDir / "model.onnx"):
    echo "VITS Melo TTS model already exists at: " & modelDir
    return

  # Download the model to temp directory
  if not fileExists(tarFile):
    echo "Downloading VITS Melo TTS model to temporary directory..."
    echo "URL: " & modelUrl
    exec "curl -L -o " & tarFile & " " & modelUrl

  # Extract the archive to testdata directory
  echo "Extracting archive to " & testDataDir & "..."
  exec "tar -xjf " & tarFile & " -C " & testDataDir

  # Remove the temporary archive
  echo "Cleaning up temporary archive..."
  rmFile(tarFile)

  echo "VITS Melo TTS model downloaded successfully to: " & modelDir
  echo ""
  echo "Model files:"
  echo "  - model.onnx: The ONNX model file"
  echo "  - tokens.txt: Token mapping file"
  echo "  - lexicon.txt: Lexicon for text-to-phoneme conversion"
  echo "  - date.fst, number.fst, phone.fst: Rule-based text processing FSTs"
