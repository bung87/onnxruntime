## config.nims
## Project-wide Nim configuration for ONNX Runtime wrapper
## 
## This file is automatically loaded by Nim when building any file in this project.
## It sets up the necessary compiler and linker flags for ONNX Runtime.

import std/[os, strutils]

# Detect platform
const isMacOS = defined(macosx)
const isLinux = defined(linux)
const isWindows = defined(windows)

# Find ONNX Runtime installation
# On macOS with Homebrew, use brew --prefix
# On Linux, check common installation paths
var onnxRuntimePath = ""

if isMacOS:
  # Try to get the path from Homebrew
  let brewPrefix = strip(staticExec("brew --prefix onnxruntime 2>/dev/null || echo ''"))
  if brewPrefix != "":
    onnxRuntimePath = brewPrefix
  else:
    # Fallback to common paths
    for path in ["/opt/homebrew/opt/onnxruntime", "/usr/local/opt/onnxruntime"]:
      if dirExists(path):
        onnxRuntimePath = path
        break
elif isLinux:
  # Common Linux installation paths
  for path in ["/usr/local", "/usr", "/opt/onnxruntime"]:
    if fileExists(path / "lib/libonnxruntime.so") or 
       fileExists(path / "lib/libonnxruntime.so.1") or
       dirExists(path / "include/onnxruntime"):
      onnxRuntimePath = path
      break

# Apply compiler and linker flags if ONNX Runtime was found
if onnxRuntimePath != "":
  # Add include path for headers
  switch("passC", "-I" & onnxRuntimePath / "include")
  
  # Add library path and link flag
  switch("passL", "-L" & onnxRuntimePath / "lib")
  switch("passL", "-lonnxruntime")
  
  # On macOS, also add rpath for easier distribution
  if isMacOS:
    switch("passL", "-Wl,-rpath," & onnxRuntimePath / "lib")
else:
  # Warn if ONNX Runtime not found, but don't error
  # (the user might have it in a custom location)
  echo "Warning: ONNX Runtime not found in standard locations."
  echo "Make sure onnxruntime is installed and available in your library path."
  echo "On macOS: brew install onnxruntime"
  echo "On Ubuntu: Download from https://github.com/microsoft/onnxruntime/releases"

# Additional compiler settings for better performance and safety
# --threads:on is recommended for ONNX Runtime since it uses threading
switch("threads", "on")

# Enable optimizations in release mode
if defined(release):
  switch("opt", "speed")
  switch("define", "danger")  # Disable all runtime checks for max performance
