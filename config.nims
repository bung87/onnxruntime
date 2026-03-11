## config.nims
## Project-wide Nim configuration for ONNX Runtime wrapper
##
## This file is automatically loaded by Nim when building any file in this project.
## It sets up the necessary compiler and linker flags for ONNX Runtime.

import std/[os, strutils]

# Additional compiler settings for better performance and safety
# --threads:on is recommended for ONNX Runtime since it uses threading
switch("threads", "on")

# begin Nimble config (version 2)
when withDir(thisDir(), system.fileExists("nimble.paths")):
  include "nimble.paths"
# end Nimble config
