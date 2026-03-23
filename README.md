# ONNX Runtime Nim Wrapper

A high-level Nim wrapper for ONNX Runtime with automatic error handling.

This wrapper **directly binds** to the ONNX Runtime C library installed on your system (via Homebrew, apt, etc.). It does not require any external Nim packages.

## Prerequisites

Make sure you have `onnxruntime` installed on your system:

```bash
# macOS with Homebrew
brew install onnxruntime

# Ubuntu/Debian
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
sudo cp onnxruntime-linux-x64-1.16.3/lib/libonnxruntime.so* /usr/local/lib/
sudo ldconfig
```

## Compilation Flags

| Flag | Description | Example |
|------|-------------|---------|
| `-d:ortPath=PATH` | Set ONNX Runtime installation path (auto-adds `include/` and `lib/` subpaths) | `-d:ortPath=/opt/onnxruntime` |
| `-d:OrtApiVersion=N` | Set ONNX Runtime API version (default: 24) | `-d:OrtApiVersion=20` |

### Examples

```bash
# Use system default paths
nim c your_app.nim

# Use custom ONNX Runtime installation path
nim c -d:ortPath=/opt/onnxruntime your_app.nim

# Use custom API version (if your ONNX Runtime version differs)
nim c -d:ortPath=/opt/onnxruntime -d:OrtApiVersion=20 your_app.nim
```

## Quick Start

```nim
import onnx_rt

# Load the model
let model = loadModel("path/to/model.onnx")

# Create input tensor
let input = newInputTensor(@[1'i64, 2, 3, 4], shape = @[1'i64, 4])

# Run inference - no need to call checkStatus!
let output = model.run(input)

# Access results
echo output.shape   # Output shape
echo output.data    # Raw output data

# Clean up
model.close()
```

## High-Level API

The high-level API handles all error checking internally. You don't need to call `checkStatus` manually.

### Model Loading

```nim
let model = loadModel("models/model.onnx")
model.close()  # Release resources when done
```

### Creating Tensors

```nim
# Input tensor from int64 data
let input = newInputTensor(@[1'i64, 2, 3], shape = @[1'i64, 3])

# Input tensor from float32 data (converted to int64 internally)
let input = newInputTensor(@[1.0'f32, 2.0, 3.0], shape = @[1'i64, 3])
```

### Running Inference

```nim
# Basic inference with single input/output
let output = model.run(input, inputName = "input", outputName = "output")

# For models with multiple inputs/outputs, use the low-level API
```

### Accessing Output

```nim
# Shape helpers
let batch = output.batchSize      # First dimension
let seqLen = output.seqLen        # Second dimension (if exists)
let features = output.featureCount # Last dimension

# Raw data access
let data = output.data  # seq[float32]
let shape = output.shape  # seq[int64]
```

### Model Introspection

```nim
let outputNames = model.getOutputNames()
echo "Model outputs: ", outputNames
```

## Low-Level API (Backward Compatible)

The low-level API is still available for advanced use cases:

```nim
import onnx_rt

# Using low-level API (requires manual checkStatus calls)
let model = newOnnxModel("path/to/model.onnx")
let output = runInference(model, input, "input", "output")
model.close()
```

## Application-Level Examples

The `tests/` directory contains application-level utilities for specific model types:

### GPT-Neo / Text Generation Models

```nim
import onnx_rt
import gpt_neo_utils  # Application-level utilities

let model = loadModel("models/tinystories.onnx")

# Use GPT-Neo specific helpers
let inputIds = newInputTensor(@[1'i64, 2, 3], shape = @[1'i64, 3])
let attentionMask = createAttentionMask(seqLen = 3)
let positionIds = createPositionIds(seqLen = 3)
let pastKeyValues = createEmptyPastKeyValues(numLayers = 8, numHeads = 16, headDim = 4)

# Run inference with GPT-Neo specific function
let output = runNeoWithCache(model, inputIds, attentionMask, positionIds, pastKeyValues)

# Access logits
let logits = output.logits.getLastLogits()
```

## Downloading Test Data

Text generation Example: Download the TinyStories-1M-ONNX files from [Hugging Face](https://huggingface.co/onnx-community/TinyStories-1M-ONNX/):

below are the files you need to download:

```bash
tests/testdata/TinyStories
├── config.json
├── merges.txt
├── model.onnx
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```

TTS Example: Download the Piper voices from [Hugging Face](https://huggingface.co/rhasspy/piper-voices/):

below are the files you need to download:

```bash
tests/testdata/piper-voices
├── voices.json
├── zh_CN-chaowen-medium.onnx
└── zh_CN-chaowen-medium.onnx.json
```

ASR Example: Download the Whisper ASR model from [Hugging Face](https://huggingface.co/onnx-community/whisper-large-v3-chinese-ONNX/):

below are the files you need to download:

```bash
tests/testdata/whisper-large-v3-zh
├── test_input.wav  # Your test audio file (16kHz, 16-bit PCM WAV)
└── onnx-community/whisper-large-v3-chinese-ONNX
    ├── generation_config.json
    ├── tokenizer.json
    ├── vocab.json
    └── onnx
        ├── encoder_model.onnx
        └── decoder_model.onnx
```

Classification Example: Download the URL-TITLE-classifier model from [Hugging Face](https://huggingface.co/firefoxrecap/URL-TITLE-classifier):

below are the files you need to download:

```bash
tests/testdata/url-title-classifier
├── config.json
├── model.onnx
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json
```

TTS example: download model via `scritps/download_vits-icefall-zh-aishell3.sh`

below are the files you need to download:

```bash
tests/testdata/vits-icefall-zh-aishell3
├── date.fst
├── lexicon.txt
├── model.onnx
├── new_heteronym.fst
├── number.fst
├── phone.fst
├── rule.far
├── speakers.txt
├── test_output.wav
└── tokens.txt
```

TTS Example: Kokoro-82M-ONNX (English TTS, high quality, 82M parameters)

Download from [Hugging Face](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX):

```bash
tests/testdata/kokoro-82m
├── model.onnx  # or model_quantized.onnx (smaller, faster)
└── voices/
    └── af.bin  # Voice file (American Female)
```

Kokoro-82M is a compact (82M params) yet high-quality English TTS model supporting multiple voices and 24kHz output.

TTS Example: Sherpa-ONNX Kokoro Multi-Lang v1.0 (Chinese + English, 53 speakers)

Download from [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models):

```bash
tests/testdata/kokoro-multi-lang
├── model.onnx              # ONNX model
├── voices.bin              # Voice vectors (53 speakers)
├── tokens.txt              # Token to ID mapping
├── lexicon-zh.txt          # Chinese lexicon
├── lexicon-us-en.txt       # English lexicon
├── espeak-ng-data/         # espeak-ng data
└── dict/                   # Dictionary files
```

Sherpa-ONNX Kokoro Multi-Lang is a multilingual TTS model supporting Chinese-English mixed text input with 53 different speakers. Unlike the HuggingFace version, you can input raw text directly without external phonemizer.
