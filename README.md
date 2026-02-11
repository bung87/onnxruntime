# ONNX Runtime Nim Wrapper

A simple Nim wrapper for ONNX Runtime.

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


## Downloading test data

Example: Download the TinyStories-1M-ONNX files from [Hugging Face](https://huggingface.co/onnx-community/TinyStories-1M-ONNX/b):

below are the files you need to download:

```bash
tests/testdata
├── config.json
├── merges.txt
├── model.onnx
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```
