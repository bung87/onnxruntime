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

ASR Example (not work): Download the Whisper ASR model from [Hugging Face](https://huggingface.co/onnx-community/whisper-large-v3-chinese-ONNX/):

below are the files you need to download:

```bash
tests/testdata/onnx-community/whisper-large-v3-chinese-ONNX
├── config.json
├── generation_config.json
├── special_tokens_map.json
├── tokenizer.json
├── vocab.json
├── encoder_model.onnx
└── decoder_model.onnx
```
