#!/bin/bash
# Download VITS AISHELL3 Chinese model (174 speakers, Pinyin-based)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2"

echo "Downloading VITS AISHELL3 Chinese model (174 speakers)..."
echo "URL: $MODEL_URL"

if command -v wget &> /dev/null; then
    wget "$MODEL_URL" -O vits-icefall-zh-aishell3.tar.bz2
elif command -v curl &> /dev/null; then
    curl -L "$MODEL_URL" -o vits-icefall-zh-aishell3.tar.bz2
else
    echo "Error: Neither wget nor curl is installed"
    exit 1
fi

echo "Extracting model..."
tar xvf vits-icefall-zh-aishell3.tar.bz2

if [ -d "vits-icefall-zh-aishell3" ]; then
    mv vits-icefall-zh-aishell3/* .
    rmdir vits-icefall-zh-aishell3 2>/dev/null || true
fi

rm -f vits-icefall-zh-aishell3.tar.bz2

echo "Model downloaded successfully!"
ls -lh
