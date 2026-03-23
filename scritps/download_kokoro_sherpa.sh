#!/bin/bash
# Download Sherpa-ONNX Kokoro Multi-Lang model
# This version supports Chinese and English mixed TTS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTDATA_DIR="$SCRIPT_DIR/../tests/testdata/kokoro-multi-lang"

echo "Creating directory..."
mkdir -p "$TESTDATA_DIR"

echo ""
echo "Downloading Sherpa-ONNX Kokoro Multi-Lang v1.0..."
echo "Model size: ~250MB (compressed)"
echo ""

cd "$TESTDATA_DIR"

# Download model
MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2"

if command -v wget &> /dev/null; then
    echo "Downloading with wget..."
    wget --show-progress "$MODEL_URL"
elif command -v curl &> /dev/null; then
    echo "Downloading with curl..."
    curl -L -O "$MODEL_URL" --progress-bar
else
    echo "Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Extract
echo ""
echo "Extracting..."
tar xf kokoro-multi-lang-v1_0.tar.bz2

# Move files to current directory if extracted to subdirectory
if [ -d "kokoro-multi-lang-v1_0" ]; then
    mv kokoro-multi-lang-v1_0/* .
    rmdir kokoro-multi-lang-v1_0
fi

# Remove archive
rm kokoro-multi-lang-v1_0.tar.bz2

echo ""
echo "Download complete!"
echo ""
echo "Files in $TESTDATA_DIR:"
ls -lh

echo ""
echo "You can now run the tests with:"
echo "  nim c -r tests/test_kokoro_sherpa.nim"
