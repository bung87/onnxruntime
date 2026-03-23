#!/bin/bash
# Download Kokoro-82M ONNX model and voices for testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTDATA_DIR="$SCRIPT_DIR/../tests/testdata/kokoro-82m"
MODELS_DIR="$TESTDATA_DIR/onnx"
VOICES_DIR="$TESTDATA_DIR/voices"

echo "Creating directories..."
mkdir -p "$MODELS_DIR"
mkdir -p "$VOICES_DIR"

echo ""
echo "Downloading Kokoro-82M ONNX models..."
echo "Available models:"
echo "  1. model.onnx (FP32, 326MB) - Best quality"
echo "  2. model_fp16.onnx (163MB) - Good quality, faster"
echo "  3. model_quantized.onnx (86MB) - Smaller, faster"
echo ""

# Default to the balanced option
MODEL_URL="https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_fp16.onnx"
MODEL_NAME="model.onnx"

# Allow user to choose
if [ "$1" == "fp32" ]; then
    MODEL_URL="https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx"
    echo "Selected: FP32 model (326MB)"
elif [ "$1" == "quantized" ]; then
    MODEL_URL="https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_quantized.onnx"
    echo "Selected: Quantized model (86MB)"
else
    echo "Selected: FP16 model (163MB) - default"
    echo "Use './download_kokoro_82m.sh fp32' for best quality or './download_kokoro_82m.sh quantized' for smallest size"
fi

echo ""
echo "Downloading model to $TESTDATA_DIR/$MODEL_NAME..."
if command -v curl &> /dev/null; then
    curl -L "$MODEL_URL" -o "$TESTDATA_DIR/$MODEL_NAME" --progress-bar
elif command -v wget &> /dev/null; then
    wget "$MODEL_URL" -O "$TESTDATA_DIR/$MODEL_NAME" --show-progress
else
    echo "Error: Neither curl nor wget found. Please install one of them."
    exit 1
fi

echo ""
echo "Downloading voices..."

# Download American Female voice (af) - the default voice
echo "Downloading af.bin (American Female)..."
if command -v curl &> /dev/null; then
    curl -L "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af.bin" -o "$VOICES_DIR/af.bin" --progress-bar
elif command -v wget &> /dev/null; then
    wget "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af.bin" -O "$VOICES_DIR/af.bin" --show-progress
fi

echo ""
echo "Download complete!"
echo ""
echo "Files downloaded to:"
echo "  Model: $TESTDATA_DIR/$MODEL_NAME"
echo "  Voice: $VOICES_DIR/af.bin"
echo ""
echo "You can now run the tests with:"
echo "  nim c -r tests/test_kokoro_tts.nim"
