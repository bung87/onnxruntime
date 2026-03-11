#!/bin/bash
# Download Sherpa-ONNX VITS MeloTTS Chinese model
# This model supports Chinese and English text-to-speech
#
# Usage:
#   ./download_model.sh         # Auto-select mirror (GitHub first, fallback to China mirrors)
#   ./download_model.sh github  # Force use GitHub
#   ./download_model.sh china   # Force use China mirror (ModelScope/hf-mirror)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Mirror selection
MIRROR="${1:-auto}"

# GitHub original URL
GITHUB_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2"

# China mirrors
# Option 1: HuggingFace Mirror (hf-mirror.com)
HF_MIRROR_URL="https://hf-mirror.com/csukuangfj/sherpa-onnx-vits-zh-ll/resolve/main/vits-melo-tts-zh_en.tar.bz2"

# Option 2: ModelScope (for some models)
# Note: vits-melo-tts-zh_en may not be on ModelScope yet, fallback to other methods

# Option 3: Gitee releases mirror (if available)
GITEE_URL="https://gitee.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2"

# Function to test if URL is accessible
test_url() {
    local url="$1"
    if command -v curl &> /dev/null; then
        curl -sI "$url" | head -1 | grep -q "200\|301\|302"
    elif command -v wget &> /dev/null; then
        wget --spider -q "$url" 2>/dev/null
    else
        return 1
    fi
}

# Function to download file
download_file() {
    local url="$1"
    local output="$2"
    
    echo "Downloading from: $url"
    
    if command -v wget &> /dev/null; then
        wget --progress=bar:force -O "$output" "$url" 2>&1 || return 1
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$output" "$url" 2>&1 || return 1
    else
        echo "Error: Neither wget nor curl is installed"
        exit 1
    fi
}

# Select download URL
case "$MIRROR" in
    github)
        echo "Using GitHub source..."
        MODEL_URL="$GITHUB_URL"
        ;;
    china|mirror|hf|huggingface)
        echo "Using HuggingFace China mirror..."
        MODEL_URL="$HF_MIRROR_URL"
        ;;
    gitee)
        echo "Using Gitee mirror..."
        MODEL_URL="$GITEE_URL"
        ;;
    auto|*)
        # Auto-select: try GitHub first, fallback to mirrors
        echo "Auto-selecting fastest mirror..."
        if test_url "$GITHUB_URL"; then
            echo "GitHub is accessible, using GitHub..."
            MODEL_URL="$GITHUB_URL"
        elif test_url "$HF_MIRROR_URL"; then
            echo "Using HuggingFace China mirror..."
            MODEL_URL="$HF_MIRROR_URL"
        else
            echo "Warning: Cannot test connectivity, trying GitHub..."
            MODEL_URL="$GITHUB_URL"
        fi
        ;;
esac

echo ""
echo "================================================"
echo "  Downloading VITS MeloTTS Chinese model"
echo "================================================"
echo "Source: $MODEL_URL"
echo ""

# Download with retry
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if download_file "$MODEL_URL" "vits-melo-tts-zh_en.tar.bz2"; then
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "Download failed, retrying ($RETRY_COUNT/$MAX_RETRIES)..."
        
        # Try fallback mirrors
        if [ "$MODEL_URL" = "$GITHUB_URL" ]; then
            echo "Trying HuggingFace mirror..."
            MODEL_URL="$HF_MIRROR_URL"
        elif [ "$MODEL_URL" = "$HF_MIRROR_URL" ]; then
            echo "Trying Gitee mirror..."
            MODEL_URL="$GITEE_URL"
        fi
        
        sleep 2
    else
        echo "Error: Download failed after $MAX_RETRIES attempts"
        echo ""
        echo "Suggestions:"
        echo "  1. Check your internet connection"
        echo "  2. Try manual download from:"
        echo "     - GitHub: $GITHUB_URL"
        echo "     - HF Mirror: $HF_MIRROR_URL"
        echo "  3. Use proxy or VPN if accessing from China"
        exit 1
    fi
done

echo ""
echo "Extracting model..."
if tar xvf vits-melo-tts-zh_en.tar.bz2; then
    echo "Extraction successful"
else
    echo "Error: Failed to extract archive"
    exit 1
fi

# Move files to current directory if they were extracted to a subdirectory
if [ -d "vits-melo-tts-zh_en" ]; then
    mv vits-melo-tts-zh_en/* .
    rmdir vits-melo-tts-zh_en 2>/dev/null || true
fi

# Cleanup
rm -f vits-melo-tts-zh_en.tar.bz2

echo ""
echo "================================================"
echo "  Model downloaded successfully!"
echo "================================================"
echo "Files:"
ls -lh

echo ""
echo "You can now run the test with:"
echo "  nim c -r tests/test_vits_tts.nim"
