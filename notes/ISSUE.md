# MeloTTS ONNX Model Issue

## Problem
The MeloTTS ONNX model (`model.onnx`) produces poor quality audio that sounds like just "a" regardless of input text.

## Root Cause
The MeloTTS model was **trained with BERT embeddings** but was **exported to ONNX without BERT support** (`disable_bert=True`). This creates a fundamental mismatch:

1. The original PyTorch model uses BERT embeddings for contextual understanding
2. The ONNX export removed the BERT inputs to simplify the model
3. Without BERT, the model cannot properly interpret phonemes and tones

## Evidence

### Python API (Original)
```python
# MeloTTS expects these inputs:
- x: phoneme IDs
- x_lengths: sequence lengths
- tones: tone IDs
- sid: speaker ID
- lang_ids: language IDs
- bert: BERT embeddings (1024-dim)
- ja_bert: Japanese BERT embeddings (768-dim)
- noise_scale, length_scale, noise_scale_w: inference parameters
```

### ONNX Model (Exported)
```
# ONNX model only has 7 inputs:
- x: phoneme IDs
- x_lengths: sequence lengths
- tones: tone IDs
- sid: speaker ID
- noise_scale: noise scale
- length_scale: length/speed scale
- noise_scale_w: noise width scale

# Missing: lang_ids, bert, ja_bert
```

## Test Results

| Configuration | Output Max | Audio Quality |
|---------------|------------|---------------|
| All-zero tones | 0.000015 | Very quiet (broken) |
| 0-indexed tones (0-4) | 0.117 | Sounds like "a" |
| 1-indexed tones (1-5) | 0.088 | Sounds like "a" |
| With/without padding | Similar | Poor quality |

## Conclusion
This ONNX model cannot produce usable audio without the BERT embeddings it was trained with. The export process removed critical inputs that the model needs for proper inference.

## Recommendation
Use alternative TTS models that are properly exported for ONNX:
- Sherpa-ONNX VITS models (designed for ONNX, no BERT dependency)
- Piper TTS (lightweight, ONNX-native)
- Export MeloTTS yourself with proper BERT integration

## References
- MeloTTS Python API: https://github.com/myshell-ai/MeloTTS/blob/main/melo/api.py
- Original model: https://github.com/myshell-ai/MeloTTS
