# Whisper Merged Decoder Implementation

## Overview

This document describes the implementation of the Whisper ASR pipeline using the merged decoder (`decoder_model_merged_fp16.onnx`).

## Important Finding: KV-cache Issue

**The FP16 merged decoder (`decoder_model_merged_fp16.onnx`) produces incorrect results when using KV-cache mode (`use_cache=true`).**

### Verified Working Mode

The only reliable approach is to **always use `use_cache=false`** and pass the **full token sequence** on each step:

```nim
# Correct approach
inputIds = [start, lang, task, notimestamps]
for each step:
    nextToken = decoder(inputIds, use_cache=false)
    inputIds.add(nextToken)
```

### Incorrect Mode (KV-cache)

Using `use_cache=true` with cached KV values produces incorrect token sequences:

```nim
# Incorrect - produces wrong results
step0: nextToken = decoder([start, lang, task, notimestamps], use_cache=false)
step1: nextToken = decoder([nextToken], use_cache=true, past_key_values=cache)
# Results in wrong tokens after step 2
```

## Test Results

### Correct Output (use_cache=false)
```
Token sequence: 1654, 12648, 7182, 3581, 25941, 7626, 1546, 11160, 34719, 6062...
Transcription: "我現在都在車子的位置吧看看能不能正常運行"
```

### Incorrect Output (use_cache=true)
```
Token sequence: 1654, 12648, 1546, 6287, 50257...
Transcription: "我現在的這個" (incorrect and truncated)
```

## Model Architecture

### Encoder Model (`encoder_model.onnx`)
- **Input**: Mel spectrogram `[1, 80, 3000]` (float32)
- **Output**: Encoder hidden states `[1, 1500, 384]` (float32)

### Merged Decoder (`decoder_model_merged_fp16.onnx`)

#### Inputs (20 tensors)
1. `input_ids` `[1, seq_len]` - int64
2. `encoder_hidden_states` `[1, 1500, 384]` - float32
3-18. `past_key_values.*` (8 decoder + 8 encoder KV tensors) - float32
19. `cache_position` `[seq_len]` - int64
20. `use_cache_branch` `[1]` - bool

#### Outputs (17 tensors)
1. `logits` `[1, seq_len, 51865]` - float32
2-17. `present.*` (8 decoder + 8 encoder KV tensors)

## Implementation

See `tests/test_whisper_asr_merged_fp16.nim` for the working implementation.

Key points:
- Always set `use_cache_branch=false`
- Always pass full `input_ids` sequence
- Create zero-filled tensors for all `past_key_values.*` inputs
- The model recomputes attention over the full sequence each step

## Performance Note

Without KV-cache, the model recomputes attention over the full sequence each step, which is O(n²) in sequence length. For long transcriptions, this is slower than KV-cache mode would be, but it produces correct results.

For the test audio (17 tokens generated), the runtime is acceptable. For longer audio, consider:
1. Using the non-merged decoder if available
2. Investigating the FP32 version of the merged decoder
3. Checking for model-specific KV-cache requirements

## Token Sequence

Standard Chinese transcription:
```
<|startoftranscript|> (50258)
<|zh|> (50260)
<|transcribe|> (50359)
<|notimestamps|> (50363)
<text tokens...>
<|endoftext|> (50257)
```

## Testing

```bash
cd tests
nim c -d:release --mm:orc -o:t_whisper_merged test_whisper_asr_merged_fp16.nim
./t_whisper_merged
```

Expected output:
```
Transcription:
============================================================
我現在都在車子的位置吧看看能不能正常運行
============================================================
Generated 17 tokens
[OK] Full pipeline (use_cache=false mode)
```
