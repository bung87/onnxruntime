# ONNX TTS Models Comparison

## Why Different ONNX TTS Models Have Different Complexity

### Piper TTS (Simplest)

**File Structure:**
```
zh_CN-chaowen-medium.onnx      # Single model file
zh_CN-chaowen-medium.onnx.json # Config with phoneme_id_map
```

**Why it's easy to use:**
1. **Single ONNX file** - Everything is self-contained
2. **JSON config** - Simple configuration with phoneme mappings
3. **Built-in phoneme handling** - Uses espeak-ng for text-to-phoneme conversion
4. **No external dependencies** - No lexicon files, tone files, or BERT models needed

**Example config (zh_CN-chaowen-medium.onnx.json):**
```json
{
  "audio": {
    "sample_rate": 22050,
    "quality": "medium"
  },
  "espeak": {
    "voice": "zh"
  },
  "phoneme_type": "pinyin",
  "num_symbols": 256,
  "num_speakers": 1,
  "inference": {
    "noise_scale": 0.667,
    "length_scale": 1.0,
    "noise_w": 0.8
  },
  "phoneme_id_map": {
    "_": [0],
    "^": [1],
    "$": [2],
    "b": [4],
    ...
  }
}
```

---

### Sherpa-ONNX VITS (Medium Complexity)

**File Structure:**
```
model.onnx      # Neural network
tokens.txt      # Phoneme vocabulary (Bopomofo symbols)
lexicon.txt     # Word-to-phoneme dictionary
*.fst           # Finite state transducers for text normalization
dict/           # Additional dictionaries
```

**Why it's more complex:**
1. **Separate components** - Model, vocabulary, and lexicon are separate
2. **External text processing** - Need to convert text to Bopomofo phonemes
3. **No built-in G2P** - Must use external tools or dictionaries
4. **Language-specific** - Uses Bopomofo (注音符号) instead of pinyin

**Example tokens.txt (Bopomofo):**
```
_ 0
， 1
。 2
ㄅ 7
ㄆ 8
ㄇ 9
...
```

---

### MeloTTS / Complex VITS (Most Complex)

**File Structure:**
```
model.onnx          # Neural network (exported without BERT!)
tokens.txt          # Phoneme vocabulary
lexicon.txt         # Word-to-phoneme dictionary
date.fst            # Date normalization
number.fst          # Number normalization
phone.fst           # Phone number normalization
new_heteronym.fst   # Heteronym handling
dict/               # Jieba dictionary for Chinese segmentation
```

**Why it's most complex:**
1. **Multiple processing stages:**
   - Text normalization (numbers, dates, abbreviations)
   - Word segmentation (jieba for Chinese)
   - G2P conversion (lexicon lookup)
   - Tone assignment
   - BERT embedding generation (REQUIRED but missing in ONNX!)

2. **External dependencies:**
   - BERT model for contextual embeddings
   - jieba for Chinese text segmentation
   - Multiple FST files for normalization

3. **Critical issue:** The ONNX export removed BERT inputs, but the model was trained with them!

---

## Comparison Table

| Feature | Piper | Sherpa-ONNX | MeloTTS/VITS |
|---------|-------|-------------|--------------|
| **Files** | 2 (.onnx + .json) | 5-7 files | 10+ files |
| **Text processing** | Built-in (espeak) | External | External + BERT |
| **Phonemes** | espeak phonemes | Bopomofo | Pinyin + Tones |
| **Tones** | Handled internally | Not needed | Separate input |
| **BERT required** | No | No | Yes (missing!) |
| **Setup** | Drop-in | Medium | Complex |
| **Quality** | Good | Good | Poor (w/o BERT) |

---

## Recommendations

### Use Piper TTS for:
- Simple deployment
- Multiple language support
- Embedded/edge devices
- Quick prototyping

### Use Sherpa-ONNX for:
- Chinese TTS without BERT dependencies
- When you need more control over text processing
- Research/experimentation with Bopomofo

### Avoid MeloTTS ONNX for:
- Production use (BERT mismatch issue)
- Real-time applications
- Until properly re-exported with BERT support

---

## Key Insight

The complexity difference comes from **where text processing happens:**

- **Piper**: Text → espeak (built-in) → phonemes → ONNX model
- **Sherpa-ONNX**: Text → External processing → Bopomofo → ONNX model  
- **MeloTTS**: Text → jieba → FST normalization → Lexicon → Phonemes + Tones + **BERT** → ONNX model

Piper bundles everything into the ONNX model + espeak, while others require external preprocessing pipelines.
