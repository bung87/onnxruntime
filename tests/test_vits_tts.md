# Sherpa-ONNX VITS TTS 测试说明

## ⚠️ 重要提示：MeloTTS 模型问题

**MeloTTS ONNX 模型 (`vits-melo-tts-zh_en`) 存在已知问题**：只能发出 "a" 的声音，无法正常合成语音。

### 问题原因

MeloTTS 模型在训练时使用了 **BERT 嵌入**，但导出 ONNX 时**移除了 BERT 支持** (`disable_bert=True`)，导致：
- PyTorch 原始模型输入：`x, tones, sid, bert, ja_bert, lang_ids...`
- ONNX 导出模型输入：`x, tones, sid` (缺少 BERT!)

没有 BERT 嵌入，模型无法正确理解音素上下文。

### 解决方案

使用 **sherpa-onnx-vits-zh-ll** 模型，这是正确导出的 ONNX 模型：

```bash
cd tests/testdata/sherpa-onnx-vits-zh-ll
./download_model.sh
```

## 添加的文件

| 文件 | 说明 |
|------|------|
| `tests/vits_utils.nim` | VITS 模型推理工具函数 |
| `tests/test_vits_tts.nim` | VITS TTS 测试套件 |
| `tests/testdata/sherpa-onnx-vits-zh-ll/download_model.sh` | 模型下载脚本 |

## 输入格式：汉字

VITS 模型接受 **汉字** 作为输入，通过 lexicon 自动转换为音素和声调：

```nim
let text = "你好世界"
let (tokenIds, tones) = textToTokenIds(text, lexicon, tokens)
```

处理流程：
1. 输入汉字（如 "你好世界"）
2. 通过 lexicon 查找每个字的音素和声调
   - 你 → phonemes: ["n", "i"], tones: [2, 2]
   - 好 → phonemes: ["h", "ao"], tones: [2, 2]
3. 转换为 token IDs 和 tone IDs
4. 传递给模型推理

## 快速开始

### 1. 下载模型

```bash
cd tests/testdata/sherpa-onnx-vits-zh-ll
./download_model.sh
```

### 2. 运行测试

```bash
nim c -r tests/test_vits_tts.nim
```

## API 使用示例

```nim
import onnxruntime, vits_utils

# 加载模型、lexicon 和 tokens
let model = loadModel("model.onnx")
let lexicon = loadLexicon("lexicon.txt")
let tokens = loadTokens("tokens.txt")

# 将汉字转换为 token IDs 和声调
let text = "你好世界"
let (tokenIds, tones) = textToTokenIds(text, lexicon, tokens)

echo "Token IDs: ", tokenIds  // e.g., @[62, 40, 37, 16, ...]
echo "Tones: ", tones         // e.g., @[2, 2, 2, 2, ...]

// 运行推理 (5 speakers, sid 0-4)
let output = runVitsTTS(
  model, 
  tokenIds,
  tones = tones,            // 声调信息
  sid = 0,                  // 说话人 ID (0-4)
  noiseScale = 0.667'f32,   // 随机性控制
  lengthScale = 1.0'f32,    // 语速控制
  noiseScaleW = 0.8'f32     // 音长变化
)

echo "Generated ", output.data.len, " samples"

// 转换为 int16 样本并保存为 WAV
let samples = output.toInt16Samples()
// ... 保存为 WAV 文件

model.close()
```

## 可用模型

| 模型 | 语言 | 说话人 | 状态 |
|------|------|--------|------|
| `sherpa-onnx-vits-zh-ll` | 中文 | 5人 | ✅ 推荐 |
| `vits-melo-tts-zh_en` | 中英 | 1人 | ❌ 有 BERT 问题 |
| `vits-icefall-zh-aishell3` | 中文 | 174人 | ✅ 可用 |
| `vits-zh-hf-fanchen-C` | 中文 | 187人 | ✅ 可用 |

## 参考资料

- [Sherpa-ONNX TTS 文档](https://k2-fsa.github.io/sherpa/onnx/tts/)
- [MeloTTS Issue](notes/ISSUE.md)
