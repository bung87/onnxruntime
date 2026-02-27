## test_url_title_classifier.nim
## Test URL-TITLE-classifier ONNX model for multi-label web classification
## Model: https://huggingface.co/firefoxrecap/URL-TITLE-classifier

import std/[unittest, os, json, math, tables]
import onnxruntime
import ./bpe_tokenizer

const TestDir = currentSourcePath().parentDir / "testdata" / "url-title-classifier"
const ModelPath = TestDir / "model.onnx"
const ConfigPath = TestDir / "config.json"
const TokenizerPath = TestDir / "tokenizer.json"
const MergesPath = TestDir / "merges.txt"

const Labels = @[
  "News", "Entertainment", "Shop", "Chat", "Education",
  "Government", "Health", "Technology", "Work", "Travel", "Uncategorized"
]

const MaxSeqLen = 128
const PadTokenId = 50283
const ClsTokenId = 50281
const EosTokenId = 50282

type
  ModelConfig = ref object
    padTokenId: int
    clsTokenId: int
    eosTokenId: int

proc loadConfig(path: string): ModelConfig =
  result = new(ModelConfig)
  let jsonNode = parseJson(readFile(path))
  result.padTokenId = jsonNode{"pad_token_id"}.getInt(PadTokenId)
  result.clsTokenId = jsonNode{"cls_token_id"}.getInt(ClsTokenId)
  result.eosTokenId = jsonNode{"eos_token_id"}.getInt(EosTokenId)

proc sigmoid(x: float32): float32 {.inline.} =
  1.0f32 / (1.0f32 + exp(-x))

proc getTopPrediction(probs: seq[float32]): string =
  var maxIdx = 0
  var maxProb = probs[0]
  for i in 1 ..< probs.len:
    if probs[i] > maxProb:
      maxProb = probs[i]
      maxIdx = i
  if maxIdx < Labels.len: Labels[maxIdx] else: "Unknown"

proc padOrTruncate(tokens: var seq[int64], maxLength: int, padTokenId: int64) =
  if tokens.len > maxLength:
    tokens.setLen(maxLength)
  else:
    let padCount = maxLength - tokens.len
    for _ in 0 ..< padCount:
      tokens.add(padTokenId)

suite "URL-TITLE Classifier":

  test "Full pipeline - classify websites":
    if not fileExists(ModelPath) or not fileExists(TokenizerPath) or not fileExists(ConfigPath):
      skip()

    let model = loadModel(ModelPath)
    let config = loadConfig(ConfigPath)

    # Initialize and load BPE tokenizer
    var tokenizer = initBPETokenizer()
    tokenizer.loadTokenizerJson(TokenizerPath)
    if fileExists(MergesPath):
      tokenizer.loadBpeMerges(MergesPath)

    check tokenizer.vocab.len > 0
    check tokenizer.loaded

    var inputIds = NamedInputTensor(
      name: "input_ids",
      shape: @[1'i64, MaxSeqLen.int64]
    )

    var attentionMask = NamedInputTensor(
      name: "attention_mask",
      shape: @[1'i64, MaxSeqLen.int64]
    )

    let testCases = @[
      ("cnn.com:Breaking News Today", "News"),
      ("github.com:Software Development Code", "Technology"),
      ("amazon.com:Shopping Buy Products Online", "Shop"),
      ("youtube.com:Watch Videos Entertainment", "Entertainment"),
      ("coursera.org:Online Learning Education Courses", "Education"),
    ]

    let padTokenId = config.padTokenId.int64
    let clsTokenId = config.clsTokenId.int64

    for (inputText, expectedLabel) in testCases:
      var tokens = tokenizer.encode(inputText)
      # Add CLS token at the beginning for BERT-style models
      tokens.insert(clsTokenId, 0)
      padOrTruncate(tokens, MaxSeqLen, padTokenId)

      var mask = newSeq[int64](MaxSeqLen)
      for i in 0 ..< MaxSeqLen:
        mask[i] = if tokens[i] != padTokenId: 1'i64 else: 0'i64

      inputIds.data = tokens
      attentionMask.data = mask

      let output = model.internal.runInferenceMultiInput(@[inputIds, attentionMask], "logits")

      check output.shape[1] == Labels.len.int64

      var probs = newSeq[float32](output.data.len)
      for i in 0 ..< output.data.len:
        probs[i] = sigmoid(output.data[i])

      let topLabel = getTopPrediction(probs)
      check topLabel == expectedLabel

    model.close()
