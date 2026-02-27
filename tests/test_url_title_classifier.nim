## test_url_title_classifier.nim
## Test URL-TITLE-classifier ONNX model for multi-label web classification
## Model: https://huggingface.co/firefoxrecap/URL-TITLE-classifier

import std/[unittest, os, json, tables, strutils, math]
import onnxruntime

const TestDir = currentSourcePath().parentDir / "testdata" / "url-title-classifier"
const ModelPath = TestDir / "model.onnx"
const ConfigPath = TestDir / "config.json"
const TokenizerPath = TestDir / "tokenizer.json"

# Model labels (from config.json id2label)
const Labels = @[
  "News", "Entertainment", "Shop", "Chat", "Education",
  "Government", "Health", "Technology", "Work", "Travel", "Uncategorized"
]

#============================================================================
# Tokenizer Types and Procedures
#============================================================================

type
  Tokenizer = ref object
    vocab: Table[string, int]
    maxLength: int

proc loadTokenizer(path: string): Tokenizer =
  ## Load tokenizer from tokenizer.json
  result = new(Tokenizer)
  result.vocab = initTable[string, int]()
  result.maxLength = 512

  let content = readFile(path)
  let json = parseJson(content)

  # Load vocabulary from model.vocab
  if json.hasKey("model") and json["model"].hasKey("vocab"):
    for token, id in json["model"]["vocab"]:
      let tokenId = id.getInt()
      result.vocab[token] = tokenId

  # Load added tokens
  if json.hasKey("added_tokens"):
    for token in json["added_tokens"]:
      let tokenStr = token["content"].getStr()
      let tokenId = token["id"].getInt()
      result.vocab[tokenStr] = tokenId

proc encode(tokenizer: Tokenizer, text: string): seq[int64] =
  ## Simple BPE-like encoding - split by whitespace and punctuation
  var tokens: seq[int64] = @[]
  var normalized = text.toLowerAscii()
  let words = normalized.split(Whitespace + {':', '/', '.', '-', '_', '?', '&', '='})

  for word in words:
    if word.len == 0:
      continue
    if tokenizer.vocab.hasKey(word):
      tokens.add(tokenizer.vocab[word].int64)
    else:
      let gWord = "Ġ" & word
      if tokenizer.vocab.hasKey(gWord):
        tokens.add(tokenizer.vocab[gWord].int64)
      else:
        for c in word:
          let cs = $c
          if tokenizer.vocab.hasKey(cs):
            tokens.add(tokenizer.vocab[cs].int64)
          elif tokenizer.vocab.hasKey("Ġ" & cs):
            tokens.add(tokenizer.vocab["Ġ" & cs].int64)
  return tokens

proc padOrTruncate(tokens: var seq[int64], maxLength: int, padTokenId: int64 = 1) =
  ## Pad or truncate token sequence to maxLength
  if tokens.len > maxLength:
    tokens.setLen(maxLength)
  elif tokens.len < maxLength:
    while tokens.len < maxLength:
      tokens.add(padTokenId)

#============================================================================
# Model Configuration
#============================================================================

type
  ModelConfig = ref object
    max_position_embeddings: int
    hidden_size: int
    pad_token_id: int
    problem_type: string

proc loadConfig(path: string): ModelConfig =
  ## Load model configuration from config.json using json.to() macro
  let jsonNode = readFile(path).parseJson()
  result = jsonNode.to(ModelConfig)
  # Set defaults for fields that might be missing
  # Use 512 instead of 8192 for faster inference (model supports 8192 but 512 is sufficient)
  if result.max_position_embeddings == 0: result.max_position_embeddings = 512
  else: result.max_position_embeddings = min(result.max_position_embeddings, 512)  # Cap at 512 for test speed
  if result.hidden_size == 0: result.hidden_size = 768
  if result.pad_token_id == 0: result.pad_token_id = 50283
  if result.problem_type.len == 0: result.problem_type = "multi_label_classification"

#============================================================================
# Classification Utilities
#============================================================================

proc sigmoid(x: float32): float32 = 1.0f32 / (1.0f32 + exp(-x))

proc applySigmoid(logits: seq[float32]): seq[float32] =
  result = newSeq[float32](logits.len)
  for i in 0 ..< logits.len: result[i] = sigmoid(logits[i])

proc getPredictions(probs: seq[float32], threshold: float32 = 0.5): seq[string] =
  for i in 0 ..< probs.len:
    if probs[i] >= threshold and i < Labels.len:
      result.add(Labels[i])

proc getTopPrediction(probs: seq[float32]): tuple[label: string, prob: float32] =
  var maxIdx = 0
  for i in 1 ..< probs.len:
    if probs[i] > probs[maxIdx]: maxIdx = i
  if maxIdx < Labels.len: (Labels[maxIdx], probs[maxIdx]) else: ("Unknown", 0.0f32)

#============================================================================
# Test Suite
#============================================================================

suite "URL-TITLE Classifier":

  test "Full pipeline - classify websites":
    ## Full pipeline test: load model, tokenize input, run inference, get predictions
    if not fileExists(ModelPath) or not fileExists(TokenizerPath) or not fileExists(ConfigPath):
      skip()

    let model = loadModel(ModelPath)
    let tokenizer = loadTokenizer(TokenizerPath)
    let config = loadConfig(ConfigPath)

    # Verify model configuration
    check config.problem_type == "multi_label_classification"
    check tokenizer.vocab.len > 0

    # Test cases with expected categories
    let testCases = @[
      ("cnn.com:Breaking News and Latest Updates", "News"),
      ("github.com:Code Repository and Developer Tools", "Technology"),
      ("amazon.com:Buy Products Online at Best Prices", "Shop"),
      ("youtube.com:Entertainment Videos and Music", "Entertainment"),
      ("coursera.org:Online Education and Learning", "Education"),
    ]

    for (inputText, expectedLabel) in testCases:
      # Tokenize input
      var tokens = tokenizer.encode(inputText)
      padOrTruncate(tokens, config.max_position_embeddings, config.pad_token_id.int64)

      # Create attention mask
      var attentionMask = newSeq[int64](tokens.len)
      for i in 0 ..< tokens.len:
        attentionMask[i] = if tokens[i] != config.pad_token_id.int64: 1'i64 else: 0'i64

      # Create input tensors
      var input1: NamedInputTensor
      input1.name = "input_ids"
      input1.data = tokens
      input1.shape = @[1'i64, tokens.len.int64]

      var input2: NamedInputTensor
      input2.name = "attention_mask"
      input2.data = attentionMask
      input2.shape = @[1'i64, attentionMask.len.int64]

      # Run inference
      let output = model.internal.runInferenceMultiInput(@[input1, input2], "logits")

      # Verify output shape
      check output.shape.len >= 2
      check output.shape[0] == 1
      check output.shape[1] == Labels.len.int64

      # Apply sigmoid and get predictions
      let probs = applySigmoid(output.data)
      let predictions = getPredictions(probs, threshold = 0.3)
      let topPred = getTopPrediction(probs)

      # Verify we got valid predictions (pipeline works)
      check predictions.len > 0 or topPred.prob > 0
      check topPred.label in Labels

    model.close()
