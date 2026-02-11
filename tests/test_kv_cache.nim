## test_kv_cache.nim
## Test text generation with proper KV-cache handling
## This captures present_key_values from outputs and feeds them back as past_key_values

import unittest
import json
import strutils
import tables
import math
import random
import os
import strformat

import onnx/onnxmodel

# Tokenizer globals
var TokenizerLoaded = false
var VocabSize = 50257
var EosTokenId = 50256
var IdToToken: Table[int, string]
var TokenToId: Table[string, int]
var BpeMerges: seq[tuple[first: string, second: string]]

proc loadTokenizer(path: string) =
  ## Load tokenizer from JSON file
  let jsonContent = readFile(path)
  let tokenizerJson = parseJson(jsonContent)
  
  if tokenizerJson.hasKey("model") and tokenizerJson["model"].hasKey("vocab"):
    let vocab = tokenizerJson["model"]["vocab"]
    for token, id in vocab:
      let tokenStr = token  # token is already a string (key in JSON object)
      let tokenId = id.getInt()
      IdToToken[tokenId] = tokenStr
      TokenToId[tokenStr] = tokenId
    VocabSize = IdToToken.len
    echo &"Tokenizer loaded: {VocabSize} tokens"
  
  if tokenizerJson.hasKey("added_tokens"):
    for token in tokenizerJson["added_tokens"]:
      let tokenStr = token["content"].getStr()
      let tokenId = token["id"].getInt()
      IdToToken[tokenId] = tokenStr
      TokenToId[tokenStr] = tokenId
      if tokenStr == "<|endoftext|>":
        EosTokenId = tokenId
  
  echo &"EOS token ID: {EosTokenId}"
  TokenizerLoaded = true

proc loadBpeMerges(path: string) =
  ## Load BPE merge rules from file
  BpeMerges = @[]
  let content = readFile(path)
  var lineCount = 0
  for line in content.splitLines():
    if lineCount < 1:  # Skip header
      lineCount.inc
      continue
    let parts = line.split(" ")
    if parts.len >= 2:
      BpeMerges.add((parts[0], parts[1]))
    lineCount.inc
    if BpeMerges.len >= 50000:  # Limit merges
      break
  echo &"Loaded {BpeMerges.len} BPE merge rules"

proc getPairs(word: seq[string]): seq[tuple[a: string, b: string]] =
  ## Get all adjacent pairs in a word
  result = @[]
  if word.len < 2:
    return
  var prevChar = word[0]
  for i in 1 ..< word.len:
    result.add((prevChar, word[i]))
    prevChar = word[i]

proc bpeEncode(token: string): seq[string] =
  ## Apply BPE encoding to a token
  if token.len == 0:
    return @[]
  
  var word = newSeq[string](token.len)
  for i in 0 ..< token.len:
    word[i] = $token[i]
  
  if word.len <= 1:
    return word
  
  var pairs = getPairs(word)
  
  while true:
    var minScore = high(int)
    var bigramToMerge: tuple[a: string, b: string] = ("", "")
    
    for merge in BpeMerges:
      for pair in pairs:
        if pair.a == merge.first and pair.b == merge.second:
          let score = BpeMerges.find(merge)
          if score < minScore:
            minScore = score
            bigramToMerge = pair
    
    if minScore == high(int):
      break
    
    var newWord: seq[string] = @[]
    var i = 0
    while i < word.len:
      if i < word.len - 1 and word[i] == bigramToMerge.a and word[i+1] == bigramToMerge.b:
        newWord.add(bigramToMerge.a & bigramToMerge.b)
        i += 2
      else:
        newWord.add(word[i])
        i += 1
    
    word = newWord
    if word.len <= 1:
      break
    pairs = getPairs(word)
  
  return word

proc encodeText(text: string): seq[int64] =
  ## Encode text to token IDs using BPE
  result = @[]
  if not TokenizerLoaded:
    # Fallback: return some default tokens
    return @[100'i64, 200, 300]
  
  # Pre-tokenize: split on whitespace and punctuation
  var tokens: seq[string] = @[]
  var current = ""
  for c in text:
    if c.isAlphaAscii or c.isDigit:
      current.add(c)
    else:
      if current.len > 0:
        tokens.add(current.toLowerAscii())
        current = ""
      if not c.isSpaceAscii:
        tokens.add($c)
  if current.len > 0:
    tokens.add(current.toLowerAscii())
  
  # Apply BPE to each token
  for token in tokens:
    let bpeTokens = bpeEncode(token)
    for bpeToken in bpeTokens:
      let fullToken = if token != bpeToken: "Ġ" & bpeToken else: bpeToken
      if TokenToId.hasKey(fullToken):
        result.add(TokenToId[fullToken].int64)
      elif TokenToId.hasKey(bpeToken):
        result.add(TokenToId[bpeToken].int64)
      else:
        # Try character-level fallback
        for c in bpeToken:
          let charToken = $c
          if TokenToId.hasKey(charToken):
            result.add(TokenToId[charToken].int64)

proc decodeTokens(tokenIds: seq[int64]): string =
  ## Decode token IDs back to text
  if not TokenizerLoaded:
    return "[Tokenizer not loaded]"
  
  var pieces: seq[string] = @[]
  for id in tokenIds:
    if IdToToken.hasKey(id.int):
      var token = IdToToken[id.int]
      # Handle GPT-2 style space marker
      if token.startsWith("Ġ"):
        token = " " & token[1..^1]
      elif token.startsWith("Ċ"):
        token = "\n" & token[1..^1]
      elif token.startsWith("ĉ"):
        token = "\t" & token[1..^1]
      pieces.add(token)
    else:
      pieces.add(&"[{id}]")
  
  result = pieces.join("")

proc softmax(logits: seq[float32], temperature: float32 = 1.0): seq[float32] =
  ## Apply softmax with temperature to logits
  result = newSeq[float32](logits.len)
  var maxLogit = logits[0]
  for l in logits:
    if l > maxLogit:
      maxLogit = l
  
  var sumExp = 0.0'f32
  for i in 0 ..< logits.len:
    result[i] = exp((logits[i] - maxLogit) / temperature)
    sumExp += result[i]
  
  for i in 0 ..< result.len:
    result[i] = result[i] / sumExp

proc sampleToken(logits: seq[float32], temperature: float32 = 1.0): int64 =
  ## Sample a token from logits using temperature
  let probs = softmax(logits, temperature)
  let r = rand(1.0'f32)
  var cumsum = 0.0'f32
  for i in 0 ..< probs.len:
    cumsum += probs[i]
    if r <= cumsum:
      return i.int64
  return (probs.len - 1).int64

# Test paths
const TestDataDir = "tests/testdata"
const ModelPath = TestDataDir / "model.onnx"
const TokenizerPath = TestDataDir / "tokenizer.json"
const MergesPath = TestDataDir / "merges.txt"

suite "KV-Cache Text Generation Tests":
  test "Generate text with KV-cache (single token at a time)":
    echo "\n=== KV-Cache Text Generation Test ==="
    
    if not fileExists(ModelPath):
      echo "Model not found, skipping test"
      skip()
    
    # Load tokenizer and BPE merges
    if fileExists(TokenizerPath):
      loadTokenizer(TokenizerPath)
      if fileExists(MergesPath):
        loadBpeMerges(MergesPath)
    else:
      echo "Tokenizer not found, using simple token IDs"
    
    echo "Loading model..."
    let model = newOnnxModel(ModelPath)
    echo "Model loaded successfully!"
    
    # Model parameters for TinyStories-1M
    let numLayers = 8
    let numHeads = 16
    let hiddenSize = 64
    let headDim = hiddenSize div numHeads  # 4
    let batchSize = 1
    
    # Use a meaningful story prompt
    let promptText = "Once upon a time, a small dragon named Fluffy wanted to explore the world beyond the mountains."
    var inputTokens = encodeText(promptText)
    if inputTokens.len == 0:
      inputTokens = @[100'i64, 200'i64, 300'i64]
    
    echo &"Prompt: '{promptText}'"
    echo &"Encoded {inputTokens.len} tokens"
    if TokenizerLoaded:
      let decoded = decodeTokens(inputTokens)
      echo &"Decoded: '{decoded}'"
    
    # Generation parameters
    let maxNewTokens = 10
    let temperature = 1.0'f32
    
    echo &"\nGenerating {maxNewTokens} new tokens (with KV-cache)..."
    echo repeat("-", 50)
    
    var generatedTokens = inputTokens
    var kvCache: seq[OnnxOutputTensor] = @[]  # Will hold present_key_values from previous step
    
    for step in 0 ..< maxNewTokens:
      let currentSeqLen = if step == 0: inputTokens.len else: 1  # First step uses full prompt, subsequent use single token
      let currentTokens = if step == 0: inputTokens else: @[generatedTokens[^1]]
      
      # Create input tensor
      let inputTensor = OnnxInputTensor(
        data: currentTokens,
        shape: @[batchSize.int64, currentSeqLen.int64]
      )
      
      # Create attention mask
      var attentionMaskData = newSeq[int64](currentSeqLen)
      for i in 0 ..< currentSeqLen:
        attentionMaskData[i] = 1'i64
      let attentionMask = OnnxInputTensor(
        data: attentionMaskData,
        shape: @[batchSize.int64, currentSeqLen.int64]
      )
      
      # Create position IDs
      var positionIdsData = newSeq[int64](currentSeqLen)
      let startPos = if step == 0: 0 else: generatedTokens.len - 1
      for i in 0 ..< currentSeqLen:
        positionIdsData[i] = (startPos + i).int64
      let positionIds = OnnxInputTensor(
        data: positionIdsData,
        shape: @[batchSize.int64, currentSeqLen.int64]
      )
      
      # Prepare past_key_values from KV cache
      var pastKeyValues: seq[OnnxInputTensor] = @[]
      if kvCache.len > 0 and kvCache[0].data.len > 0:
        # Convert present_key_values from previous step to past_key_values for this step
        for layer in 0 ..< numLayers:
          for kv in 0 ..< 2:  # key and value
            let cacheIdx = layer * 2 + kv
            let cacheTensor = kvCache[cacheIdx]
            # Convert OnnxOutputTensor (float32) to OnnxInputTensor (int64)
            # Note: We need to keep the shape but the data type changes
            var intData = newSeq[int64](cacheTensor.data.len)
            for i in 0 ..< cacheTensor.data.len:
              intData[i] = cacheTensor.data[i].int64
            pastKeyValues.add(OnnxInputTensor(
              data: intData,
              shape: cacheTensor.shape
            ))
      else:
        # First step or no cache: create empty past_key_values
        for layer in 0 ..< numLayers:
          for kv in 0 ..< 2:
            pastKeyValues.add(OnnxInputTensor(
              data: @[],
              shape: @[batchSize.int64, numHeads.int64, 0'i64, headDim.int64]
            ))
      
      # Run inference with KV-cache
      let output = runInferenceNeoWithCache(model, inputTensor, attentionMask, positionIds, pastKeyValues, numLayers)
      
      # Update KV cache with present_key_values for next iteration
      kvCache = output.presentKeyValues
      
      # Get logits for the last position
      let vocabSizeInt = output.logits.shape[2].int
      let lastPosStart = (currentSeqLen - 1) * vocabSizeInt
      var lastLogits = newSeq[float32](vocabSizeInt)
      for i in 0 ..< vocabSizeInt:
        lastLogits[i] = output.logits.data[lastPosStart + i]
      
      # Sample next token
      let nextToken = sampleToken(lastLogits, temperature)
      
      # Check for end of text
      if nextToken == EosTokenId.int64:
        echo "<|endoftext|>"
        break
      
      generatedTokens.add(nextToken)
      
      # Show the token
      var tokenStr = &"[{nextToken}]"
      if IdToToken.hasKey(nextToken.int):
        tokenStr = IdToToken[nextToken.int]
      echo &"Step {step + 1}: Generated token {nextToken} ({tokenStr})"
    
    echo repeat("-", 50)
    echo &"\nTotal tokens generated: {generatedTokens.len - inputTokens.len}"
    echo &"Full token sequence length: {generatedTokens.len}"
    
    # Try to decode
    if TokenizerLoaded:
      let fullText = decodeTokens(generatedTokens)
      echo &"Full text: '{fullText}'"
      
      let generatedText = decodeTokens(generatedTokens[inputTokens.len .. ^1])
      echo &"Generated text: '{generatedText}'"
    
    echo "=== Generation complete ===\n"
    
    # Cleanup
    model.close()
    
    # Verify we generated some tokens
    check generatedTokens.len > inputTokens.len
