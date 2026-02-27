## test_text_generation.nim
## Production-ready text generation test with quality improvements

import unittest
import json
import strutils
import tables
import math
import random
import os
import strformat
import algorithm
import sets
import hashes

import onnxruntime, gpt_neo_utils, bpe_tokenizer

#==============================================================================
# Configuration Types
#==============================================================================

type
  GenerationConfig* = object
    ## Configuration for text generation with GPT-Neo/TinyStories models
    ##
    ## TUNING GUIDE:
    ## =============
    ## 
    ## For COHERENT, FOCUSED output (factual, consistent):
    ##   temperature: 0.5-0.7, topK: 20-40, topP: 0.85-0.90
    ##
    ## For BALANCED output (recommended default):
    ##   temperature: 0.75-0.85, topK: 40-60, topP: 0.90-0.95
    ##
    ## For CREATIVE, VARIED output (storytelling, brainstorming):
    ##   temperature: 0.9-1.1, topK: 60-100, topP: 0.95-0.99
    ##
    ## To REDUCE REPETITION:
    ##   Increase repetitionPenalty: 1.1-1.3 (higher = less repetition)
    ##
    ## To PREVENT GARBLED TEXT:
    ##   Lower temperature, use stopSequences to catch UTF-8 errors
    ##
    maxNewTokens*: int          ## Maximum number of new tokens to generate
    temperature*: float32       ## Sampling temperature (0.0 = greedy, 1.0 = random)
    topK*: int                  ## Top-k sampling (0 = disabled)
    topP*: float32              ## Nucleus sampling (0.0 = disabled)
    repetitionPenalty*: float32 ## Penalty for repeating tokens (1.0 = no penalty)
    seed*: int                  ## Random seed for reproducibility
    eosTokenId*: int            ## End-of-sequence token ID
    padTokenId*: int            ## Padding token ID
    stopSequences*: seq[string] ## Stop generation when these sequences appear
    maxRepetitionLen*: int      ## Maximum length of repeated text to allow
    minTokenProb*: float32      ## Minimum token probability to consider

proc defaultGenerationConfig*(): GenerationConfig =
  ## Return default generation configuration
  result.maxNewTokens = 100
  result.temperature = 0.8'f32
  result.topK = 50
  result.topP = 0.95'f32
  result.repetitionPenalty = 1.2'f32  ## Increased from 1.1 to reduce repetition
  result.seed = 42
  result.eosTokenId = 50256
  result.padTokenId = 50256
  result.stopSequences = @["<|endoftext|>", "Ã", "Â", "âĢ", "ï¿½"]
  result.maxRepetitionLen = 10
  result.minTokenProb = 0.001'f32

#==============================================================================
# Tokenizer Types and Procedures
#==============================================================================

## Tokenizer implementation is provided by `tests/bpe_tokenizer.nim`

#==============================================================================
# Sampling Utilities
#==============================================================================

proc softmax(logits: seq[float32], temperature: float32 = 1.0): seq[float32] =
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

proc topKFilter(logits: var seq[float32], k: int) =
  if k <= 0 or k >= logits.len:
    return
  
  var sorted = logits.sorted(SortOrder.Descending)
  let threshold = sorted[k - 1]
  
  for i in 0 ..< logits.len:
    if logits[i] < threshold:
      logits[i] = -Inf.float32

proc topPFilter(logits: var seq[float32], p: float32) =
  if p <= 0.0 or p >= 1.0:
    return
  
  var probs = softmax(logits)
  
  var indexed: seq[tuple[prob: float32, idx: int]] = @[]
  for i, prob in probs:
    indexed.add((prob, i))
  indexed.sort(proc(a, b: auto): int = cmp(b.prob, a.prob))
  
  var cumsum = 0.0'f32
  var cutoffIdx = 0
  for i, (prob, _) in indexed:
    cumsum += prob
    if cumsum > p:
      cutoffIdx = i
      break
  
  var keep = initHashSet[int]()
  for i in 0 .. cutoffIdx:
    keep.incl(indexed[i].idx)
  
  for i in 0 ..< logits.len:
    if i notin keep:
      logits[i] = -Inf.float32

proc applyRepetitionPenalty(logits: var seq[float32], tokenIds: seq[int64], penalty: float32) =
  if penalty <= 1.0:
    return
  
  for id in tokenIds:
    let idx = id.int
    if idx >= 0 and idx < logits.len:
      if logits[idx] > 0:
        logits[idx] = logits[idx] / penalty
      else:
        logits[idx] = logits[idx] * penalty

proc sampleToken*(logits: seq[float32], config: GenerationConfig, 
                  generatedTokens: seq[int64] = @[]): int64 =
  var filteredLogits = logits
  
  if config.repetitionPenalty > 1.0:
    applyRepetitionPenalty(filteredLogits, generatedTokens, config.repetitionPenalty)
  
  if config.topK > 0:
    topKFilter(filteredLogits, config.topK)
  
  if config.topP > 0.0 and config.topP < 1.0:
    topPFilter(filteredLogits, config.topP)
  
  var probs = softmax(filteredLogits, config.temperature)
  
  # Filter out very low probability tokens
  for i in 0 ..< probs.len:
    if probs[i] < config.minTokenProb:
      probs[i] = 0.0
  
  # Renormalize
  var sumProb = 0.0'f32
  for p in probs:
    sumProb += p
  
  if sumProb > 0:
    for i in 0 ..< probs.len:
      probs[i] = probs[i] / sumProb
  
  let r = rand(1.0'f32)
  var cumsum = 0.0'f32
  for i in 0 ..< probs.len:
    cumsum += probs[i]
    if r <= cumsum:
      return i.int64
  
  return (probs.len - 1).int64

#==============================================================================
# Repetition Detection
#==============================================================================

proc hasRepetition*(text: string, maxRepetitionLen: int): bool =
  ## Check if text has repetitive patterns
  if text.len < maxRepetitionLen * 2:
    return false
  
  # Check for repeated phrases of various lengths
  for phraseLen in 3 .. maxRepetitionLen:
    if text.len < phraseLen * 2:
      continue
    
    let recentText = text[max(0, text.len - phraseLen * 4) .. ^1]
    
    for i in 0 ..< recentText.len - phraseLen:
      let phrase = recentText[i ..< i + phraseLen]
      var count = 0
      var pos = 0
      while pos <= recentText.len - phraseLen:
        if recentText[pos ..< pos + phraseLen] == phrase:
          count += 1
          if count >= 3:  # Same phrase appears 3+ times
            return true
          pos += phraseLen
        else:
          pos += 1
  
  return false

proc shouldStopGeneration*(text: string, config: GenerationConfig): bool =
  ## Check if generation should stop based on stop sequences or quality issues
  
  # Check stop sequences
  for stopSeq in config.stopSequences:
    if stopSeq in text:
      return true
  
  # Check for excessive repetition
  if hasRepetition(text, config.maxRepetitionLen):
    return true
  
  # Check for garbled UTF-8 indicators
  let garbledIndicators = ["Ã", "Â", "ï¿½", "âĢ", "âĿ", "Ŀ", "â"]
  for indicator in garbledIndicators:
    if indicator in text:
      return true
  
  return false

#==============================================================================
# Text Generation
#==============================================================================

proc generateText*(
  model: Model,
  tokenizer: BPETokenizer,
  prompt: string,
  config: GenerationConfig,
  numLayers: int = 8,
  numHeads: int = 16,
  headDim: int = 4
): tuple[text: string, tokens: seq[int64], stoppedEarly: bool] =
  ## Generate text from a prompt using the model
  ## Returns: (generated text, all tokens, whether stopped early)
  
  let inputTokens = tokenizer.encode(prompt)
  if inputTokens.len == 0:
    return ("", @[], false)
  
  var generatedTokens = inputTokens
  let batchSize = 1'i64
  var stoppedEarly = false
  var lastText = ""
  
  for step in 0 ..< config.maxNewTokens:
    let currentSeqLen = generatedTokens.len
    
    # Create input tensor
    let inputTensor = newInputTensor(generatedTokens, shape = @[batchSize, currentSeqLen.int64])
    
    # Create attention mask
    let attentionMask = createAttentionMask(currentSeqLen, batchSize = 1)
    
    # Create position IDs
    let positionIds = createPositionIds(currentSeqLen, batchSize = 1)
    
    # Create empty past_key_values
    let pastKeyValues = createEmptyPastKeyValues(numLayers, numHeads, headDim, batchSize = 1, seqLen = 0)
    
    # Run inference
    let output = runNeoWithCache(
      model, inputTensor, attentionMask, positionIds, pastKeyValues, numLayers
    )
    
    # Get logits for last position
    let vocabSize = output.logits.vocabSize.int
    let lastPosStart = (currentSeqLen - 1) * vocabSize
    var lastLogits = newSeq[float32](vocabSize)
    for i in 0 ..< vocabSize:
      lastLogits[i] = output.logits.data[lastPosStart + i]
    
    # Sample next token
    let nextToken = sampleToken(lastLogits, config, generatedTokens)
    
    # Check for EOS
    if nextToken.int == config.eosTokenId:
      break
    
    generatedTokens.add(nextToken)
    
    # Check for quality issues every 5 tokens
    if step mod 5 == 0:
      let currentText = tokenizer.decode(generatedTokens)
      let newText = currentText[lastText.len .. ^1]
      
      if shouldStopGeneration(newText, config):
        echo &"  [Stopping early at step {step} due to quality issue]"
        stoppedEarly = true
        # Remove the last few tokens that caused the issue
        generatedTokens = generatedTokens[0 ..< max(generatedTokens.len - 3, inputTokens.len)]
        break
      
      lastText = currentText
  
  let fullText = tokenizer.decode(generatedTokens)
  return (fullText, generatedTokens, stoppedEarly)

#==============================================================================
# Tests
#==============================================================================

suite "Text Generation Tests":
  
  const
    TestDataDir = "tests/testdata/TinyStories"
    ModelPath = TestDataDir / "model.onnx"
    TokenizerPath = TestDataDir / "tokenizer.json"
    MergesPath = TestDataDir / "merges.txt"
  
  test "Tokenizer encode/decode roundtrip":
    var tokenizer = initBPETokenizer()
    
    if not fileExists(TokenizerPath):
      skip()
    
    tokenizer.loadTokenizerJson(TokenizerPath)
    if fileExists(MergesPath):
      tokenizer.loadBpeMerges(MergesPath)
    
    let testTexts = @[
      "Hello world",
      "Once upon a time",
      "The quick brown fox"
    ]
    
    for text in testTexts:
      let encoded = tokenizer.encode(text)
      check encoded.len > 0
      let decoded = tokenizer.decode(encoded)
      check decoded.len > 0
  
  test "Generate text with quality controls":
    if not fileExists(ModelPath):
      echo "Model not found at " & ModelPath & ", skipping test"
      skip()
    
    var tokenizer = initBPETokenizer()
    if fileExists(TokenizerPath):
      tokenizer.loadTokenizerJson(TokenizerPath)
      if fileExists(MergesPath):
        tokenizer.loadBpeMerges(MergesPath)
    
    if not tokenizer.loaded:
      echo "Tokenizer not found, skipping test"
      skip()
    
    echo "\n=== Text Generation Test ==="
    echo "Loading model from " & ModelPath & "..."
    let model = loadModel(ModelPath)
    echo "Model loaded successfully!"
    
    # Configure generation with quality improvements
    #
    # TUNING GUIDE:
    # =============
    # 
    # temperature (0.0 - 2.0+):
    #   - 0.5-0.7: More focused, deterministic output
    #   - 0.8-1.0: Balanced creativity (default: 0.8)
    #   - 1.0+: More random, creative but may be incoherent
    #
    # topK (0 - vocab_size):
    #   - 20-40: Conservative, only most likely tokens
    #   - 50-100: Balanced variety (recommended: 50)
    #   - 0: Disabled, consider all tokens
    #
    # topP / nucleus (0.0 - 1.0):
    #   - 0.85-0.90: Tight nucleus, high quality
    #   - 0.92-0.95: Balanced (recommended: 0.92)
    #   - 0.99: Almost all tokens considered
    #
    # repetitionPenalty (1.0 - 2.0):
    #   - 1.0: No penalty (may repeat)
    #   - 1.1-1.2: Gentle reduction (recommended: 1.15)
    #   - 1.3+: Aggressive (may stop generation early)
    #
    # maxNewTokens:
    #   - 10-20: Short completion (~15-30 words)
    #   - 50: Paragraph (~75 words)
    #   - 100-150: Short story (~150-225 words)
    #   - 200+: Long narrative (quality may degrade with small models)
    #
    var config = defaultGenerationConfig()
    config.maxNewTokens = 150         # Generate ~150 tokens for a short story
    config.temperature = 0.75'f32     # Balanced creativity
    config.topK = 50                  # Moderate variety
    config.topP = 0.92'f32            # Balanced nucleus
    config.repetitionPenalty = 1.15'f32 # Gentle repetition reduction
    config.seed = 42
    randomize(config.seed)
    
    let prompt = "Once upon a time, a small dragon named Fluffy wanted to explore the world beyond the mountains."
    echo "\nPrompt: '" & prompt & "'"
    echo "Configuration:"
    echo "  maxNewTokens: " & $config.maxNewTokens
    echo "  temperature: " & $config.temperature
    echo "  topK: " & $config.topK
    echo "  topP: " & $config.topP
    echo "  repetitionPenalty: " & $config.repetitionPenalty
    echo "\nGenerating..."
    echo repeat("-", 60)
    
    let (generatedText, tokens, stoppedEarly) = generateText(
      model, tokenizer, prompt, config,
      numLayers = 8, numHeads = 16, headDim = 4
    )
    
    echo "\nGenerated text:"
    echo generatedText
    echo ""
    echo repeat("-", 60)
    echo "Stats:"
    echo "  Total tokens: " & $tokens.len
    echo "  New tokens: " & $(tokens.len - tokenizer.encode(prompt).len)
    echo "  Stopped early: " & $stoppedEarly
    
    model.close()
    
    let inputTokens = tokenizer.encode(prompt)
    check tokens.len > inputTokens.len
