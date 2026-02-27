## bpe_tokenizer.nim
## BPE Tokenizer implementation for GPT-2/GPT-Neo style models

import std/[json, tables, strutils]

type
  BPETokenizer* = ref object
    vocab*: Table[string, int]
    idToToken*: Table[int, string]
    merges*: seq[tuple[first: string, second: string]]
    vocabSize*: int
    eosTokenId*: int
    padTokenId*: int
    loaded*: bool

proc initBPETokenizer*(): BPETokenizer =
  ## Initialize empty BPE tokenizer
  result = new(BPETokenizer)
  result.vocab = initTable[string, int]()
  result.idToToken = initTable[int, string]()
  result.merges = @[]
  result.vocabSize = 50257
  result.eosTokenId = 50256
  result.padTokenId = 50256
  result.loaded = false

proc loadTokenizerJson*(tokenizer: BPETokenizer, path: string) =
  ## Load tokenizer vocabulary from JSON file
  let content = readFile(path)
  let json = parseJson(content)
  
  if json.hasKey("model") and json["model"].hasKey("vocab"):
    for token, id in json["model"]["vocab"]:
      let tokenId = id.getInt()
      tokenizer.vocab[token] = tokenId
      tokenizer.idToToken[tokenId] = token
    tokenizer.vocabSize = tokenizer.vocab.len
  
  # Load merges from tokenizer.json if present
  if json.hasKey("model") and json["model"].hasKey("merges"):
    tokenizer.merges = @[]
    for merge in json["model"]["merges"]:
      if merge.len >= 2:
        tokenizer.merges.add((merge[0].getStr(), merge[1].getStr()))
  
  if json.hasKey("added_tokens"):
    for token in json["added_tokens"]:
      let tokenStr = token["content"].getStr()
      let tokenId = token["id"].getInt()
      tokenizer.vocab[tokenStr] = tokenId
      tokenizer.idToToken[tokenId] = tokenStr
      if tokenStr == "<|endoftext|>":
        tokenizer.eosTokenId = tokenId
  
  tokenizer.loaded = true

proc loadBpeMerges*(tokenizer: BPETokenizer, path: string) =
  ## Load BPE merge rules from file
  tokenizer.merges = @[]
  let content = readFile(path)
  var lineCount = 0
  for line in content.splitLines():
    if lineCount < 1:  # Skip header line
      lineCount.inc
      continue
    let parts = line.split(" ")
    if parts.len >= 2:
      tokenizer.merges.add((parts[0], parts[1]))
    lineCount.inc
    if tokenizer.merges.len >= 50000:
      break

proc getPairs(word: seq[string]): seq[tuple[a: string, b: string]] =
  result = @[]
  if word.len < 2:
    return
  for i in 0 ..< word.len - 1:
    result.add((word[i], word[i+1]))

proc bpeEncode(tokenizer: BPETokenizer, token: string): seq[string] =
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
    
    for merge in tokenizer.merges:
      for pair in pairs:
        if pair.a == merge.first and pair.b == merge.second:
          let score = tokenizer.merges.find(merge)
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

proc encode*(tokenizer: BPETokenizer, text: string): seq[int64] =
  ## Encode text to token IDs using BPE
  result = @[]
  if not tokenizer.loaded:
    return result
  
  var tokens: seq[tuple[word: string, hasSpacePrefix: bool]] = @[]
  var current = ""
  var wordHasSpacePrefix = false
  var isFirstToken = true
  
  for c in text:
    if c.isAlphaAscii or c.isDigit:
      if current.len == 0:
        wordHasSpacePrefix = not isFirstToken
      current.add(c)
    else:
      if current.len > 0:
        tokens.add((current.toLowerAscii(), wordHasSpacePrefix))
        isFirstToken = false
        current = ""
        wordHasSpacePrefix = false
      if c.isSpaceAscii:
        wordHasSpacePrefix = true
      else:
        tokens.add(($c, false))
        isFirstToken = false
        wordHasSpacePrefix = false
  
  if current.len > 0:
    tokens.add((current.toLowerAscii(), wordHasSpacePrefix))
  
  for (token, hasSpacePrefix) in tokens:
    let bpeTokens = tokenizer.bpeEncode(token)
    var isFirstSubToken = true
    for bpeToken in bpeTokens:
      let fullToken = if hasSpacePrefix and isFirstSubToken: "Ġ" & bpeToken else: bpeToken
      isFirstSubToken = false
      
      if hasSpacePrefix and tokenizer.vocab.hasKey(fullToken):
        result.add(tokenizer.vocab[fullToken].int64)
      elif tokenizer.vocab.hasKey(bpeToken):
        result.add(tokenizer.vocab[bpeToken].int64)
      elif tokenizer.vocab.hasKey(fullToken):
        result.add(tokenizer.vocab[fullToken].int64)
      else:
        for c in bpeToken:
          let charToken = $c
          if tokenizer.vocab.hasKey(charToken):
            result.add(tokenizer.vocab[charToken].int64)

proc decode*(tokenizer: BPETokenizer, tokenIds: seq[int64]): string =
  ## Decode token IDs back to text
  if not tokenizer.loaded:
    return ""
  
  var decoded = ""
  for id in tokenIds:
    if tokenizer.idToToken.hasKey(id.int):
      let token = tokenizer.idToToken[id.int]
      if token.len > 0:
        let firstByte = token[0]
        if firstByte == '\xc4':
          if token.len > 1:
            let secondByte = token[1]
            if secondByte == '\xa0':
              decoded.add(" ")
              if token.len > 2:
                decoded.add(token[2..^1])
            elif secondByte == '\x8a':
              decoded.add("\n")
              if token.len > 2:
                decoded.add(token[2..^1])
            elif secondByte == '\x89':
              decoded.add("\t")
              if token.len > 2:
                decoded.add(token[2..^1])
            else:
              decoded.add(token)
          else:
            decoded.add(token)
        else:
          decoded.add(token)
      else:
        decoded.add(token)
    else:
      decoded.add("[" & $id & "]")
  
  return decoded
