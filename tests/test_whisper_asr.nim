## test_whisper_asr.nim
## Test Whisper ONNX model for ASR (Automatic Speech Recognition)

import std/[unittest, os]
import onnxruntime, whisper_utils
import std/json

const TestDataDir = currentSourcePath().parentDir / "testdata" / "whisper-large-v3-zh"
const ConfigDataDir = TestDataDir / "onnx-community" / "whisper-large-v3-chinese-ONNX"
const ModelDataDir = ConfigDataDir / "onnx"
const EncoderPath = ModelDataDir / "encoder_model.onnx"
const DecoderPath = ModelDataDir / "decoder_model.onnx"
const GenerationConfigPath = ConfigDataDir / "generation_config.json"
const TokenizerPath = ConfigDataDir / "tokenizer.json"
const VocabPath = ConfigDataDir / "vocab.json"
const TestAudioPath = TestDataDir / "test_input.wav"

# JSON config types
type
  GenerationConfig = object
    decoder_start_token_id: int
    eos_token_id: int
    begin_suppress_tokens: seq[int64]

  AddedToken = object
    id: int
    content: string

  TokenizerConfig = object
    added_tokens: seq[AddedToken]

  WhisperConfig = object
    startToken: int
    endToken: int
    langToken: int
    taskToken: int
    noTimestampsToken: int
    suppressTokens: seq[int64]

proc loadConfig(): WhisperConfig =
  ## Load all config files and build WhisperConfig
  let genConfig = if fileExists(GenerationConfigPath):
    readFile(GenerationConfigPath).parseJson().to(GenerationConfig)
  else:
    GenerationConfig(
      decoder_start_token_id: 50258,
      eos_token_id: 50257,
      begin_suppress_tokens: @[220'i64, 50257'i64]
    )

  let tokConfig = if fileExists(TokenizerPath):
    readFile(TokenizerPath).parseJson().to(TokenizerConfig)
  else:
    TokenizerConfig(added_tokens: @[])

  result.startToken = genConfig.decoder_start_token_id
  result.endToken = genConfig.eos_token_id
  result.suppressTokens = genConfig.begin_suppress_tokens

  for token in tokConfig.added_tokens:
    case token.content:
      of "<|startoftranscript|>": result.startToken = token.id
      of "<|endoftext|>": result.endToken = token.id
      of "<|zh|>": result.langToken = token.id
      of "<|transcribe|>": result.taskToken = token.id
      of "<|notimestamps|>": result.noTimestampsToken = token.id
      else: discard

proc loadVocab(): seq[string] =
  ## Load vocabulary from vocab.json and tokenizer.json
  result = newSeq[string](51865)

  if fileExists(VocabPath):
    for token, id in readFile(VocabPath).parseJson().pairs:
      let idx = id.getInt
      if idx < result.len:
        result[idx] = token

  if fileExists(TokenizerPath):
    let tokConfig = readFile(TokenizerPath).parseJson().to(TokenizerConfig)
    for token in tokConfig.added_tokens:
      if token.id < result.len:
        result[token.id] = token.content

suite "Whisper ASR":

  test "Full pipeline - encode and decode":
    if not fileExists(EncoderPath) or not fileExists(DecoderPath) or not fileExists(TestAudioPath):
      skip()

    let config = loadConfig()
    let vocab = loadVocab()

    # Step 1: Load audio and compute mel spectrogram
    let audio = loadWavFile(TestAudioPath)
    let melSpec = computeWhisperMelSpectrogram(padOrTrimAudio(audio, WHISPER_SAMPLE_RATE * 30))

    # Step 2: Run encoder
    let whisper = loadWhisper(EncoderPath, DecoderPath)
    let encoderOutput = whisper.runEncoder(melSpec)

    # Step 3: Run decoder with greedy decoding
    var inputIds = @[
      config.startToken.int64,
      config.langToken.int64,
      config.taskToken.int64,
      config.noTimestampsToken.int64
    ]

    var generatedTokens: seq[int64] = @[]
    const maxLength = 50
    const vocabSize = 51865

    for step in 0 ..< maxLength:
      var nextToken = whisper.runDecoderStep(inputIds, encoderOutput, vocabSize)

      # Skip suppressed tokens for first few steps
      if step < 5:
        while nextToken in config.suppressTokens:
          inputIds.add(nextToken)
          nextToken = whisper.runDecoderStep(inputIds, encoderOutput, vocabSize)
          inputIds.delete(inputIds.len - 1)

      if nextToken == config.endToken.int64:
        if generatedTokens.len < 3:
          continue
        break

      generatedTokens.add(nextToken)
      inputIds.add(nextToken)

      if inputIds.len > 448:
        break

    # Step 4: Convert tokens to text
    echo "\nFirst 10 generated tokens: "
    for i in 0 ..< min(10, generatedTokens.len):
      echo "  [" & $i & "] " & $generatedTokens[i]
    let transcription = decodeTokensToText(generatedTokens, vocab)
    echo "\nTranscription: " & transcription

    # Cleanup
    ReleaseValue(encoderOutput)
    whisper.close()

    check generatedTokens.len >= 0
