##
## NOTE: MeloTTS model (vits-melo-tts-zh_en) has a known issue where it only
## produces "a" sounds because it was exported without BERT embeddings.
## See: notes/ISSUE.md
##


import unittest
import os
import tables

import onnx_rt, vits_utils

# Test paths
# Using vits-icefall-zh-aishell3 (174 speakers, Pinyin-based)
# MeloTTS has BERT dependency issue - see notes/ISSUE.md
const TestDataDir = "tests/testdata/vits-icefall-zh-aishell3"
const ModelPath = TestDataDir / "model.onnx"
const LexiconPath = TestDataDir / "lexicon.txt"
const TokensPath = TestDataDir / "tokens.txt"

# Test using Chinese characters
const TestChinese = "你好世界"

proc writeWavFile(path: string, pcmData: seq[int16], sampleRate: int) =
  ## Write PCM data to WAV file
  let numChannels = 1.uint16
  let bitsPerSample = 16.uint16
  let byteRate = sampleRate.uint32 * numChannels.uint32 * (bitsPerSample div 8).uint32
  let blockAlign = numChannels * (bitsPerSample div 8)
  let dataSize = (pcmData.len * sizeof(int16)).uint32
  let chunkSize = 36.uint32 + dataSize

  var f = open(path, fmWrite)
  defer: f.close()

  # RIFF header
  f.write("RIFF")
  discard f.writeBuffer(chunkSize.addr, 4)
  f.write("WAVE")

  # fmt subchunk
  f.write("fmt ")
  let subchunkSize = 16.uint32
  discard f.writeBuffer(subchunkSize.addr, 4)
  let audioFormat = 1.uint16
  discard f.writeBuffer(audioFormat.addr, 2)
  discard f.writeBuffer(numChannels.addr, 2)
  let sr = sampleRate.uint32
  discard f.writeBuffer(sr.addr, 4)
  discard f.writeBuffer(byteRate.addr, 4)
  discard f.writeBuffer(blockAlign.addr, 2)
  discard f.writeBuffer(bitsPerSample.addr, 2)

  # data subchunk
  f.write("data")
  discard f.writeBuffer(dataSize.addr, 4)

  # Write PCM data
  for sample in pcmData:
    discard f.writeBuffer(sample.addr, sizeof(int16))

suite "Sherpa-ONNX VITS TTS (AISHELL3 Chinese)":

  test "Load lexicon and tokens":
    if not fileExists(LexiconPath):
      echo "Skipping: Lexicon file not found: ", LexiconPath
      echo "Please download the model from:"
      echo "  cd tests/testdata/sherpa-onnx-vits-zh-ll"
      echo "  ./download_model.sh"
      skip()
    elif not fileExists(TokensPath):
      echo "Skipping: Tokens file not found: ", TokensPath
      skip()
    else:
      let lexicon = loadLexicon(LexiconPath)
      let tokens = loadTokens(TokensPath)

      check lexicon.len() > 0
      check tokens.len() > 0

      echo "Loaded ", lexicon.len(), " lexicon entries"
      echo "Loaded ", tokens.len(), " tokens"

      # Check some specific characters
      if lexicon.hasKey("你"):
        echo "你 -> phonemes: ", lexicon["你"].phonemes
      if lexicon.hasKey("好"):
        echo "好 -> phonemes: ", lexicon["好"].phonemes

  test "Convert Chinese characters to token IDs":
    if not fileExists(LexiconPath) or not fileExists(TokensPath):
      echo "Skipping: Lexicon or tokens not found"
      skip()
    else:
      let lexicon = loadLexicon(LexiconPath)
      let tokens = loadTokens(TokensPath)

      # Test character to phoneme and tone conversion
      let (phonemes, tones) = charactersToPhonemesAndTones(TestChinese, lexicon)
      echo "Input: ", TestChinese
      echo "Phonemes: ", phonemes
      echo "Tones: ", tones
      check phonemes.len > 0
      check tones.len > 0

      # Test phoneme to token ID conversion (without blanks)
      let tokenIds = phonemesToIds(phonemes, tokens, addBlank = false)
      echo "Token IDs: ", tokenIds
      check tokenIds.len > 0

      # Test combined function
      let tokenIds2 = textToTokenIds(TestChinese, lexicon, tokens)
      check tokenIds2.len > 0
      check tokenIds == tokenIds2

  test "Full pipeline - Chinese text to WAV file":
    if not fileExists(ModelPath):
      echo "Skipping: Model not found: ", ModelPath
      echo "Please download from:"
      echo "  cd tests/testdata/sherpa-onnx-vits-zh-ll"
      echo "  ./download_model.sh"
      skip()
    elif not fileExists(LexiconPath) or not fileExists(TokensPath):
      echo "Skipping: Lexicon or tokens not found"
      skip()
    else:
      # Load resources
      let lexicon = loadLexicon(LexiconPath)
      let tokens = loadTokens(TokensPath)

      check lexicon.len() > 0
      check tokens.len() > 0

      # Convert Chinese text to token IDs
      let tokenIds = textToTokenIds(TestChinese, lexicon, tokens)
      check tokenIds.len > 0
      echo "Token IDs for '", TestChinese, "': ", tokenIds

      # Load model and run inference
      let model = loadModel(ModelPath)

      # AISHELL3 model (174 speakers, use sid=0)
      let output = runVitsTTS(
        model,
        tokenIds,
        sid = 108,
        noiseScale = 0.667'f32,
        lengthScale = 1.0'f32,
        noiseScaleW = 0.8'f32
      )

      check output.data.len > 0
      echo "Generated ", output.data.len, " audio samples (~", output.data.len / 8000, " seconds)"

      # Convert to int16
      let samples = output.toInt16Samples()

      # Save to WAV file (AISHELL3 uses 8000 Hz sample rate)
      let outputPath = TestDataDir / "test_output.wav"
      writeWavFile(outputPath, output.toInt16Samples(), 8000)
      check fileExists(outputPath)
      echo "Saved output to: ", outputPath

      model.close()

  test "Test with different speed (length scale)":
    if not fileExists(ModelPath) or not fileExists(LexiconPath) or not fileExists(TokensPath):
      echo "Skipping: Model, lexicon or tokens not found"
      skip()
    else:
      let lexicon = loadLexicon(LexiconPath)
      let tokens = loadTokens(TokensPath)
      let tokenIds = textToTokenIds("你好", lexicon, tokens)

      let model = loadModel(ModelPath)

      # Test with slower speed
      let outputSlow = runVitsTTS(
        model, tokenIds,
        lengthScale = 1.5'f32  # Slower
      )

      # Test with faster speed
      let outputFast = runVitsTTS(
        model, tokenIds,
        lengthScale = 0.7'f32  # Faster
      )

      # Slower should produce more samples
      check outputSlow.data.len > outputFast.data.len
      echo "Slow (1.5x): ", outputSlow.data.len, " samples"
      echo "Fast (0.7x): ", outputFast.data.len, " samples"

      model.close()
