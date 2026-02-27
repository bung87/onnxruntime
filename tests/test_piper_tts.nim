## test_piper_tts.nim
## Test Text-to-Speech inference using Piper voice model

import unittest
import json
import tables
import os

import onnxruntime, piper_utils

# Test paths
const TestDataDir = "tests/testdata/piper-voices"
const ModelPath = TestDataDir / "zh_CN-chaowen-medium.onnx"
const ConfigPath = TestDataDir / "zh_CN-chaowen-medium.onnx.json"

# JSON types for config parsing
type
  InferenceConfig = object
    noise_scale: float
    length_scale: float
    noise_w: float

  AudioConfig = object
    sample_rate: int

  PiperConfigJson = object
    audio: AudioConfig
    num_speakers: int
    inference: InferenceConfig
    hop_length: int
    phoneme_id_map: Table[string, seq[int64]]
    speaker_id_map: Table[string, int64]

# Static test phoneme sequence
const TestPhonemes = @[
  "^", "q", "i", "a", "n", "1", "w", "a", "n", "4", " ",
  "b", "u", "2", " ", "y", "a", "o", "4", " ", "w", "a", "n", "g", "4", " ",
  "j", "i", "4", " ", "j", "i", "e", "1", " ", "j", "i", "3", " ",
  "d", "o", "u", "4", " ", "zh", "e", "n", "g", "1", "$"
]

proc loadConfig(path: string): PiperConfigJson =
  ## Load Piper config from JSON file
  readFile(path).parseJson().to(PiperConfigJson)

proc phonemesToIds(phonemes: seq[string], phonemeMap: Table[string, seq[int64]]): seq[int64] =
  ## Convert phoneme sequence to ID sequence
  for phoneme in phonemes:
    if phonemeMap.hasKey(phoneme):
      result.add(phonemeMap[phoneme])
    elif phonemeMap.hasKey("_"):
      result.add(phonemeMap["_"])

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

suite "Piper TTS":
  test "Full pipeline - phonemes to WAV file":
    if not fileExists(ModelPath) or not fileExists(ConfigPath):
      skip()

    # Load config using JSON to type mapping
    let config = loadConfig(ConfigPath)
    check config.phoneme_id_map.len > 0

    # Load model and run inference
    let model = loadModel(ModelPath)
    let phonemeIds = phonemesToIds(TestPhonemes, config.phoneme_id_map)
    let output = runPiper(
      model, phonemeIds,
      noiseScale = config.inference.noise_scale.float32,
      lengthScale = config.inference.length_scale.float32,
      noiseW = config.inference.noise_w.float32,
      hasSpeakerId = false
    )
    check output.data.len > 0

    # Save to WAV file
    let outputPath = TestDataDir / "test_output.wav"
    writeWavFile(outputPath, output.toInt16Samples(), config.audio.sample_rate)
    check fileExists(outputPath)

    model.close()
