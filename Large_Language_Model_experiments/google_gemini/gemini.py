import requests
import json
import os
import yaml
import base64
from datasets import load_dataset

with open(os.path.join("./configs", "config.yaml")) as f:
    config = yaml.safe_load(f)


def audio_prep(audio: dict):
    """
    reference: https://openrouter.ai/docs/guides/overview/multimodal/audio
    :param: audio
    :return: encoded audio
    """
    path = "audios/" + config["language_tag"] + "/audio/test/" + audio["path"]
    with open(path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')


ds = load_dataset("wav2gloss/fieldwork", split="test").select_columns(["id", "transcription", "audio", "language"])
ds = ds.filter(lambda x: x["language"] == config["language_tag"])
examples = ds[:5]

for i in range(0, 5):
    response = requests.post(
      url="https://openrouter.ai/api/v1/chat/completions",
      headers={
        "Authorization": "Bearer " + config["api_key"],
      },
      data=json.dumps({
        "model": "google/gemini-2.5-flash-lite",
        "messages": [
          {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Create an IPA (International Phonetic Alphabet) Transcription of the following" +
                            config["language"] + "audio. The output should be in the following format: " +
                            examples["id"][i] + " <ipa_transcription>."
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_prep(examples["audio"][i]),
                        "format": "wav"
                    }
                }
            ]
          }
        ]
      })
    )

    print(response.json())
    os.makedirs("out", exist_ok=True)
    with open("out/" + config["language"] + "_gemini_" + str(i) + ".json", "w", encoding="utf-8") as f:
        json.dump(response.json(), f)


# PROMPTS:
#
# "Create an IPA (International Phonetic Alphabet) Transcription of the following " +
# config["language"] + " audio. The output should be in the following format: " +
# ex1["id"][i] +
# " <ipa_transcription>."
#
# "Create an IPA (International Phonetic Alphabet) Transcription of this audio file. +
# "The output should be in the following format: " +
# ex1["id"][i] +
# " <transcription>."
#
# Transcribe this Persian audio file in Persian (Farsi) using the Latin alphabet. " +
# "The output should be in the following format: " +
# ex1["id"][i] +
# " <transcription>."
#
# Transcribe this Yali (Apahapsili) audio file in Yali (Apahapsili) using the Latin alphabet. " +
# "The output should be in the following format: " +
# ex1["id"][i] +
# " <transcription>."
