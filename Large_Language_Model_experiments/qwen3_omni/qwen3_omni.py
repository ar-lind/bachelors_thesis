import os
import warnings

import evaluate
import numpy as np
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor
from transformers import Qwen3OmniMoeForConditionalGeneration
from datasets import load_dataset

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# reference: https://github.com/QwenLM/Qwen3-Omni/blob/main/cookbooks/speech_recognition.ipynb

def _load_model_processor():
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL_PATH, device_map="auto", dtype='auto')
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    return model, processor


def run_model(model, processor, messages, return_audio, use_audio_in_video):
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
                       use_audio_in_video=use_audio_in_video)
    inputs = inputs.to(model.device).to(model.dtype)
    text_ids, audio = model.generate(**inputs,
                                        thinker_return_dict_in_generate=True,
                                        thinker_max_new_tokens=8192,
                                        thinker_do_sample=False,
                                        speaker="Ethan",
                                        use_audio_in_video=use_audio_in_video,
                                        return_audio=return_audio)
    response = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]
    if audio is not None:
        audio = np.array(audio.reshape(-1).detach().cpu().numpy() * 32767).astype(np.int16)
    return response, audio


def get_files(dir_path):
    files_list = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".wav"):
            files_list.append(filename)
    return files_list


if __name__ == '__main__':
    MODEL_PATH = "./"
    model, processor = _load_model_processor()
    USE_AUDIO_IN_VIDEO = False
    RETURN_AUDIO = False
    # change language ID and name
    LANG_ID = "apah1238"
    LANG_FULL = "Yali (Apahapsili)"

    # edit fieldwork data path
    files_l = get_files("./path/to/fieldwork/data/" + LANG_ID + "/audio/test/")
    preds = {}

    for file in files_l:
        audio_path = "/path/to/fieldwork/data/" + LANG_ID + "/audio/test/" + file

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "Transcribe this" + LANG_FULL + " audio."},
                ]
            }
        ]

        response, _ = run_model(model=model, messages=message, processor=processor, return_audio=RETURN_AUDIO,
                                use_audio_in_video=USE_AUDIO_IN_VIDEO)

        response_word_list = response.split()
        amt_words = len(response_word_list)
        if amt_words > 15:
            preds[file] = " ".join(response_word_list[:15])
        else:
            preds[file] = response

    preds = [v for k, v in sorted(preds.items())]

    ds = load_dataset("wav2gloss/fieldwork", split="test").select_columns(["id", "transcription", "language"])
    df = ds.to_pandas()
    df = df[df["language"] == LANG_ID]
    labels = dict(zip(df["id"], df["transcription"]))

    labels = [v for k, v in sorted(labels.items())]

    cer = evaluate.load("cer")
    with open("out/" + LANG_ID + "_qwen3_omni.txt", "w", encoding="utf-8") as f:
        for p, l in zip(preds, labels):
            f.write(p + "\n" +
                    l + "\n\n")
        f.write("cer: " + str(cer.compute(predictions=preds, references=labels)))
