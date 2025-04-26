from datasets import load_dataset, concatenate_datasets, Dataset
from huggingface_hub import snapshot_download, login as hf_login
import os
from dotenv import load_dotenv
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
import numpy as np

load_dotenv()

hf_login(os.getenv("HF_TOKEN"))

"""
for finetuning whisper on sound tags

finetuning possible:
https://huggingface.co/datasets/keithito/lj_speech -> single speaker
https://huggingface.co/datasets/badayvedat/VCTK

load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train") -> common voice english
"""

CPU_COUNT = os.cpu_count()
TARGET_SAMPLE_RATE = 16000 # whisper sample rate

whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")


ds_sounds = load_dataset("lmms-lab/vocalsound")
ds_sounds = concatenate_datasets([ds_sounds["val"], ds_sounds["test"]])

ds_speak = snapshot_download(
    repo_id="badayvedat/VCTK",
    repo_type="dataset",
    revision="main",
    max_workers=os.cpu_count(),
)
ds_speak = load_dataset("badayvedat/VCTK")
ds_speak = ds_speak["train"].select(range(len(ds_sounds)))

ds = []

for i in range(len(ds_sounds)):
    ds.append({
        "audio": ds_speak[i]["flac"]["array"],
        "sr": ds_speak[i]["flac"]["sampling_rate"],
        "text": ds_speak[i]["txt"]
    })
    ds.append({
        "audio": ds_sounds[i]["audio"]["array"],
        "sr": ds_sounds[i]["audio"]["sampling_rate"],
        "text": '(' + ds_sounds[i]["answer"].upper() + ')'
    })

dataset = Dataset.from_dict(ds)

def map_fn(batch):
    # Resample audio to target sample rate
    audio = librosa.resample(
        y=batch["audio"],
        orig_sr=batch["sr"],
        target_sr=TARGET_SAMPLE_RATE
    ).astype(np.float32).tolist()

    input_features = whisper_processor(
        audio=audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt"
    ).input_features[0].tolist()

    tokens = whisper_processor.tokenizer(batch["text"]).input_ids

    return {
        "input_features": input_features,
        "labels": tokens,
        "attention_mask": [1] * len(tokens)
    }

dataset = dataset.map(map_fn, num_proc=CPU_COUNT)

print(dataset[0])

dataset.push_to_hub("edwindn/whisper-tags-v1")
