from datasets import load_dataset, concatenate_datasets, Dataset
from huggingface_hub import snapshot_download, login as hf_login
import os
from dotenv import load_dotenv
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
import numpy as np
from tqdm import tqdm

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

print(0)
whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

print(1)
ds_sounds = load_dataset("lmms-lab/vocalsound")
print(1.1)
ds_sounds = concatenate_datasets([ds_sounds["val"], ds_sounds["test"]])
print(1.2)
ds_speak = snapshot_download(
    repo_id="badayvedat/VCTK",
    repo_type="dataset",
    revision="main",
    max_workers=os.cpu_count(),
)
print(2)
ds_speak = load_dataset("badayvedat/VCTK")
ds_speak = ds_speak["train"].select(range(len(ds_sounds)))
print(3)
ds = []

#for i in tqdm(range(len(ds_sounds))):
for i in range(10):
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
print(4)

# ?
ds_dict = {
    "audio": [item["audio"] for item in ds],
    "sr": [item["sr"] for item in ds],
    "text": [item["text"] for item in ds]
}
dataset = Dataset.from_dict(ds_dict)
# -----

print(5)
def map_fn(batch):
    # Resample audio to target sample rate
    audio = librosa.resample(
        y=np.array(batch["audio"]),
        orig_sr=batch["sr"],
        target_sr=TARGET_SAMPLE_RATE
    ).astype(np.float32).tolist()

    input_features = whisper_processor(
        audio=audio,
        sampling_rate=TARGET_SAMPLE_RATE,
    ).input_features[0]

    tokens = whisper_processor.tokenizer(batch["text"]).input_ids

    return {
        "input_features": input_features,
        "labels": tokens,
        "attention_mask": [1] * len(tokens)
    }

dataset = dataset.map(map_fn, num_proc=1)
print(6)
print(dataset[0])
print(7)
quit()
dataset.push_to_hub("edwindn/whisper-tags-v1")
