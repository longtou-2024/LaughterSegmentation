from pathlib import Path
import time
import argparse
import io
import json
from collections import Counter

from tqdm import tqdm
import torch
import torchaudio
import webdataset as wds
import whisper

SAMPLE_RATE = 16000

import torch
import numpy as np
import random

def remove_punctuation(text, punctuation=['.', ',', '?', '!', '"', "'", ' ']):
     return ''.join([char for char in text if char not in punctuation])

def set_seed(seed):
    # Set the random seed for PyTorch (CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed for Python's built-in random module
    random.seed(seed)

    # Enable deterministic algorithms in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_audio(audio, time_slice):
    s_time, e_time = time_slice
    s_idx, e_idx = int(s_time * SAMPLE_RATE), int(e_time * SAMPLE_RATE)
    left = audio[:s_idx]
    right = audio[e_idx:]
    return left, right

parser = argparse.ArgumentParser()
parser.add_argument("--shard_url", default="wds_laughter/shard-000000.tar")
parser.add_argument("--outdir", default="wds_laughter_tag")
args = parser.parse_args()

model = whisper.load_model("turbo", device="cuda")

Path(args.outdir).mkdir(exist_ok=True)
f_log = open("log.txt", 'w')
writer = wds.ShardWriter(f"{args.outdir}/shard-%06d.tar", maxsize=1e9)
dataset = wds.WebDataset(args.shard_url)
#cnt = Counter()
for sample in tqdm(dataset):
    if "mp3" in sample:
        audio, fmt = sample["mp3"], "mp3"
    elif "wav" in sample:
        audio, fmt = sample["wav"], "wav"
    else:
        raise Exception()

    audio_tensor, sample_rate = torchaudio.load(io.BytesIO(audio), format=fmt)
    transform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
    audio_tensor = transform(audio_tensor)
    audio_mono = torch.mean(audio_tensor, dim=0, keepdim=False)

    # NOTE(longtou): too many samples of asr result from emilia_pipe is different 
    # so here, transcribe mutliple times
    #json_data = json.load(io.BytesIO(sample["json"]))
    #full_text = json_data["text"].strip()

    laughter_json = json.load(io.BytesIO(sample["laughter.json"]))
    time_slices = [(x['start_sec'], x['end_sec']) for x in laughter_json.values()]

    # deal with only single event
    if len(time_slices) > 1: continue
    assert len(time_slices) == 1
    set_seed(42)
    full_text = model.transcribe(audio=audio_mono, language="ko", condition_on_previous_text=False)["text"].strip()
    audio_l, audio_r = split_audio(audio_mono, time_slices[0])
    audio_wo_laugh = torch.cat([audio_l, audio_r], dim=0)
    set_seed(42)
    text_wo_laugh = model.transcribe(audio=audio_wo_laugh, language="ko", condition_on_previous_text=False)["text"].strip()
    if remove_punctuation(text_wo_laugh) == remove_punctuation(full_text): # insert tag
        audio_seg_long = max([audio_l, audio_r], key=lambda x: len(x))
        set_seed(42)
        text_seg = model.transcribe(audio=audio_seg_long, language="ko", condition_on_previous_text=False)["text"].strip()

        eojeol_split = text_seg.split(" ")
        first_three_eojeol = " ".join(eojeol_split[:3])
        last_three_eojeol = " ".join(eojeol_split[-3:])
        if len(audio_l) > len(audio_r): # asr on left
            insert_idx = full_text.find(last_three_eojeol)
            if insert_idx != -1:
                insert_idx += len(last_three_eojeol) - 1
        else: # asr on right
            insert_idx = full_text.rfind(first_three_eojeol)

        if insert_idx == -1:
            msg = f"{sample['__key__']}: asr not correct\n"
            msg += f"\tfull_text: {full_text}\n"
            msg += f"\ttext_seg: {text_seg}\n"
            msg += f"\ttext_remain: {full_text.replace(text_seg, '')}\n"
            f_log.write(msg)
            continue

        transcript = full_text[:insert_idx].strip() + " " + "[laughter]" + " " + full_text[insert_idx:].strip()
        #import pdb; pdb.set_trace()
    else: # pad tag
        s_idx, e_idx = None, None
        full_text_eojeol = full_text.split(' ')
        text_wo_laugh_eojeol = text_wo_laugh.split(' ')
        for idx, (x, y) in enumerate(zip(full_text_eojeol, text_wo_laugh_eojeol)):
            if x != y:
                s_idx = idx
                break
        if s_idx:
            _stop = -min(len(full_text_eojeol), len(text_wo_laugh_eojeol))
            for idx in range(-1, _stop-1, -1): # search from last idx
                e_idx = idx
                if full_text_eojeol[idx] != text_wo_laugh_eojeol[idx]:
                    break # found e_idx
            # NOTE(longtou): even if not found, e_idx could be valid
            # translate minus -> plus idx
            e_idx = len(full_text_eojeol) + e_idx + 1
            if e_idx <= s_idx:
                e_idx = None

        if s_idx is None or e_idx is None:
            msg = f"{sample['__key__']}: segmentation not correct\n"
            msg += f"\tfull_text: {full_text}\n"
            msg += f"\ttext_wo_laugh: {text_wo_laugh}\n"
            f_log.write(msg)
            continue
        elif len(full_text_eojeol[s_idx:e_idx]) > 2:
            msg = f"{sample['__key__']}: segmentation not correct\n"
            msg += f"\tfull_text: {full_text}\n"
            msg += f"\ttext_wo_laugh: {text_wo_laugh}\n"
            msg += f"\ttext_sliced: {' '.join(full_text_eojeol[s_idx:e_idx])}\n"
            f_log.write(msg)
            continue
        else:
            text_tobe_tag = " ".join(full_text_eojeol[s_idx:e_idx])
            text_tobe_tag = "<laughter>" + text_tobe_tag + "</laughter>"
            transcript = " ".join(full_text_eojeol[:s_idx]) + " " + text_tobe_tag + " " + " ".join(full_text_eojeol[e_idx:])

            #import pdb; pdb.set_trace()

    laughter_json["transcript"] = transcript
    sample["laughter.json"] = json.dumps(laughter_json, ensure_ascii=False)
    writer.write(sample)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# NOTE(longtou): if consider multi segmentation
## cut off laugh part
#audio_wo_laugh = []
#audio_r = audio_mono
#for s_time, e_time in time_slices:
#    audio_l, audio_r = split_audio(audio_r, (s_time, e_time))
#    audio_wo_laugh.append(audio_l)
#audio_wo_laugh.append(audio_r)
#audio_wo_laugh = torch.cat(audio_wo_laugh, dim=0)
#result = model.transcribe(audio=audio_wo_laugh, language="ko")
