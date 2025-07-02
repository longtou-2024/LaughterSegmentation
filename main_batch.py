import argparse
import json
import os
import os.path as osp
import io
from pathlib import Path
from time import time
from time import sleep
from collections import defaultdict

import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.utils import mediainfo
import safetensors
from scipy import signal
import torch
from transformers.trainer_utils import set_seed
import webdataset as wds
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
import torchaudio
#import soundfile as sf

#import sys
#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/train')
from evaluation._utils.utils import concat_close, remove_short
from train.model import Model

class TimeSegmentBoundary:
    def __init__(self, durations):
        self.durations = durations
        self.n_segment = len(durations)

        d_cum = np.cumsum([0] + durations)
        self.segments = list(zip(d_cum[:-1], d_cum[1:]))

    def __len__(self):
        return self.n_segment

    def get_index(self, time_slice):
        assert len(time_slice) == 2
        idx = None
        for i, seg in enumerate(self.segments):
            if seg[0] <= time_slice[0] < seg[1]:
                # NOTE(longtou): this may occur, at sample boundary; <laugh> + <sli> region?
                if time_slice[1] > seg[1]:
                    # method 1)
                    break # skip this slice
                    # method 2)
                    #if time_slice[1] > seg[1] + 1:
                    #    # NOTE(longtou): exceed large margin more than 1s
                    #    raise Exception(f"{time_slice} exceed boundary: {seg}")
                    #else:
                    #    # clip so that it does not exceed sample boundary
                    #    time_slice[1] = seg[1]
                idx = i
                break
        return idx

    def revise_time_slice(self, time_slice, idx):
        s_time, e_time = time_slice
        seg_s_time, _ = self.segments[idx]
        s_time -= seg_s_time
        e_time -= seg_s_time
        return s_time, e_time

class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def __iter__(self):
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)

def concat_until_max_dur(dataset, max_dur=60):
    buf = []
    dur_sum = 0
    for sample in dataset:
        if "mp3" in sample:
            #audio = AudioSegment.from_file(io.BytesIO(sample["mp3"]), format="mp3")
            audio_tensor, sample_rate = torchaudio.load(io.BytesIO(sample["mp3"]), format="mp3")
        elif "wav" in sample:
            #audio = AudioSegment.from_file(io.BytesIO(sample["wav"]), format="wav")
            audio_tensor, sample_rate = torchaudio.load(io.BytesIO(sample["wav"]), format="wav")
        else:
            raise Exception

        this_dur = audio_tensor.size(1) / sample_rate
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        audio_tensor = transform(audio_tensor)
        audio_mono = torch.mean(audio_tensor, dim=0, keepdim=False)
        audio_np = audio_mono.numpy()
        audio_np = custom_amplituder_small_portion(audio_np, 16000)
        
        sample["audio_np"] = audio_np
        sample["sample_rate"] = sample_rate
        sample["this_dur"] = this_dur

        if dur_sum + this_dur < max_dur:
            dur_sum += this_dur
            buf.append(sample)
        else:
            yield buf
            # restart
            buf = [sample]
            dur_sum = this_dur
    if len(buf) > 0:
        yield buf


# convert mp3 to wav io
def convert_mp3_bytes_to_wav_io(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

def merge_events(event_lists):
    merged_events = {}
    merged_event_idx = 0
    has_merged = False
    for event_list in event_lists:
        for event in event_list.values():
            if not merged_events:
                # If merged_events is empty, add the first event
                merged_events[str(merged_event_idx)] = event.copy()
                merged_event_idx += 1
            else:
                merged = False
                for merged_event in merged_events.values():
                    if event["start_sec"] <= merged_event["end_sec"] and event["end_sec"] >= merged_event["start_sec"]:
                        # Events overlap, merge them
                        merged_event["start_sec"] = min(event["start_sec"], merged_event["start_sec"])
                        merged_event["end_sec"] = max(event["end_sec"], merged_event["end_sec"])
                        merged = True
                        has_merged = True
                        # break
                if not merged:
                    # If the event does not overlap with any merged event, add it to merged_events
                    merged_events[str(merged_event_idx)] = event.copy()
                    merged_event_idx += 1
    if has_merged:
        merged_events = merge_events([merged_events])
    merged_events = sorted(merged_events.values(), key=lambda x: x["start_sec"])
    merged_events = {str(idx): val for idx, val in enumerate(merged_events)}
    return merged_events

# bandpass
def bandpass(x, samplerate, fp=np.array([1000,3000]), fs=np.array([1000,3000]), gpass=3, gstop=40):
    fn = samplerate / 2 # nyquist frequency
    wp = fp / fn  # normalizing the passband frequency by the Nyquist frequency
    ws = fs / fn  # normalizing the stopband frequency by the Nyquist frequency
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  # calculate the order and normalized frequency of the Butterworth
    b, a = signal.butter(N, Wn, "band") # calculate the numerator and denominator of the filter transfer function
    y = signal.filtfilt(b, a, x) # filter the signal
    return y

def custom_amplituder_small_portion(array, sr, mul_fac=5):
    # 32767 is max value of signed short
    dub_audio = AudioSegment(
                (array*32767).astype("int16").tobytes(), 
                sample_width=2, 
                frame_rate=sr, 
                channels=1,
                )
    
    dub_audio = dub_audio.set_frame_rate(sr)
    silent_section = detect_silence(dub_audio, min_silence_len=270, silence_thresh=-35)

    sr_mul = sr // 1000
    for sec in silent_section:
        fade_len = int(sr*.15) # 0.15 sec
        if (sec[1]-sec[0])*sr_mul > (fade_len*2):
            array[sec[0]*sr_mul: sec[0]*sr_mul + fade_len] *= np.linspace(1, mul_fac, fade_len)
            array[sec[0]*sr_mul + fade_len: sec[1]*sr_mul - fade_len] *= mul_fac
            if sec[1]*sr_mul < len(array):
                array[sec[1]*sr_mul - fade_len: sec[1]*sr_mul] *= np.linspace(mul_fac, 1, fade_len)
        else:
            array[sec[0]*sr_mul: sec[1]*sr_mul] *= mul_fac
    array = librosa.util.normalize(array)
    return array

def segment_laughing(model, sample_concat, sr, batch_size, input_sec, over_lap_sec, min_laugh_len=0.5, laugh_prob=0.8):
    with torch.no_grad():
        laughter = {}
        laughter_idx = 0

        audio_concat = [sample["audio_np"] for sample in sample_concat]
        audio_concat = np.concatenate(audio_concat, axis=0)
        audio_array = audio_concat

        #audio_array = librosa.load(wav_io, sr=sr, mono=True)[0]
        #audio_array = custom_amplituder_small_portion(audio_array, sr)

        # get each array of 7 sec 
        for array_idx in range(0, len(audio_array), int(sr*(input_sec-over_lap_sec))*batch_size):
            batched_arrays = []
            should_break = False
            for batch_idx in range(batch_size):
                array = audio_array[array_idx+batch_idx*int(sr*(input_sec-over_lap_sec)): array_idx+batch_idx*int(sr*(input_sec-over_lap_sec))+sr*input_sec]
                if len(array) < sr*input_sec:
                    # fill 0 to the end of array
                    array = np.append(array, np.zeros(sr*input_sec-len(array)))
                    should_break = True
                batched_arrays.append(array)
                if should_break:
                    break

            input_values = torch.from_numpy(np.array(batched_arrays)).type(torch.FloatTensor)
            outputs = model(input_values=input_values)

            logits = outputs[1]

            #  --- predict ends ---

            preds = torch.sigmoid(logits.to(torch.float32))

            for batch_idx, pred in enumerate(preds): # each batch
                frame_pred = list(map(round, pred.cpu().tolist(), [3]*len(pred)))

                # change to 0, 1
                frame_pred = (np.array(frame_pred)>=laugh_prob).astype(int)

                batch_start_sec = (array_idx+batch_idx*int(sr*(input_sec-over_lap_sec)))/float(sr)
                frame_count = len(frame_pred)
                start_idx = None
                end_idx = None
                status = "not_laughing"
                for idx, frame in enumerate(frame_pred):
                    if frame == 1:
                        if status == "not_laughing":
                            start_idx = idx
                            status = "laughing"

                        # if the last frame is laughing
                        if status == "laughing" and idx == frame_count-1:
                            laughter[str(laughter_idx)] = {
                                "start_sec": batch_start_sec + (input_sec/frame_count)*start_idx,
                                "end_sec": batch_start_sec + input_sec,
                            }
                            laughter_idx += 1
                            start_idx = None
                            end_idx = None
                    elif frame == 0:
                        # end of laughter
                        if status == "laughing":
                            end_idx = idx
                            status = "not_laughing"
                            if start_idx == 0:
                                laughter[str(laughter_idx)] = {
                                "start_sec": batch_start_sec + (input_sec/frame_count)*start_idx,
                                "end_sec": batch_start_sec + (input_sec/frame_count)*end_idx,
                                }
                            else:
                                laughter[str(laughter_idx)] = {
                                    "start_sec": batch_start_sec + (input_sec/frame_count)*start_idx,
                                    "end_sec": batch_start_sec + (input_sec/frame_count)*end_idx,
                                }
                            laughter_idx += 1
                            start_idx = None
                            end_idx = None

        # >>> from here escape for loop (slide audio time axis)
        if over_lap_sec > .0:
            laughter = merge_events([laughter])

        laughter = concat_close(laughter, 0.2)
        laughter = remove_short(laughter, min_laugh_len)

        laughter_json = None
        if len(laughter) != 0:
            laughter_json = laughter

        return sample_concat, laughter_json

# NOTE(longtou): batch_size=100 -> 5s * 100 -> 6m?
def main(args, input_sec=7, batch_size=100):
    audio_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

    model_path = args.model_path

    sr = 16000
    seed = 42
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

    over_lap_sec = 2.
    assert input_sec > over_lap_sec

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Model(audio_model_name, device, sr).to(device)

    if not osp.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Download the model file and place it in the specified path.")
    state_dict = safetensors.torch.load_file(model_path, device.index if device.type=="cuda" else "cpu")
    # state_dict = torch.load(model_path) # use when model is .bin format
    model.load_state_dict(state_dict)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model.eval()

    writer = wds.ShardWriter(f"{args.output_dir}/shard-%06d.tar",
                             maxsize=1e9,
                             )

    dataset = wds.WebDataset(args.shard_url)
    dataset = Processor(dataset, concat_until_max_dur, max_dur=args.max_dur)
    data_loader = DataLoader(dataset, batch_size=None, num_workers=args.nj, prefetch_factor=100, pin_memory=True)
    inputs = {"sr": sr, "batch_size": batch_size,
              "input_sec": input_sec, "over_lap_sec": over_lap_sec,
              "min_laugh_len": args.min_laugh_len,"laugh_prob": args.laugh_prob,
              }

    for idx, sample in tqdm(enumerate(data_loader)):
        #if idx > 200: break
        result_concat, laughter_json = segment_laughing(model, sample, **inputs)

        if laughter_json:
            time_seg_bound = TimeSegmentBoundary([sample['this_dur'] for sample in result_concat])
            sample_idx_to_laugh = defaultdict(list)
            for time_slice_dict in laughter_json.values():
                time_slice = time_slice_dict["start_sec"], time_slice_dict["end_sec"]
                sample_idx = time_seg_bound.get_index(time_slice)
                if sample_idx:
                    s_time, e_time = time_seg_bound.revise_time_slice(time_slice, sample_idx)
                    sample_idx_to_laugh[sample_idx].append((s_time, e_time))

            # write each sample
            for sample_idx, time_slices in sample_idx_to_laugh.items():
                example = result_concat[sample_idx]
                this_laughter_json = {}
                # collect each laugh
                for laugh_idx, time_slice in enumerate(time_slices):
                    this_laughter_json[laugh_idx] = {"start_sec": time_slice[0], "end_sec": time_slice[1]}
                example["laughter.json"] = json.dumps(this_laughter_json)
                del example["audio_np"]
                del example["sample_rate"]
                del example["this_dur"]
                writer.write(example)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_url', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="wds_laughter")
    parser.add_argument('--model_path', type=str,
        default="/home/longtou.2024/mount/longtou/saved/LaughterSegmentation/models/model.safetensors")
    parser.add_argument('--min_laugh_len', default=0.5, type=float)
    parser.add_argument('--laugh_prob', default=0.8, type=float)
    parser.add_argument('--nj', default=2, type=int)
    parser.add_argument('--max_dur', default=60, type=int, help="max audio duration of batch in sec")
    args = parser.parse_args()

    main(args)
