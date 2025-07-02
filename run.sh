#!/usr/bin/env bash
# longtou.2024

#shard_url="gs://ai-lab-speech-bucket/longtou/db/mediazen_teen/emilia_pipe_v2/shard-000{000,100}.tar"
shard_url="/home/longtou.2024/mount/longtou/db/mediazen_teen/emilia_pipe_v2/shard-000{000,100}.tar"
output_dir="wds_laughter"
min_laugh_len=0.5
laugh_prob=0.8
nj=2
max_dur=60

python main_batch.py --shard_url ${shard_url} --output_dir $output_dir --min_laugh_len $min_laugh_len --laugh_prob $laugh_prob --max_dur $max_dur --nj $nj

#gcloud storage cp -r $output_dir gs://ai-lab-speech-bucket/longtou/tmp/
