#!/usr/bin/env bash
# longtou.2024

shard_url="gs://ai-lab-speech-bucket/longtou/db/mediazen_teen/emilia_pipe_v2/shard-000{000,100}.tar"
python main_batch.py --shard_url ${shard_url}
