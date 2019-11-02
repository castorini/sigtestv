python -m sigtestv.run.sigseed_jiant_glue\
    -pp /mnt/hdd/jiant\
    -d /mnt/hdd/glue\
    -emb /mnt/hdd/glove/glove.840B.300d.txt\
    --seed-iter 1000\
    --task-name sst\
    --learning-rate $2\
    --patience 10000\
    --transfer-nonstatic\
    -l jiant-log-sst-$1.jsonl
