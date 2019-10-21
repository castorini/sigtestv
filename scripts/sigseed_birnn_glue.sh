export GLUE_DIR=/mnt/nvme/orig-glue
export TASK_NAME=SST-2

python -m sigtestv.run.sigseed_birnn_glue \
  --seed-range $1 $2 \
  --task-name $TASK_NAME \
  --output-dir /mnt/hdd/run-birnn \
  --log-file birnn-sst2-$1-$2.jsonl \
  --logger-endpoint http://hydra.cs.uwaterloo.ca:8080/submit
