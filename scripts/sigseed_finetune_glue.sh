export GLUE_DIR=/mnt/nvme/orig-glue
export TASK_NAME=SST-2

python -m sigtestv.run.sigseed_finetune_glue \
  --seed-range $1 $2 \
  --model-type bert \
  --model-name-or-path bert-base-uncased \
  --task-name $TASK_NAME \
  --data-dir $GLUE_DIR \
  --learning-rate 3e-5 \
  --output-dir /mnt/hdd/run \
  --logger-endpoint http://hydra.cs.uwaterloo.ca:8080/submit
