export GLUE_DIR=/mnt/nvme/orig-glue
export TASK_NAME=SST-2

python -m sigtestv.run.sigseed_finetune_glue \
  --seed-range 100 200 \
  --model-type bert \
  --model-name-or-path bert-base-uncased \
  --task-name $TASK_NAME \
  --data-dir $GLUE_DIR \
  --learning-rate 4e-5 \
  --output-dir /mnt/hdd/run
