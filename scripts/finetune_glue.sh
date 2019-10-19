export GLUE_DIR=/mnt/nvme/orig-glue
export TASK_NAME=SST-2

python -m sigtestv.run.finetune_bert_glue \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --seed $SEED \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 4e-5 \
  --num_train_epochs 3.0 \
  --save_steps 4000 \
  --overwrite_output_dir \
  --output_dir $OUTPUT_DIR
