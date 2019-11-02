export GLUE_DIR=/mnt/hdd/glue
export TASK_NAME=STS-B

python -m sigtestv.run.finetune_glue \
  --model_type xlnet \
  --model_name_or_path xlnet-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --seed $SEED \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --save_steps 4000 \
  --overwrite_output_dir \
  --no_reload_after_save \
  --no_save_checkpoint \
  --output_dir $OUTPUT_DIR
