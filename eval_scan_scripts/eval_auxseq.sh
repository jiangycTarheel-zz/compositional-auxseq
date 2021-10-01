export EVAL_EPOCH=5000

python3 evaluate_scan.py \
--config_name=t5-2l-2h \
--model_name_or_path=t5-small-dualemb-countaction \
--run_id=auxseq-00 \
--evaluate_epoch=$EVAL_EPOCH \
--batch_size=1024 \
--num_beams=1 \
--data_split=test \
--trained_on_task=mcd1 \
--eval_task=mcd1 \
--predict_action_count \
--predict_action_group \
--predict_action_count_mode=MT \
--action_count_attention_kv=f,c

#--use_gt_action_count_ids \  Use this option if you want to feed in the ground-truth auxseq1
#--use_gt_action_group_ids \  Use this option if you want to feed in the ground-truth auxseq2
