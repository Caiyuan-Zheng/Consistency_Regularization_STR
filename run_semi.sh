
python semi_train.py \
   --train_1 <path_to_SynthText> \
   --train_2 <path_to_Synth90k> \
   --unl_train_1 <path_to_unlabeled_dataset> \
   --valid_data <path_to_validation_data> \
   --eval_data <path_to_evaluation_data> \
   --eval_type simple \
   --model_name TRBA \
   --exp_name semi_exp \
   --Aug rand \
   --Aug_semi rand \
   --semi KLDiv \
   --workers 4 \
   --unl_workers 4 \
   --optimizer adamw \
   --weight_decay 0.01 \
   --lr 0.001 
   
