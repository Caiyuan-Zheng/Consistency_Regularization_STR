python semi_train.py \
   --train_1 <path_to_SynthText> \
   --train_2 <path_to_Synth90k \
   --valid_data <path_to_validation_data> \
   --eval_data <path_to_evaluation_data> \
   --model_name TRBA \
   --exp_name baseline_exp \
   --semi None \
   --workers 4 \
   --optimizer adam \
   --lr 0.001 
   