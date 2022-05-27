python semi_train.py \
   --train_1 path_to_SynthText \
   --train_2 path_to_Synth90k \
   --valid_data path_to_validation_data \
   --eval_data eval_and_val/evaluation \
   --eval_type simple \
   --batchSize 32 \
   --model_name TRBA \
   --exp_name test \
   --Aug rand \
   --Aug_semi rand \
   --semi None \
   --num_iter 250000 \
   --workers 4 \
   --optimizer adam \
   --lr 0.001 \
   --saved_model saved_models/trba_baseline.pth \
   --mode test
   