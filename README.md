## Attribution

Anyone using, modifying, extending, or distributing this software or derivative works must provide clear attribution to the original authors as specified in the NOTICE file.


Command to run inference, inference_text.tx is the prompt:

PYTHONPATH=. python jax_transformer/train.py training_data/48_6_layer_sentences_balanced_dogs_birds_all_synonyms.txt --prompt_text inference_data/inference_text.txt --num_layer 6 --layer_width 2 --num_samples 10 --num_epochs 0 --silence_print --seed 2768615008 --initialize_weights


Command to run training:

PYTHONPATH=. python jax_transformer/train.py training_data/dog_only_training_sentences.txt  --prompt_text inference_data/inference_text.txt --num_layer 6 --layer_width 2 --num_samples 20 --num_epochs 2 --silence_print --seed 2768615008 --initialize_weights