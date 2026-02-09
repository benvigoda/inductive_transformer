## Attribution

If you are reading this, then you already have a very deep understanding of AI. Please contact us, we would like to talk to you!

Anyone using, modifying, extending, or distributing this software or derivative works must provide clear attribution to the original authors as specified in the NOTICE file.



Command to run inference, inference_text.tx is the prompt:

PYTHONPATH=. python jax_transformer/train.py training_data/48_6_layer_sentences_balanced_dogs_birds_all_synonyms.txt --prompt_text inference_data/inference_text.txt --num_layer 6 --layer_width 2 --num_samples 10 --num_epochs 0 --silence_print --seed 2768615008 --initialize_weights


Command to run training:

PYTHONPATH=. python jax_transformer/train.py training_data/dog_only_training_sentences.txt  --prompt_text inference_data/inference_text.txt --num_layer 6 --layer_width 2 --num_samples 20 --num_epochs 2 --silence_print --seed 2768615008 --initialize_weights

PYTHONPATH=. python jax_transformer/train.py training_data/48_6_layer_sentences_balanced_dogs_birds_all_synonyms.txt  --prompt_text inference_data/inference_text.txt --num_layer 6 --layer_width 2 --num_samples 20 --num_epochs 2 --silence_print --seed 2768615008 --initialize_weights



time PYTHONPATH=. python jax_transformer/train.py training_data/48_6_layer_sentences_balanced_dogs_birds_all_synonyms.txt --prompt_text inference_data/inference_text_6_words_sentence.txt --num_layer 6 --layer_width 2 --num_samples 10 --num_epochs 1000 --silence_print

time PYTHONPATH=. python jax_transformer/train.py training_data/366_6_layer_sentences_dogs_worms.txt --prompt_text inference_data/inference_text_6_and_2_words_sentences.txt --num_layer 6 --layer_width 2 --num_samples 12 --num_epochs 5000 --silence_print


time PYTHONPATH=. python jax_transformer/train.py training_data/128_6_layer_sentences_dogs_worms.txt --prompt_text inference_data/inference_text_dog_worm.txt --num_layer 6 --layer_width 2 --num_samples 12 --num_epochs 500 --silence_print --initialize_weights
