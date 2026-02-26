


# The computer decides:
# Randomly choose a word from vocab
# Randomly choose a column, layer and position
# If word isn’t already in word_list
# add it
# We watch the histograms converge as we turn up N
# For loop N = 1:10:100 (take the same word and add it multiple places)

batch_perturbation_experiments():
    num_word_adds = 100
    skip = 10
    print 
    for i in range(1, skip, num_word_adds+1):
        # call this command that I have been calling from the terminal window:
        '''
        this is to see if more wriggly training data helps move wriggly and amplify it on right side:
        time PYTHONPATH=. python jax_transformer/train.py training_data/128_6_layer_sentences_dogs_worms.txt --prompt_text inference_data/inference_text.txt --num_layer 6 --layer_width 2 --num_samples 100 --num_epochs 200 --silence_print --initialize_weights --batch_move num_word_adds
        '''

# in weights.py, we want to add a new function that does these things

# batch_pertubation(num_word_adds:int, vocab)
#     for i in range(1, skip, num_word_adds+1):
#         Randomly choose an integer column from uniform distribution between 0 and 1, inclusive
#         Randomly choose an integer layer from uniform distribution between 0 and 5, inclusive
#         Randomly choose an integer position from uniform distribution between 1 and 5, inclusive
#         Randomly choose a word from vocab which is available if we thread it through from train.py
#         synonym_list =  figure out how to get the SynonymList for those indices column, layer, position
#         if word is not in synonym_list: 
#             # synonym_list_name:str = ask the synonym_list for its name as a string that we can use
#             SynonymList(synonym_list_name, layer, position, column, anavan.get_synonyms_of_word(synonym_list_name, add_words={word})),  # noqa: E241

        
