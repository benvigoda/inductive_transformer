import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import numpy as np  # type: ignore
from jax_transformer.helper_functions import EPSILON, get_num_layers
from inductive_transformer.datasets.anavan import make_cat_dog_anavan, make_cat_dog_worm_bird_anavan  # type: ignore

strong = jnp.log(1.0 - EPSILON)  # Amplify the signal
weak = jnp.log(EPSILON)  # Dampen the signal


'''
to write this, use example code weights_broad_init.py
'''

'''
command line arguments:
noise_variance = 0.1
'''



class WeightTreeMapKeys:
    """ All possible index values
    encoder_decoder = ["encoder", "decoder"]
    factor = ["position", "token", "attention"]
    layer_width_idx = [0, 1] #left, right
    position_idx = [0, 1, 2, 3, 4, 5]
    token_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ... 53]
    """
    def __init__(self, encoder_decoder, layer, factor, position_idx, token_idx, noise_value):
        self.encoder_decoder = encoder_decoder
        self.layer = layer
        self.factor = factor
        self.position_idx = position_idx
        self.token_idx = token_idx
        self.noise_value = noise_value



# create the treemap for all weights

set_all_random():
    pass

set_all_weak():
    #in future this could be more like set_strong, a subset of all weights
    #for now, it's all weights overwriting all random
    init all weights = WEAK


set_attention_weights():
    # use the existing code for this - there's a simple rule we don't need a dictionary
    weight_to_set = attention[layer_index][position, token]
    1's for attention weights that go straight up and 0's for attention weights that cross over

set_position_weights(synoyms):
    # position STRONGs:
    # in layer=5, layer_width=0 or 1, the position=0 is STRONG
    # in layer=4, layer_width=0 or 1, the position=1 is STRONG
    ...
    # in layer=0, layer_width=0 or 1, the position=5 is STRONG

    
class synomym_list():
    name
    layer
    layer_width_idx
    token_list
    
class synonyms():
    __init__():
        # synoyms of small at if layer=5, layer_width=0, 
        new synomym_list(name=small, layer = 5, layer_width_idx = 0, token_list = fron_anavan)
        
        # if layer=4, layer_width=0, if synoyms of dog, set weights = STRONG
        # if layer=3, layer_width=0, if synoyms of often set weights =  STRONG
        # if layer=2, layer_width=0, if synoyms of fear set weights =  STRONG
        # if layer=1, layer_width=0, if synoyms of large set weights =  STRONG
        # if layer=0, layer_width=0, if synoyms of cats set weights =  STRONG

        # if layer=5, layer_width=1, if synoyms of wriggly set weights =  STRONG
        # if layer=4, layer_width=1, if synoyms of worms set weights =  STRONG
        # if layer=3, layer_width=1, if synoyms of sometimes set weights =  STRONG
        # if layer=2, layer_width=1, if synoyms of chase set weights =  STRONG
        # if layer=1, layer_width=1, if synoyms of angry set weights =  STRONG
        # if layer=0, layer_width=1, if synoyms of birds set weights =  STRONG


set_token_weights():
    for synonym_list in synonym_lists:
        for token in synonym_list.tokens
            weights[synonym_list.layer, synonym_list.layer_width_idx, token] = STRONG


add_perturbations():
    loop idx:
        if idx is in user_perturb_indices:
            weights[idx] += noise

set_weights():
    # set_all_random()
    set_all_weak()
    set_attention_weights()
    set_position_weights()
    set_token_weights()
    add_perturbations()
'''


