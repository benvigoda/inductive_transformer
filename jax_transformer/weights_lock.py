command line args:
--lock_weights dictonary
--loose_weights dictionary

mask_type = jnp.float32

weights_lock():

'''
# ":" means use all indices
lock_weight_coordinates = {
    "type": "lock",
    "path": "weights",
    "encoder_decoder": ":",
    "layer": ":",
    "factor": ["position", "token", "attention"],
    "position_idx": [0, 1, 2, 3, 4, 5, 6],
    "token_idx": ":",
    "layer_width": 1,
    "noise_value": 0.1,
}

# if given a lock_weights coordinates then this mask is 1's everywhere,
except 0's at those coordinates
if lock
    set all weights = 1
    loop over all coordinates
        # this mask zeros out the gradient update at places where it is zero
        set the mask = 0 at lock spots
        
# if given a loose_weights coordinates then this mask is 0's everywhere,
except 1's at those coordinates
if loose
    set all weights = 0
    loop over all coordinates
        # this mask allows gradient update at places where it is one
        set the mask = 1 at loose spots


'''



'''
    # in future we will want:
    # in layer=3, layer_width=2, for synoyms of die set weights =  STRONG
    # in layer=2, layer_width=2, for synoyms of of set weights =  STRONG
    # in layer=1, layer_width=2, for synoyms of rainy set weights =  STRONG
    # in layer=0, layer_width=2, for synoyms of days set weights =  STRONG
    # need encoder attention weight in layer 3 that is STRONG from layer_width 3 up to layer_width 2
    # need encoder attention weight in layer 3 that is STRONG from layer_width 2 down to layer_width 3
'''