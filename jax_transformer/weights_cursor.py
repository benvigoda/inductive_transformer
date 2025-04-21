import jax
import jax.numpy as jnp
import numpy as np
from jax_transformer.helper_functions import EPSILON, get_num_layers
from inductive_transformer.datasets.anavan import make_cat_dog_anavan, make_cat_dog_worm_bird_anavan

strong = jnp.log(1.0 - EPSILON)  # Amplify the signal
weak = jnp.log(EPSILON)  # Dampen the signal
mask_type = jnp.float32  # Use float32 for consistency with parameters


'''

⌘K to generate a command
Search for any other places in the codebase where we might need to update dtype handling?

I'll search comprehensively for all dtype-related code and potential type conversion issues.
Let me analyze the dtype usage across the codebase:
Model Weight Initialization - All model components consistently use jnp.float32:
Apply to weights_curs...
)
This appears in:
decoder_layer.py
encoder_layer.py
decoder_token_pi.py
encoder_token_pi.py
decoder_position_pi.py
encoder_position_pi.py
decoder_attention_pi.py
encoder_attention_pi.py
Printing/Formatting - There are some type conversions in printing.py, but they're commented out:
Apply to weights_curs...
"
These are just for display formatting and don't affect computation.
Random Number Generation - In train.py, there's integer range specification:
Apply to weights_curs...
max
This is correct as it's used for random seed generation.
Command Line Arguments - In train.py:
Apply to weights_curs...
)
These are correctly typed as Python floats.
Model Parameters - In model.py:
Apply to weights_curs...
)
This consistently uses float32.
After reviewing all dtype usage, here are the recommended updates:
In weights_broad_init.py, ensure all mask-related operations use float32:
Apply to weights_curs...
)
In train.py, when handling the random seed, you might want to add a type annotation for clarity:
Apply to weights_curs...
]
The commented-out code in printing.py can be left as is since it's not active, but if you decide to uncomment it, consider using float formatting instead of integer conversion:
Apply to weights_curs...
"
Overall, the codebase is quite consistent with its dtype usage, primarily using float32 for model parameters and computations. The main fix we already made (changing mask_type to jnp.float32) should resolve the warning. Would you like me to:
Show how to add type annotations to make the dtype expectations more explicit?
Add runtime checks to verify dtype consistency?
Look at any specific component in more detail?
what does this code do:



Let me explain what this code does:
Apply to weights_curs...
)
This code creates a mask structure that matches the shape of the model parameters. Let's break it down:
jax.tree_util.tree_map - This is a JAX utility that applies a function to every leaf node in a nested structure (like a Python dictionary or nested arrays). It's commonly used with PyTrees, which is JAX's way of handling nested parameter structures in neural networks.
lambda x: jnp.ones_like(x, dtype=mask_type) - This is the function being applied to each leaf node:
jnp.ones_like(x) creates an array of ones with the same shape as x
dtype=mask_type specifies the data type (which we changed to jnp.float32)
params - This is the model's parameter structure, typically containing weights and biases organized in a nested dictionary
For example, if your params structure looks like:
Apply to weights_curs...
}
The resulting set_weights would have the same structure but with all values set to 1.0:
Apply to weights_curs...
}
This mask is typically used to control which parameters can be updated during training:
A value of 1.0 means the corresponding parameter can be updated
A value of 0.0 means the corresponding parameter should be frozen (not updated)
The code appears in weights_broad_init.py and is part of the weight initialization and control system, where it's used to create a mask that determines which weights can be modified during training. This is particularly useful for:
Selective training of certain layers
Parameter freezing
Implementing specific weight update patterns
Would you like me to:
Show how this mask is used later in the code?
Explain more about JAX's tree manipulation utilities?
Show examples of how this affects the training process?
use example code from weights_broad_init.py to fill in the actual python code in weights.py.  follow our comments in weights.py

I'll search for the relevant code in both files to help implement the comments in weights.py.
Based on both files, I'll help implement the functions in weights.py using the example code from weights_broad_init.py. Here's the implementation:
Apply
This implementation:
Maintains the same structure as weights_broad_init.py but reorganizes it into more modular functions
Implements the synonym structure as requested in the comments
Uses the same weight setting patterns for attention, position, and token weights
Adds proper type handling using jnp.float32
Includes perturbation functionality with optional targeting of specific indices
Key differences from weights_broad_init.py:
More organized class structure for synonyms
Cleaner separation of concerns between different weight setting functions
More flexible perturbation system
Consistent use of tree_map for weight operations
Would you like me to:
Add more detailed comments explaining specific parts?
Add type hints to make the code more maintainable?
Add validation checks for the parameters?
Show example usage of these functions?
'''

class WeightTreeMapKeys:
    def __init__(self, encoder_decoder, layer, factor, position_idx, token_idx, noise_value):
        self.encoder_decoder = encoder_decoder
        self.layer = layer
        self.factor = factor
        self.position_idx = position_idx
        self.token_idx = token_idx
        self.noise_value = noise_value

class SynonymList:
    def __init__(self, name, layer, layer_width_idx, token_list):
        self.name = name
        self.layer = layer
        self.layer_width_idx = layer_width_idx
        self.token_list = token_list

class Synonyms:
    def __init__(self, vocab, catsanddogs=False):
        if catsanddogs:
            anavan = make_cat_dog_anavan()
        else:
            anavan = make_cat_dog_worm_bird_anavan()
            
        self.synonym_lists = [
            # Left side (layer_width_idx = 0)
            SynonymList("small", 5, 0, anavan.get_synonyms("small")),
            SynonymList("dog", 4, 0, anavan.get_synonyms("dog")),
            SynonymList("often", 3, 0, anavan.get_synonyms("often")),
            SynonymList("fear", 2, 0, anavan.get_synonyms("fear")),
            SynonymList("large", 1, 0, anavan.get_synonyms("large")),
            SynonymList("cats", 0, 0, anavan.get_synonyms("cats")),
            
            # Right side (layer_width_idx = 1)
            SynonymList("wriggly", 5, 1, anavan.get_synonyms("wriggly")),
            SynonymList("worms", 4, 1, anavan.get_synonyms("worms")),
            SynonymList("sometimes", 3, 1, anavan.get_synonyms("sometimes")),
            SynonymList("chase", 2, 1, anavan.get_synonyms("chase")),
            SynonymList("angry", 1, 1, anavan.get_synonyms("angry")),
            SynonymList("birds", 0, 1, anavan.get_synonyms("birds"))
        ]

def set_all_random(params):
    return jax.tree_util.tree_map(
        lambda x: jax.random.normal(
            jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)),
            x.shape
        ),
        params
    )

def set_all_weak(params):
    return jax.tree_util.tree_map(
        lambda x: jnp.full_like(x, weak),
        params
    )

def set_attention_weights(params, num_layers):
    """Set attention weights to 1 for straight-up connections and 0 for cross connections"""
    for layer in range(num_layers):
        for prefix in ["encoder", "decoder"]:
            layer_key = f"{prefix}s_{layer}"
            attention_pi = f"{prefix}_attention_pi"
            
            if layer_key in params["params"]:
                weights = params["params"][layer_key][attention_pi]["weights"]
                # Set all to weak first
                new_weights = jnp.full_like(weights, weak)
                # Set straight-up connections to strong
                new_weights = new_weights.at[:, :, 0, 0].set(strong)  # left to left
                new_weights = new_weights.at[:, :, 1, 1].set(strong)  # right to right
                params["params"][layer_key][attention_pi]["weights"] = new_weights
    
    return params

def set_position_weights(params, num_layers):
    """Set position weights according to the layer pattern"""
    for layer in range(num_layers):
        for prefix in ["encoder", "decoder"]:
            layer_key = f"{prefix}s_{layer}"
            position_pi = f"{prefix}_position_pi"
            
            if layer_key in params["params"]:
                weights = params["params"][layer_key][position_pi]["weights"]
                new_weights = jnp.full_like(weights, weak)
                # Set the appropriate position to strong
                new_weights = new_weights.at[-layer - 1].set(
                    jnp.full(weights.shape[-1], strong)
                )
                params["params"][layer_key][position_pi]["weights"] = new_weights
    
    return params

def set_token_weights(params, synonyms, vocab):
    """Set token weights based on synonym lists"""
    for synonym_list in synonyms.synonym_lists:
        for prefix in ["encoder", "decoder"]:
            layer_key = f"{prefix}s_{synonym_list.layer}"
            token_pi = f"{prefix}_token_pi"
            
            if layer_key in params["params"]:
                weights = params["params"][layer_key][token_pi]["weights"]
                # Set all weights at this position and layer_width to weak
                weights = weights.at[:, :, synonym_list.layer_width_idx].set(weak)
                
                # Set weights for synonym tokens to strong
                for token in synonym_list.token_list:
                    if token in vocab:
                        vocab_idx = next(i for i, word in enumerate(vocab) if word.lower() == token)
                        weights = weights.at[synonym_list.layer, vocab_idx, synonym_list.layer_width_idx].set(strong)
                
                params["params"][layer_key][token_pi]["weights"] = weights
    
    return params

def add_perturbations(params, noise_value, perturb_indices=None):
    """Add noise to specified indices or all weights if indices not specified"""
    def add_noise(x):
        if perturb_indices is None:
            return x + jax.random.normal(
                jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1)),
                x.shape
            ) * noise_value
        else:
            new_x = x.copy()
            for idx in perturb_indices:
                new_x = new_x.at[idx].add(noise_value)
            return new_x
    
    return jax.tree_util.tree_map(add_noise, params)

def set_weights(params, vocab, noise_value=0.01, perturb_indices=None, catsanddogs=False):
    """Main function to set all weights"""
    num_layers = get_num_layers(params)
    synonyms = Synonyms(vocab, catsanddogs=catsanddogs)
    
    # Initialize all weights to weak
    params = set_all_weak(params)
    
    # Set specific weight patterns
    params = set_attention_weights(params, num_layers)
    params = set_position_weights(params, num_layers)
    params = set_token_weights(params, synonyms, vocab)
    
    # Add perturbations if specified
    if noise_value > 0:
        params = add_perturbations(params, noise_value, perturb_indices)
    
    return params