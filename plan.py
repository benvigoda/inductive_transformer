'''
# Project Plan:

TODO:

DONE: weights code and printing look good

We need to make sure the input is correct in code and in printing

The input is shape num_layers=2, num_positions=2, vocab=6, layer_width=3

we want little dog, on the left column, little in position=0, layer_num=0 and dog in position 1, layer_num=1
let's make sure we are matching how we set the weights up
small in num_layer=0, position=0, lw=0  # small is word #3 in the vocab
dog in num_layer=1, position=1, lw=0  # dog is word #4

input[0] == input[1] for both num_layers
input[0][0][:][0] = [0, 0, 0, 1, 0, 0]

Then hopefully our output of inference will be good

TODO: Make the input match exactly the weights


TODO:
The code does not converge

Our overall strategy is to set everything, both weights and inputs and targets and
get "correct behavior"
Then we plan to slowly "loosen" up the weights and see if we learn back to the correct behavior
This led us to want to make sure the attention stack was doing "correct behavior"

In a new branch off main called "testing-understanding-attention-input"
Instead of our old inputs to the encoder attention in layer 0 being 0.5, 0.5,
we tested [1, 0] = [0, 1] = self.probable; [0, 1] = [1, 0] = self.improbable

The result in the attention weights was to learn to tilt the encoder layer 1 weights 80-20
instead of 50-50 as we got before.  The decoder attention weights seemed more or less unaffected.

QUESTION: Did the encoder layer 1 attention learned to "untilt" the activations flowing
out of the encoder to the decoder?

The decoder attention stack has no target that it is trying to hit.  This starts to beg the
question, why do we have the attention stack at all?  What role are we going to want it
to play?

We should think about that.  What is the correct behavior for the attention stack?

That said, we can also proceed without settling this issue.  The main goal is to see what
we can learn in the position pi's.

So next time we could focus on trying to learn one layer of position weights in the decoder.

Fix printing for inference







2. DONE: but the outputs don't look right (normalization is wrong) So needs further investigation
 Google sheet printing to verify weights we are learning in position_pi's
    encoder
    layer = 0, lw = 0: position = 0, token = little
    layer = 0, lw = 1: position = 0, token = big
    layer = 1, lw = 0: position = 1, token = dog
    layer = 1, lw = 1: position = 1, token = cat

    decoder
    layer = 1, lw = 0: position = 0, token = little
    layer = 1, lw = 1: position = 0, token = big
    layer = 0, lw = 0: position = 1, token = dog
    layer = 0, lw = 1: position = 1, token = cat

1. Training output data is wrong?:
Thomas to write what the training data looks like
Ben write down what the symmetries are desired

    shape = (NUM_LAYERS, NUM_POSITIONS, VOCAB_SIZE, LAYER_WIDTH)
    input and target, for all layers:

    decoder target output:
    set all values = epsilon, then:
    (layer = 1, position = 0, vocab[small] = 1, lw=0)
    (layer = 1, position = 0, vocab[big] = 1, lw=1)
    (layer = 0, position = 1, vocab[dog] = 1, lw=0)
    (layer = 0, position = 1, vocab[cat] = 1, lw=1)



    encoder input experiment #1:
    (layer = 1, position = 0, vocab[small] = 1, lw=0)
    (layer = 1, position = 0, vocab[big] = 1, lw=1)
    (layer = 0, position = 1, vocab[dog] = 1, lw=0)
    (layer = 0, position = 1, vocab[cat] = 1, lw=1)


    encoder input experiment #2:
    (layer = 1, position = 0, vocab[small] = 1, lw=0)
    (layer = 1, position = 0, vocab[small] = 1, lw=1)

    (layer = 1, position = 0, vocab[big] = 1, lw=0)
    (layer = 1, position = 0, vocab[big] = 1, lw=1)

    (layer = 0, position = 1, vocab[dog] = 1, lw=0)
    (layer = 0, position = 1, vocab[dog] = 1, lw=1)

    (layer = 0, position = 1, vocab[cat] = 1, lw=0)
    (layer = 0, position = 1, vocab[cat] = 1, lw=1)


    encoder input experiment #3:
    (layer = 1, position = 0, vocab[small] = 1, lw=0)
    (layer = 1, position = 0, vocab[small] = 1, lw=1)
    (layer = 0, position = 0, vocab[small] = 1, lw=0)
    (layer = 0, position = 0, vocab[small] = 1, lw=1)

    (layer = 1, position = 0, vocab[big] = 1, lw=0)
    (layer = 1, position = 0, vocab[big] = 1, lw=1)
    (layer = 0, position = 0, vocab[big] = 1, lw=0)
    (layer = 0, position = 0, vocab[big] = 1, lw=1)

    (layer = 0, position = 1, vocab[dog] = 1, lw=0)
    (layer = 0, position = 1, vocab[dog] = 1, lw=1)
    (layer = 1, position = 1, vocab[dog] = 1, lw=0)
    (layer = 1, position = 1, vocab[dog] = 1, lw=1)

    (layer = 0, position = 1, vocab[cat] = 1, lw=0)
    (layer = 0, position = 1, vocab[cat] = 1, lw=1)
    (layer = 1, position = 1, vocab[cat] = 1, lw=0)
    (layer = 1, position = 1, vocab[cat] = 1, lw=1)








    position = 0, token = little
    position = 0, token = big
    position = 1, token = dog
    position = 1, token = cat



3. Google sheet print all the encoder and decoder activations:
position_pi.rho, position_pi.x, token.pi.t
    a) Look at self.rho
    b) Fix the google-sheet to print with position in decoder

# TODO:
1. In encoder_bernoulli_categorical.py we have
class EncoderBernoulliCategorical(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderBernoulliCategorical, self).__init__()
        ...
        self.v = None

    def forward(self, u):
        v = torch.empty((self.hyperparams.layer_width, self.hyperparams.layer_width), device=u.device)
        v = u[1] / u[0]
It seems it would be more efficient to initialize v once in the init instead of every time in the forward? Can we do that?
This is not the only place where that happens. Let's fix it everywhere else? Can we first confirm that it's a useful thing to do?

Actually, we can probably just very simply delete that `v = torch.empty((self.hyperparams.layer_width, self.hyperparams.layer_width), device=u.device)` line, right?

2. Do we need the 1e-9 in the prob_weights in encoder_position_pi.py? Doesn't the custom_normalize already take care of that?
`prob_weights = self.relu(self.weights) + 1e-9`
This is not the only place this happens.

3. broadcast the weights in encoder_position_pi.py instead of stacking/catting them

DONE
0. Swap position indexing in weights we set in perturbation test:
encoder: [1][1][0][0]. decoder: [0][0][1][1]
1. TEST: python main.py training_text.txt inference_text.txt --train --num_layer 2 --layer_width 4 --num_data_points 10000 --silence_google_sheet
2. Modify the decoder to also have position_pi
3. Fix text_parsing output to also include position in the decoder, once that's done
        ###########
        # FIXME: The output doesn't have position yet, so we just drop the position dimension
        output_tensor = input_tensor[:, 0, :, :]
        ###########

1. Format data input to be fed into the new position model
1. Fix Ben's github
2. Get layer_width=3 working
Generalization to more than 2 layer_width:
1. encoder-and: Done
2. encoder-attention: Done (was already good)
3. encoder-token: Done (was already good)
4. encoder-bernoulli-categorical: Done
5. encoder-categorical-bernoulli: same as decoder-categorical bernoulli?
6. encoder-universe: Done

7. decoder-and: done (to review)
8. decoder-attention: Done (was already good)
9. decoder-token: Done (was already good)
10. decoder-bernoulli-categorical: same as encoder-bernoulli-categorical?
11. decoder-categorical-bernoulli: done (to review)
12. decoder-universe: Done


ideating:

1. trace through activations when the weights are set the wrong way
2. write the print statements in advance




writing:

1. annotated transformer comparison
'''
