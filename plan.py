'''
# Project Plan:
1. TEST: python main.py training_text.txt inference_text.txt --train --num_layer 2 --layer_width 4 --num_data_points 10000 --silence_google_sheet
2. Modify the decoder to also have position_pi
3. Fix text_parsing output to also include position in the decoder, once that's done
        ###########
        # FIXME: The output doesn't have position yet, so we just drop the position dimension
        output_tensor = input_tensor[:, 0, :, :]
        ###########
4. Fix the print to google-sheet to include the new position code

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

3. 

DONE
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
