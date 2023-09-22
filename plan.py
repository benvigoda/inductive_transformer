'''
weekend:

debug:
see plan below


final math writing - rewrite the main body math to:
1. verify clean and readable
2. remove some aspects appendix
3. add the google deepmind citation and speak to it in the prior art
4. good enough and send to Jeff & Steve & Bill for feedback

results tables:
1. fill in some representative numbers in the black weights criss cross diagram which means we need final simulation results
2. do 100 training runs, fill in the weight means and variance averaged, this will only work on relu'd normalized weights



DONE: compile
DONE: add attention output and include it in the loss function
DONE: output to google sheet
DONE: Review the output to google sheet
1. In layer_0, it did pick-up on the size, but it's very close to 50-50.
        This is expected.
   In layer_1 on the other hand, preds is matching truth very well.
        This is great!
2. The token weights were learned perfectly!

DONE:
The decoder attention weights seem misplaced.
Could it be that row 4 should be row 0?
It doesn't look like it's misplaced
TODO:
Why is the decoder attention weights seem off?

Possible next prints:
1. Print z' output from the encoder, i.e. input to the decoder
2. Print the weights normalized, turning negative values to 0 before normalizing




Hypotheses/ideas:

NOPE: maybe coming from the encoder we need to convert to z_categorical and then back to z_bernoulli
it's already tilted 100% - 0%.  the encoder is perfect.

maybe the fact that we ignore the backward messages to the decoder AND is a problem?
essentially we have no residual connections going to the decoder

(it would not be an issue for the pi's nor for the categorical-bernoulli-categoricals
it would not be an issue for the decoder_universe)

the pi_t's in decoder layer 1 are getting good inputs and outputting the right thing

in the decoder maybe we wrote the wrong indexing for the decoder categorical to bernoulli  v --> u
or maybe we wrote the wrong indexing in decoder_universe

way to test it:
write down truth table from v_categorical to z_bernoulli and test it with saturated probabilities

v[0][0]   v[0][1]     z[0]
0         0           0
0         1           1
1         0           1
1         1           1

same for v[1][0], v[1][1] and z[1]

set v at input to model.decoder_layer_1.decoder_categorical_bernoulli.v
observe model.decoder_layer_1.decoder_universe.z
run inference






'''
