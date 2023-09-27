'''
ideating:

1. trace through activations when the weights are set the wrong way
2. write the print statements in advance




writing:

X send to Jeff & Steve for feedback
X move pencil edits into appendices
0. fix the decoder AND in latex
1. Jeff's edits
2. annotated transformer comparison
3. go through all FIXME's
4. citations
5. for extra credit: see if we can figure out the relationship between embeddings vectors and attention position



Sharing:
- send to Bill
- Matt Barr
- Alex Wissner Gross
- Martin McCormick?
- Jonathan Yedidia
- David Blei student
- Jake Neely
- Theo?
- go over every citation

Ben & Thomas:
- DONE: DecoderBernoulliCategorical(nn.Module): forward() might have an error
        Turns out it was the attention target that we don't train on anymore.

Thomas:
DONE: fill in the diagram with all the weights with actual numbers
        - What do we do about the standard-deviations?
        - Write a quick script that does 100 runs and averages the weights?
        https://chat.openai.com/share/7378c48e-10d1-4d22-acae-7252f4840204
        Just put the training loop into its own loop. Store the weights at the end of the loops, average and compute std dev.

- proofread and spotcheck every step and equation
- review how Ben discusses marginalizing out position in the main body
- go over every citation too
- send to Helene Rochais
- send to Jonathan Heckman


DONE: results tables:
DONE: 1. fill in some representative numbers in the black weights criss cross diagram which means we need final simulation results
DONE: 2. do 100 training runs, fill in the weight means and variance averaged, this will only work on relu'd normalized weights


DONE: Ben:
DONE: - make slides for Tuesday if we are going to MIT


'''
