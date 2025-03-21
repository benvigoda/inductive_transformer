from dataclasses import dataclass
import jax.numpy as jnp  # type: ignore

from inductive_transformer.jax_transformer.helper_functions import custom_normalize


@dataclass
class EncoderAnd:
    def __call__(self, x, y):
        # OLD AND:
        # z_1 = x[1] * y[1]
        # z_0 = x[0] * y[1] + x[1] * y[0] + x[0] * y[0]
        # NEW EQUAL
        z_1 = x[1] + y[1]
        z_0 = x[0] + y[0]
        z = jnp.stack([z_0, z_1])

        z = custom_normalize(z, axis=0)
        return z

    '''
        Change softequals to + instead of *:

        z_1 = x[1] + y[1]
        z_0 = x[0] + y[0]
        z = jnp.stack([z_0, z_1])

        z = custom_normalize(z, axis=0)

        # if this does hit NaNs, then the equals cannot be the issue, because with a plus, 
        # there is no extreme where the derivative goes to infinity.  

        # verify that this sum and then stack and then normalize does have non-infinite gradients everywhere!
        # move on to log probability domain and the numerical underflow hypothesis

        # if this does NOT NaN out, it doesn't totally prove that the equals gate is the issue
        # but it would encourage us regardless to try to make an equals gate in isolation and run back-prop on it
        # see if this isolated equals shows NaNs and if it does put some epsilons or something to prevent the NaNs
        # at the corners of the function
        # once that works move the safe equals back into the overall system
        let's do this in pytorch


        # alternatively we could start moving to the log prob domain.  the equals gate would actually become a sum 
        # just like what we just tested, but it would be the correct function in that context!

        # an option would be take a log of the probabilities, go into the softequals and do it in the log prob domain
        # and then at the end of the softequals, switch back to probabilities
        # but ths doesn't help because there will still be values that have an infinite derivative with respect to others
        # and that will cause NaNs in back-prop
        # we would need to move everything into the log prob domain
        # the easy situation is
        y =  w * x where w is a vector of weights and x and y are vectors of activations

        in the prob domain this is element-wise multiplication *
        in the log prob domain this is element-wise addition +
        all the token pi's are therefore easy!

        then it comes down to universe and categorical-bernoulli gates where we might have a sum or products of prob
        but we just checked, and they are all super eadsy to translate to log prob domain

        so what's better?  translate everything or try to make robust equals gates in the prob domain?

        probably the same amount of effort either way, but going to the log prob domain is the good fight.
        
        '''