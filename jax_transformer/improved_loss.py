import jax.numpy as jnp
from jax.nn import logsumexp, log_softmax


def jensen_shannon_with_branch_entropy(truths, t_out, t_per_branch, entropy_weight=0.1):
    """
    JS loss with entropy regularization to encourage branch specialization.
    Low entropy = one branch dominates (good)
    High entropy = both branches contribute equally (bad)
    """
    # Standard JS loss
    P = jnp.exp(truths)
    Q = jnp.exp(t_out)
    M = 0.5 * (P + Q)

    kl_p_m = jnp.sum(P * jnp.log(P / (M + 1e-8)), axis=-1)
    kl_q_m = jnp.sum(Q * jnp.log(Q / (M + 1e-8)), axis=-1)
    js_loss = 0.5 * (kl_p_m + kl_q_m).mean()

    # Branch entropy regularization
    if t_per_branch is not None and t_per_branch.shape[-1] == 2:
        # Get branch contributions (after layer aggregation)
        # Shape: (batch, positions, vocab, layer_width=2)

        # Compute how much each branch contributes to each position
        branch_contributions = logsumexp(t_per_branch, axis=2)  # Sum over vocab
        # Shape: (batch, positions, layer_width=2)

        # Convert to probabilities and compute entropy
        branch_probs = jnp.exp(log_softmax(branch_contributions, axis=-1))
        entropy = -jnp.sum(branch_probs * jnp.log(branch_probs + 1e-10), axis=-1)

        # We want LOW entropy (one branch active), so we ADD entropy as penalty
        entropy_penalty = entropy_weight * entropy.mean()
        print(f"entropy_penalty: {entropy_penalty}")

        total_loss = js_loss + entropy_penalty
    else:
        total_loss = js_loss

    return total_loss


def jensen_shannon_with_orthogonality(truths, t_out, t_per_branch, ortho_weight=0.1):
    """
    Encourage branches to have orthogonal (uncorrelated) outputs.
    """
    # Standard JS loss
    P = jnp.exp(truths)
    Q = jnp.exp(t_out)
    M = 0.5 * (P + Q)

    kl_p_m = jnp.sum(P * jnp.log(P / (M + 1e-8)), axis=-1)
    kl_q_m = jnp.sum(Q * jnp.log(Q / (M + 1e-8)), axis=-1)
    js_loss = 0.5 * (kl_p_m + kl_q_m).mean()

    # Orthogonality constraint
    if t_per_branch is not None and t_per_branch.shape[-1] == 2:
        left = t_per_branch[..., 0].reshape(-1)  # Flatten
        right = t_per_branch[..., 1].reshape(-1)

        # Normalize
        left = left - left.mean()
        right = right - right.mean()

        # Compute correlation (should be close to 0)
        correlation = jnp.dot(left, right) / (jnp.linalg.norm(left) * jnp.linalg.norm(right) + 1e-10)

        # Penalize high correlation
        ortho_penalty = ortho_weight * jnp.abs(correlation)

        total_loss = js_loss + ortho_penalty
    else:
        total_loss = js_loss

    return total_loss