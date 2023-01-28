import numpy as np


def _normalize(u, axis=0, eps=1e-15):
    """Normalizes the values within the axis in a way that they sum up to 1.
    Args:
        u: Input array to normalize.
        axis: Axis over which to normalize.
        eps: Minimum value threshold for numerical stability.
    Returns:
        Tuple of the normalized values, and the normalizing denominator.
    """
    u = np.where(u == 0, 0, np.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = np.where(c == 0, 1, c)
    return u / c, c


# Helper functions for the two key filtering steps
def _condition_on(probs, ll):
    """Condition on new emissions, given in the form of log likelihoods
    for each discrete state, while avoiding numerical underflow.
    Args:
        probs(k): prior for state k
        ll(k): log likelihood for state k
    Returns:
        probs(k): posterior for state k
    """
    ll_max = ll.max()
    new_probs = probs * np.exp(ll - ll_max)
    new_probs, norm = _normalize(new_probs)
    log_norm = np.log(norm) + ll_max
    return new_probs, log_norm


def _predict(probs, A):
    return A.T @ probs


def hmm_filter(
    initial_distribution,
    transition_matrix,
    log_likelihoods,
):
    num_timesteps = log_likelihoods.shape[0]

    marginal_likelihood = 0.0

    filtered_probs = np.zeros_like(log_likelihoods)
    predicted_probs = np.zeros_like(log_likelihoods)
    predicted_probs[0] = initial_distribution

    for t in range(num_timesteps):
        filtered_probs[t], log_norm = _condition_on(
            predicted_probs[t - 1], log_likelihoods[t]
        )
        marginal_likelihood += log_norm
        predicted_probs[t] = _predict(filtered_probs[t], transition_matrix)

    return marginal_likelihood, filtered_probs, predicted_probs


def hmm_smoother(
    filtered_probs,
    predicted_probs,
    transition_matrix,
):
    num_timesteps = filtered_probs.shape[0]

    smoothed_probs = np.zeros_like(filtered_probs)
    smoothed_probs[-1] = filtered_probs[-1]

    for t in range(num_timesteps - 2, -1, -1):
        relative_probs_next = np.where(
            np.isclose(predicted_probs[t + 1], 0.0),
            0.0,
            smoothed_probs[t + 1] / predicted_probs[t + 1],
        )
        smoothed_probs[t] = filtered_probs[t] * (
            transition_matrix @ relative_probs_next
        )
        smoothed_probs[t] /= smoothed_probs[t].sum()

    return filtered_probs, predicted_probs, smoothed_probs
