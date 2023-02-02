import numpy as np


def forward(initial_conditions, log_likelihood, transition_matrix):
    n_time = log_likelihood.shape[0]

    predictive_distribution = np.zeros_like(log_likelihood)
    causal_posterior = np.zeros_like(log_likelihood)
    max_log_likelihood = np.max(log_likelihood, axis=1, keepdims=True)
    likelihood = np.exp(log_likelihood - max_log_likelihood)
    likelihood = np.clip(likelihood, a_min=1e-15, a_max=1.0)

    predictive_distribution[0] = initial_conditions
    causal_posterior[0] = initial_conditions * likelihood[0]
    norm = np.nansum(causal_posterior[0])
    marginal_likelihood = np.log(norm)
    causal_posterior[0] /= norm

    for t in range(1, n_time):
        # Predict
        predictive_distribution[t] = transition_matrix.T @ causal_posterior[t - 1]
        # Update
        causal_posterior[t] = predictive_distribution[t] * likelihood[t]
        # Normalize
        norm = np.nansum(causal_posterior[t])
        marginal_likelihood += np.log(norm)
        causal_posterior[t] /= norm

    marginal_likelihood += np.sum(max_log_likelihood)

    return causal_posterior, predictive_distribution, marginal_likelihood


def smoother(causal_posterior, predictive_distribution, transition_matrix):
    n_time = causal_posterior.shape[0]

    acausal_posterior = np.zeros_like(causal_posterior)
    acausal_posterior[-1] = causal_posterior[-1]

    for t in range(n_time - 2, -1, -1):
        # Handle divide by zero
        relative_distribution = np.where(
            np.isclose(predictive_distribution[t + 1], 0.0),
            0.0,
            acausal_posterior[t + 1] / predictive_distribution[t + 1],
        )
        acausal_posterior[t] = causal_posterior[t] * (
            transition_matrix @ relative_distribution
        )
        acausal_posterior[t] /= acausal_posterior[t].sum()

    return acausal_posterior


def estimate_transition_matrix(
    causal_posterior,
    predictive_distribution,
    transition_matrix,
    acausal_posterior,
):
    relative_distribution = np.where(
        np.isclose(predictive_distribution[1:], 0.0),
        0.0,
        acausal_posterior[1:] / predictive_distribution[1:],
    )[:, np.newaxis]

    # p(x_t, x_{t+1} | O_{1:T})
    joint_distribution = (
        transition_matrix[np.newaxis]
        * causal_posterior[:-1, :, np.newaxis]
        * relative_distribution
    )

    new_transition_matrix = (
        joint_distribution.sum(axis=0)
        / acausal_posterior[:-1].sum(axis=0, keepdims=True).T
    )

    new_transition_matrix /= new_transition_matrix.sum(axis=1, keepdims=True)

    return new_transition_matrix


def estimate_parameters_via_em(initial_conditions, log_likelihood, transition_matrix):
    # Expectation
    causal_posterior, predictive_distribution, marginal_likelihood = forward(
        initial_conditions, log_likelihood, transition_matrix
    )
    acausal_posterior = smoother(
        causal_posterior, predictive_distribution, transition_matrix
    )

    # Maximization
    initial_conditions = acausal_posterior[0]
    transition_matrix = estimate_transition_matrix(
        causal_posterior,
        predictive_distribution,
        transition_matrix,
        acausal_posterior,
    )

    return initial_conditions, transition_matrix, acausal_posterior, marginal_likelihood