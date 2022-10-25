import numpy as np


def forward(initial_conditions, observations_ind, emission_matrix, transition_matrix):

    n_states = len(initial_conditions)
    n_time = len(observations_ind)

    likelihood = emission_matrix[:, observations_ind].T

    causal_posterior = np.zeros((n_time, n_states))
    causal_posterior[0] = initial_conditions * likelihood[0]
    scaling = np.zeros((n_time,))
    scaling[0] = causal_posterior[0].sum()
    causal_posterior[0] /= scaling[0]

    for time_ind in range(1, n_time):
        causal_posterior[time_ind] = likelihood[time_ind] * (
            transition_matrix.T @ causal_posterior[time_ind - 1]
        )
        scaling[time_ind] = causal_posterior[time_ind].sum()
        causal_posterior[time_ind] /= scaling[time_ind]

    # log probability of observations given model
    data_log_likelihood = np.sum(np.log(scaling))

    return causal_posterior, data_log_likelihood, scaling


def correction_smoothing(causal_posterior, transition_matrix):
    n_time, n_states = causal_posterior.shape

    acausal_posterior = np.zeros((n_time, n_states))
    acausal_posterior[-1] = causal_posterior[-1].copy()

    for time_ind in range(n_time - 2, -1, -1):
        acausal_posterior[time_ind] = causal_posterior[time_ind] * np.sum(
            transition_matrix
            * acausal_posterior[time_ind + 1]
            / (transition_matrix.T @ causal_posterior[time_ind] + np.spacing(1)),
            axis=1,
        )
    return acausal_posterior


def parallel_smoothing(
    initial_conditions,
    observations_ind,
    emission_matrix,
    transition_matrix,
    scaling,
):
    n_states = len(initial_conditions)
    n_time = len(observations_ind)
    backward_posterior = np.zeros((n_time, n_states))
    backward_posterior[-1] = 1 / scaling[-1]

    likelihood = emission_matrix[:, observations_ind].T

    for time_ind in range(n_time - 2, -1, -1):
        backward_posterior[time_ind] = transition_matrix @ (
            likelihood[time_ind + 1] * backward_posterior[time_ind + 1]
        )

        backward_posterior[time_ind] /= scaling[time_ind]

    return backward_posterior


def get_acausal_posterior_from_parallel_smoothing(causal_posterior, backward_posterior):
    acausal_posterior = causal_posterior * backward_posterior
    acausal_posterior /= acausal_posterior.sum(axis=1, keepdims=True)

    return acausal_posterior


def update_transition_matrix_from_parallel_smoothing(
    causal_posterior,
    backward_posterior,
    likelihood,
    transition_matrix,
    data_log_likelihood,
):

    n_states = causal_posterior.shape[1]
    new_transition_matrix = np.empty((n_states, n_states))

    for from_state in range(n_states):
        for to_state in range(n_states):
            new_transition_matrix[from_state, to_state] = np.sum(
                causal_posterior[:-1, from_state]
                * likelihood[1:, to_state]
                * backward_posterior[1:, to_state]
                * transition_matrix[from_state, to_state]
            )

    new_transition_matrix /= np.exp(data_log_likelihood)

    new_transition_matrix /= new_transition_matrix.sum(axis=1, keepdims=True)

    return new_transition_matrix


def update_transition_matrix_from_correction_smoothing(
    causal_posterior,
    acausal_posterior,
    likelihood,
    transition_matrix,
):
    n_states = causal_posterior.shape[1]
    new_transition_matrix = np.empty((n_states, n_states))

    for from_state in range(n_states):
        for to_state in range(n_states):
            new_transition_matrix[from_state, to_state] = np.sum(
                causal_posterior[:-1, from_state]
                * likelihood[1:, to_state]
                * acausal_posterior[1:, to_state]
                * transition_matrix[from_state, to_state]
                / (causal_posterior[1:, to_state] + np.spacing(1))
            )

    new_transition_matrix /= new_transition_matrix.sum(axis=1, keepdims=True)

    return new_transition_matrix
