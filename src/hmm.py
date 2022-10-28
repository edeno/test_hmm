import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, softmax


def get_likelihood(emission_matrix, observations_ind):
    return emission_matrix[:, observations_ind].T


def forward(initial_conditions, likelihood, transition_matrix):

    n_states = len(initial_conditions)
    n_time = len(likelihood)

    causal_posterior = np.zeros((n_time, n_states))
    causal_posterior[0] = initial_conditions * likelihood[0]
    scaling = np.zeros((n_time,))
    scaling[0] = causal_posterior[0].sum()
    causal_posterior[0] /= scaling[0]

    for time_ind in range(1, n_time):
        causal_posterior[time_ind] = likelihood[time_ind] * (
            causal_posterior[time_ind - 1] @ transition_matrix
        )
        scaling[time_ind] = causal_posterior[time_ind].sum()
        causal_posterior[time_ind] /= scaling[time_ind]

    # log probability of observations given model
    data_log_likelihood = np.sum(np.log(scaling))

    return causal_posterior, data_log_likelihood, scaling


def correction_smoothing(causal_posterior: np.ndarray, transition_matrix: np.ndarray):
    n_time, n_states = causal_posterior.shape

    acausal_posterior = np.zeros((n_time, n_states))
    acausal_posterior[-1] = causal_posterior[-1].copy()

    for time_ind in range(n_time - 2, -1, -1):
        numerator = transition_matrix * causal_posterior[time_ind][:, np.newaxis]
        acausal_posterior[time_ind] = np.sum(
            numerator
            * acausal_posterior[time_ind + 1]
            / (numerator.sum(axis=0) + np.spacing(1)),
            axis=1,
        )
    return acausal_posterior


def backward(
    initial_conditions: np.ndarray,
    likelihood: np.ndarray,
    transition_matrix: np.ndarray,
    scaling: np.ndarray,
) -> np.ndarray:
    """Calculates p(O_{t+1:T} \mid I_t) scaled by p(O_t \mid O_{1:t-1})

    The scaling is from the causal forward algorithm, which keeps it numerically stable.
    Unlike the forward algorithm, the returned value is not not a probability due to the scaling.

    Parameters
    ----------
    initial_conditions : np.ndarray, shape (n_states,)
    observations_ind : np.ndarray, shape (n_time,)
    emission_matrix : np.ndarray, shape (n_states, n_states)
    transition_matrix : np.ndarray, shape (n_states, n_states)
    scaling : np.ndarray, shape (n_time,)

    Returns
    -------
    scaled_backward_posterior : np.ndarray

    """
    n_states = len(initial_conditions)
    n_time = len(likelihood)

    scaled_backward_posterior = np.zeros((n_time, n_states))
    scaled_backward_posterior[-1] = 1 / scaling[-1]

    for time_ind in range(n_time - 2, -1, -1):
        scaled_backward_posterior[time_ind] = transition_matrix @ (
            likelihood[time_ind + 1] * scaled_backward_posterior[time_ind + 1]
        )

        scaled_backward_posterior[time_ind] /= scaling[time_ind]

    return scaled_backward_posterior


def get_acausal_posterior_from_parallel_smoothing(
    causal_posterior, scaled_backward_posterior
):
    acausal_posterior = causal_posterior * scaled_backward_posterior
    acausal_posterior /= acausal_posterior.sum(axis=1, keepdims=True)

    return acausal_posterior


def update_transition_matrix_from_parallel_smoothing(
    causal_posterior,
    scaled_backward_posterior,
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
                * scaled_backward_posterior[1:, to_state]
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
    n_time, n_states = causal_posterior.shape
    new_transition_matrix = np.empty((n_states, n_states))

    xi_correction = np.zeros((n_time, n_states, n_states))

    for time_ind in range(n_time - 1):
        for from_state in range(n_states):
            for to_state in range(n_states):
                xi_correction[time_ind, from_state, to_state] = (
                    causal_posterior[time_ind, from_state]
                    * likelihood[time_ind + 1, to_state]
                    * acausal_posterior[time_ind + 1, to_state]
                    * transition_matrix[from_state, to_state]
                    / (causal_posterior[time_ind + 1, to_state] + np.spacing(1))
                )
        xi_correction[time_ind] /= xi_correction[time_ind].sum()

    summed_xi_correction = xi_correction.sum(axis=0)

    new_transition_matrix = summed_xi_correction / summed_xi_correction.sum(
        axis=1, keepdims=True
    )

    return new_transition_matrix


def reconstruct_transition(off_diagonal_elements, n_states):
    """Takes logit transformed off-digaonal transition elements
    and recreates the transition matrix"""

    new_transition_matrix = np.zeros((n_states, n_states))
    is_off_diagonal = ~np.identity(n_states, dtype=bool)
    new_transition_matrix[is_off_diagonal] = off_diagonal_elements

    return softmax(new_transition_matrix, axis=1)


def negative_log_likelihood(params, initial_conditions, likelihood):
    n_states = len(initial_conditions)
    transition_matrix = reconstruct_transition(params, n_states)

    _, data_log_likelihood, _ = forward(
        initial_conditions, likelihood, transition_matrix
    )

    return -data_log_likelihood


def estimate_transition_matrix_from_gradient_descent(
    initial_conditions, likelihood, transition_matrix
):
    n_states = transition_matrix.shape[0]
    is_off_diagonal = ~np.identity(n_states, dtype=bool)

    x0 = logit(transition_matrix[is_off_diagonal])
    result = minimize(
        negative_log_likelihood,
        x0=x0,
        args=(initial_conditions, likelihood),
        method="BFGS",
    )

    transition_matrix = reconstruct_transition(result.x, n_states)

    return transition_matrix, result.success
