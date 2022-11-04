import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm

from src.hmm import (
    backward,
    correction_smoothing,
    forward,
    get_likelihood,
    update_transition_matrix_from_correction_smoothing,
    update_transition_matrix_from_parallel_smoothing,
    estimate_transition_matrix_from_gradient_descent,
)


def generate_data():
    initial_conditions = np.asarray([0.2, 0.8])
    transition_matrix = np.asarray([[0.5, 0.5], [0.3, 0.7]])
    emission_matrix = np.asarray([[0.3, 0.7], [0.8, 0.2]])

    observations = np.asarray(["N", "N", "N", "N", "N", "E", "E", "N", "N", "N"])
    observations_ind = np.asarray([0 if o == "N" else 1 for o in observations])

    return (
        initial_conditions,
        transition_matrix,
        emission_matrix,
        observations,
        observations_ind,
    )


def generate_data3():
    initial_conditions = np.asarray([1.0, 0.0, 0.0])
    transition_matrix = np.asarray([[0.0, 0.5, 0.5], [0.0, 0.9, 0.1], [0, 0, 1]])
    emission_matrix = np.asarray([[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]])

    observations = np.asarray([2, 3, 3, 2, 2, 2, 3, 2, 3])
    observations_ind = np.asarray([0 if o == 2 else 1 for o in observations])

    return (
        initial_conditions,
        transition_matrix,
        emission_matrix,
        observations,
        observations_ind,
    )


def compare_transition_matrix(use_parallel=True):
    (
        initial_conditions,
        transition_matrix,
        emission_matrix,
        _,
        observations_ind,
    ) = generate_data()

    model = hmm.CategoricalHMM(
        n_components=len(initial_conditions),
        init_params="",
        params="t",
        implementation="scaling",
        n_iter=10,
    )
    model.startprob_ = initial_conditions
    model.transmat_ = transition_matrix
    model.emissionprob_ = emission_matrix

    model.fit(observations_ind[:, np.newaxis])

    likelihood = get_likelihood(emission_matrix, observations_ind)

    graident_transition_matrix, _ = estimate_transition_matrix_from_gradient_descent(
        initial_conditions, likelihood, transition_matrix
    )

    dll = []

    for _ in range(model.monitor_.iter):

        causal_posterior, data_log_likelihood, scaling = forward(
            initial_conditions, likelihood, transition_matrix
        )

        if use_parallel:
            scaled_backward_posterior = backward(
                initial_conditions,
                likelihood,
                transition_matrix,
                scaling,
            )
            transition_matrix = update_transition_matrix_from_parallel_smoothing(
                causal_posterior,
                scaled_backward_posterior,
                likelihood,
                transition_matrix,
                data_log_likelihood,
            )
        else:
            acausal_posterior = correction_smoothing(
                causal_posterior, transition_matrix
            )
            transition_matrix = update_transition_matrix_from_correction_smoothing(
                causal_posterior,
                acausal_posterior,
                likelihood,
                transition_matrix,
            )
        dll.append(data_log_likelihood)

    plt.plot(np.exp(dll))
    plt.plot(np.exp(model.monitor_.history))

    print("hmmlearn")
    print(model.transmat_)
    if use_parallel:
        print("parallel")
    else:
        print("correction")
    print(transition_matrix)
    print("gradient")
    print(graident_transition_matrix)
    print("diff")
    print(model.transmat_ - transition_matrix)


def compare_one():
    (
        initial_conditions,
        transition_matrix,
        emission_matrix,
        _,
        observations_ind,
    ) = generate_data()

    model = hmm.CategoricalHMM(
        n_components=len(initial_conditions),
        init_params="",
        params="t",
        implementation="scaling",
        n_iter=1,
    )
    model.startprob_ = initial_conditions
    model.transmat_ = transition_matrix
    model.emissionprob_ = emission_matrix

    model.fit(observations_ind[:, np.newaxis])

    likelihood = get_likelihood(emission_matrix, observations_ind)

    causal_posterior, data_log_likelihood, scaling = forward(
        initial_conditions, likelihood, transition_matrix
    )

    acausal_posterior = correction_smoothing(causal_posterior, transition_matrix)

    scaled_backward_posterior = backward(
        initial_conditions,
        likelihood,
        transition_matrix,
        scaling,
    )
    transition_matrix_parallel = update_transition_matrix_from_parallel_smoothing(
        causal_posterior,
        scaled_backward_posterior,
        likelihood,
        transition_matrix,
        data_log_likelihood,
    )

    transition_matrix_correction = update_transition_matrix_from_correction_smoothing(
        causal_posterior,
        acausal_posterior,
        likelihood,
        transition_matrix,
    )

    print("hmmlearn")
    print(model.transmat_)
    print("parallel")
    print(transition_matrix_parallel)
    print("correction")
    print(transition_matrix_correction)
    print("hmmlearn vs. parallel")
    print(model.transmat_ - transition_matrix_parallel)
    print("hmmlearn vs. correction")
    print(model.transmat_ - transition_matrix_correction)


def simulate_poisson_spikes(rate, sampling_frequency):
    """Given a rate, returns a time series of spikes.
    Parameters
    ----------
    rate : np.ndarray, shape (n_time,)
    sampling_frequency : float
    Returns
    -------
    spikes : np.ndarray, shape (n_time,)
    """
    return np.random.poisson(rate / sampling_frequency)


def simulate_two_state_poisson(n_time=20_000, sampling_frequency=1000):

    rate = 5.0 * np.ones((n_time,))
    rate[(n_time // 6) : (2 * n_time // 6)] = 20.0
    rate[(4 * n_time // 6) : (6 * n_time // 6)] = 20.0

    time = np.arange(n_time) / sampling_frequency
    spikes = simulate_poisson_spikes(rate, sampling_frequency)

    return time, rate, spikes
