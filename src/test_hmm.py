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


def compare_transition_matrix(use_parallel=True):
    (
        initial_conditions,
        transition_matrix,
        emission_matrix,
        _,
        observations_ind,
    ) = generate_data()

    model = hmm.CategoricalHMM(
        n_components=2, init_params="", params="t", implementation="scaling", n_iter=10
    )
    model.startprob_ = initial_conditions
    model.transmat_ = transition_matrix
    model.emissionprob_ = emission_matrix

    model.fit(observations_ind[:, np.newaxis])

    dll = []

    for _ in range(model.monitor_.iter):
        likelihood = get_likelihood(emission_matrix, observations_ind)

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

    print(model.transmat_)
    print(transition_matrix)
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
        n_components=2, init_params="", params="t", implementation="scaling", n_iter=1
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

    print(model.transmat_)
    print(transition_matrix_parallel)
    print(transition_matrix_correction)
    print(model.transmat_ - transition_matrix_parallel)
    print(model.transmat_ - transition_matrix_correction)