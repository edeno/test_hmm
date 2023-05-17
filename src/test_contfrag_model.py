import jax
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from patsy import dmatrix
from replay_trajectory_classification import RandomWalk, Uniform
from tqdm.autonotebook import tqdm
from replay_trajectory_classification.core import atleast_2d
from replay_trajectory_classification.environments import get_n_bins
from patsy import build_design_matrices

from src.estimate_transition import estimate_stationary_state_transition
from src.hmm import (
    check_converged,
    fit_poisson_regression,
    forward,
    smoother,
    viterbi,
)

jax.config.update("jax_platform_name", "cpu")

EPS = 1e-15


def make_spline_design_matrix(position, place_bin_edges, knot_spacing=10):

    position = atleast_2d(position)
    inner_knots = []
    for pos, edges in zip(position.T, place_bin_edges.T):
        n_points = get_n_bins(edges, bin_size=knot_spacing)
        knots = np.linspace(edges.min(), edges.max(), n_points)[1:-1]
        knots = knots[(knots > pos.min()) & (knots < pos.max())]
        inner_knots.append(knots)

    inner_knots = np.meshgrid(*inner_knots)

    data = {}
    formula = "1 + te("
    for ind in range(position.shape[1]):
        formula += f"cr(x{ind}, knots=inner_knots[{ind}])"
        formula += ", "
        data[f"x{ind}"] = position[:, ind]

    formula += 'constraints="center")'
    return dmatrix(formula, data)


def make_spline_predict_matrix(design_info, position):
    position = atleast_2d(position)
    is_nan = np.any(np.isnan(position), axis=1)
    position[is_nan] = 0.0

    predict_data = {}
    for ind in range(position.shape[1]):
        predict_data[f"x{ind}"] = position[:, ind]

    design_matrix = build_design_matrices([design_info], predict_data)[0]
    design_matrix[is_nan] = np.nan

    return design_matrix


def setup_contfrag_model(
    is_ripple,
    position,
    env,
    emission_knot_spacing=8,
):

    random_walk = RandomWalk().make_state_transition([env])
    uniform = Uniform().make_state_transition([env])

    state_names = ["continuous", "fragmented"]

    n_states = len(state_names)
    n_env_bins = env.place_bin_centers_.shape[0]
    bin_sizes = [n_env_bins, n_env_bins]

    state_ind = np.concatenate(
        [
            ind * np.ones((bin_size,), dtype=int)
            for ind, bin_size in enumerate(bin_sizes)
        ]
    )
    n_state_bins = len(state_ind)

    initial_conditions = np.ones((n_state_bins,)) / n_state_bins

    is_training = ~is_ripple

    discrete_state_transitions = np.asarray([[0.98, 0.02], [0.02, 0.98]])

    continuous_state_transitions = np.zeros((n_state_bins, n_state_bins))

    # need to zero out transitions to invalid position bins?
    for from_state in range(n_states):
        for to_state in range(n_states):
            inds = np.ix_(state_ind == from_state, state_ind == to_state)

            if np.logical_and(from_state == 0, to_state == 0):
                continuous_state_transitions[inds] = random_walk
            else:
                continuous_state_transitions[inds] = uniform

    data = {"x": position}
    emission_design_matrix = make_spline_design_matrix(
        position, env.place_bin_edges_, knot_spacing=emission_knot_spacing
    )
    emission_predict_matrix = make_spline_predict_matrix(
        emission_design_matrix.design_info, env.place_bin_centers_
    )

    plt.figure(figsize=(15, 15))
    plt.pcolormesh(
        np.arange(state_ind.shape[0] + 1),
        np.arange(state_ind.shape[0] + 1),
        np.log(continuous_state_transitions + np.spacing(1)),
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(np.log(discrete_state_transitions + np.spacing(1)))

    return (
        emission_design_matrix,
        emission_predict_matrix,
        initial_conditions,
        discrete_state_transitions,
        continuous_state_transitions,
        state_ind,
        is_training,
        state_names,
    )


def fit_contfrag_model(
    spikes,
    design_matrix,
    predict_matrix,
    initial_conditions,
    discrete_state_transitions,
    continuous_state_transitions,
    state_ind,
    is_training,
    tolerance=1e-4,
    max_iter=20,
    fit_inital_conditions=False,
    fit_discrete_transition=False,
    debug=False,
    concentration=1.0,
    stickiness=0.0,
):

    n_time = spikes.shape[0]
    n_states = discrete_state_transitions.shape[0]
    n_state_bins = continuous_state_transitions.shape[0]

    causal_state_probabilities = np.zeros((n_time, n_states))
    acausal_state_probabilities = np.zeros((n_time, n_states))
    predictive_state_probabilities = np.zeros((n_time, n_states))

    coefficients_iter = []
    non_local_rates_iter = []
    is_training_iter = []
    acausal_posterior_iter = []

    marginal_log_likelihoods = []
    n_iter = 0
    converged = False

    log_likelihood = np.zeros((n_time, n_state_bins))

    while not converged and (n_iter < max_iter):

        # Likelihoods
        print("Likelihoods")
        if n_iter == 0:
            coefficients = np.stack(
                [
                    fit_poisson_regression(
                        design_matrix,
                        is_training.astype(float),
                        s,
                    )
                    for s in tqdm(spikes.T)
                ],
                axis=1,
            )

            non_local_rates = np.exp(predict_matrix @ coefficients)
            non_local_rates = np.clip(non_local_rates, a_min=EPS, a_max=None)
            for s, r in zip(spikes.T, non_local_rates.T):
                log_likelihood[:, state_ind == 0] += scipy.stats.poisson.logpmf(
                    s[:, np.newaxis], r[np.newaxis]
                )
            log_likelihood[:, state_ind == 1] = log_likelihood[:, state_ind == 0]

        coefficients_iter.append(coefficients)
        non_local_rates_iter.append(non_local_rates)
        is_training_iter.append(is_training.astype(float))

        discrete_state_transitions_per_bin = discrete_state_transitions[
            np.ix_(state_ind, state_ind)
        ]

        transition_matrix = (
            discrete_state_transitions_per_bin * continuous_state_transitions
        )

        # Expectation step
        print("Expectation Step")
        causal_posterior, predictive_distribution, marginal_log_likelihood = forward(
            initial_conditions, log_likelihood, transition_matrix
        )
        acausal_posterior = smoother(
            causal_posterior, predictive_distribution, transition_matrix
        )

        acausal_posterior_iter.append(acausal_posterior)

        # Maximization step
        print("Maximization Step")
        for ind in range(n_states):
            is_state = state_ind == ind
            causal_state_probabilities[:, ind] = causal_posterior[:, is_state].sum(
                axis=1
            )
            acausal_state_probabilities[:, ind] = acausal_posterior[:, is_state].sum(
                axis=1
            )
            predictive_state_probabilities[:, ind] = predictive_distribution[
                :, is_state
            ].sum(axis=1)

        if fit_discrete_transition:
            discrete_state_transitions = estimate_stationary_state_transition(
                causal_state_probabilities,
                predictive_state_probabilities,
                discrete_state_transitions,
                acausal_state_probabilities,
                concentration=concentration,
                stickiness=stickiness,
            )

        if fit_inital_conditions:
            initial_conditions = acausal_posterior[0]

        # Stats
        print("Stats")
        n_iter += 1

        marginal_log_likelihoods.append(marginal_log_likelihood)
        if n_iter > 1:
            log_likelihood_change = (
                marginal_log_likelihoods[-1] - marginal_log_likelihoods[-2]
            )
            converged, _ = check_converged(
                marginal_log_likelihoods[-1], marginal_log_likelihoods[-2], tolerance
            )

            print(
                f"iteration {n_iter}, "
                f"likelihood: {marginal_log_likelihoods[-1]}, "
                f"change: {log_likelihood_change}"
            )
        else:
            print(f"iteration {n_iter}, " f"likelihood: {marginal_log_likelihoods[-1]}")

    predicted_state = viterbi(
        initial_conditions, np.exp(log_likelihood), transition_matrix
    )[0]

    debug = (
        (
            coefficients_iter,
            non_local_rates_iter,
            is_training_iter,
            acausal_posterior_iter,
        )
        if debug
        else ()
    )
    return (
        predicted_state,
        acausal_posterior,
        acausal_state_probabilities,
        causal_posterior,
        marginal_log_likelihoods,
        initial_conditions,
        discrete_state_transitions,
        non_local_rates,
        log_likelihood,
        *debug,
    )
