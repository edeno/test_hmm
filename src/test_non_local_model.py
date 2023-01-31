import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from patsy import DesignInfo, DesignMatrix, build_design_matrices, dmatrix
from replay_trajectory_classification import RandomWalk, Uniform
from replay_trajectory_classification.classifier import joblib
from scipy.optimize import minimize
from tqdm.autonotebook import tqdm

from src.hmm import check_converged, viterbi
from src.hmm3 import estimate_transition_matrix, forward, smoother


jax.config.update("jax_platform_name", "cpu")


def load_data(work_computer=False):
    if work_computer:
        path = "/cumulus/edeno/non_local_paper/notebooks/"
    else:
        path = "/Users/edeno/Downloads/"

    position_info = pd.read_pickle(path + "Jaq_03_16_position_info.pkl")
    spikes = pd.read_pickle(path + "Jaq_03_16_spikes.pkl")
    is_ripple = pd.read_pickle(path + "Jaq_03_16_is_ripple.pkl")
    env = joblib.load(path + "Jaq_03_16_environment.pkl")

    time = np.asarray(position_info.index / np.timedelta64(1, "s"))
    spikes = np.asarray(spikes).astype(float)
    position = np.asarray(position_info.linear_position).astype(float)
    is_ripple = np.asarray(is_ripple).squeeze()

    return is_ripple, spikes, position, env, time


def fit_regression(design_matrix, weights, spikes):
    @jax.jit
    def neglogp(
        coefficients, spikes=spikes, design_matrix=design_matrix, weights=weights
    ):
        conditional_intensity = jnp.exp(design_matrix @ coefficients)
        conditional_intensity = jnp.clip(conditional_intensity, a_min=1e-15, a_max=None)
        log_likelihood = weights * jax.scipy.stats.poisson.logpmf(
            spikes, conditional_intensity
        )
        return -log_likelihood.sum()

    dlike = jax.grad(neglogp)
    dlike2 = jax.hessian(neglogp)

    initial_condition = np.array([np.log(np.average(spikes, weights=weights))])
    initial_condition = np.concatenate(
        [initial_condition, np.zeros(design_matrix.shape[1] - 1)]
    )

    res = minimize(
        neglogp, x0=initial_condition, method="Newton-CG", jac=dlike, hess=dlike2
    )

    return res.x


def fit_penalized_regression(design_matrix, weights, spikes, penalty=0.0):
    @jax.jit
    def neglogp(
        coefficients, spikes=spikes, design_matrix=design_matrix, weights=weights
    ):
        conditional_intensity = jnp.exp(design_matrix @ coefficients)
        conditional_intensity = jnp.clip(conditional_intensity, a_min=1e-15, a_max=None)
        log_likelihood = weights * jax.scipy.stats.poisson.logpmf(
            spikes, conditional_intensity
        )
        return -log_likelihood.mean() + penalty * (
            coefficients[1:] @ coefficients[1:].T
        )

    dlike = jax.grad(neglogp)
    dlike2 = jax.hessian(neglogp)

    initial_condition = np.array([np.log(np.average(spikes, weights=weights))])
    initial_condition = np.concatenate(
        [initial_condition, np.zeros(design_matrix.shape[1] - 1)]
    )

    res = minimize(
        neglogp, x0=initial_condition, method="Newton-CG", jac=dlike, hess=dlike2
    )

    return res.x


def make_spline_predict_matrix(
    design_info: DesignInfo, place_bin_centers: np.ndarray
) -> DesignMatrix:
    """Make a design matrix for position bins"""
    predict_data = {}
    predict_data[f"x"] = place_bin_centers

    return build_design_matrices([design_info], predict_data)[0]


def setup_nonlocal_switching_model(
    is_ripple,
    spikes,
    position,
    env,
    sampling_frequency=500,
    df=15,
    no_spike_to_no_spike_prob=0.9999,
    no_spike_rate=1e-5,
):

    random_walk = RandomWalk().make_state_transition([env])
    uniform = Uniform().make_state_transition([env])

    state_names = ["local", "no spike", "non-local"]

    n_states = len(state_names)
    n_env_bins = env.place_bin_centers_.shape[0]
    bin_sizes = [1, 1, n_env_bins]

    state_ind = np.concatenate(
        [
            ind * np.ones((bin_size,), dtype=int)
            for ind, bin_size in enumerate(bin_sizes)
        ]
    )
    n_state_bins = len(state_ind)

    initial_conditions = np.zeros((n_state_bins,))
    initial_conditions[state_ind == 0] = 1.0

    is_training = ~is_ripple

    local_to_local_prob = (
        np.logical_and(~is_ripple[:-1], ~is_ripple[1:]).sum() / (~is_ripple[:-1]).sum()
    )

    non_local_to_non_local_prob = (
        np.logical_and(is_ripple[:-1], is_ripple[1:]).sum() / is_ripple[:-1].sum()
    )

    discrete_state_transitions = np.asarray(
        [
            [
                local_to_local_prob,
                (1 - local_to_local_prob) / 2,
                (1 - local_to_local_prob) / 2,
            ],
            [
                (1 - no_spike_to_no_spike_prob) / 2,
                no_spike_to_no_spike_prob,
                (1 - no_spike_to_no_spike_prob) / 2,
            ],
            [
                (1 - non_local_to_non_local_prob) / 2,
                (1 - non_local_to_non_local_prob) / 2,
                non_local_to_non_local_prob,
            ],
        ]
    )

    continuous_state_transitions = np.zeros((n_state_bins, n_state_bins))

    # need to zero out transitions to invalid position bins?
    for from_state in range(n_states):
        for to_state in range(n_states):
            inds = np.ix_(state_ind == from_state, state_ind == to_state)

            if (bin_sizes[from_state] == 1) & (bin_sizes[to_state] == 1):
                # transition from discrete to discrete
                continuous_state_transitions[inds] = 1.0
            elif (bin_sizes[from_state] > 1) & (bin_sizes[to_state] == 1):
                # transition from continuous to discrete
                continuous_state_transitions[inds] = 1.0
            elif (bin_sizes[from_state] == 1) & (bin_sizes[to_state] > 1):
                # transition from discrete to continuous
                continuous_state_transitions[inds] = uniform[0]
            else:
                # transition from continuous to continuous
                if from_state != to_state:
                    continuous_state_transitions[inds] = uniform
                else:
                    continuous_state_transitions[inds] = random_walk

    data = {"x": position}
    design_matrix = dmatrix(f"bs(x, df={df})", data)
    zero_rates = np.ones_like(spikes) * no_spike_rate / sampling_frequency
    predict_matrix = make_spline_predict_matrix(
        design_matrix.design_info, env.place_bin_centers_
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(np.log(continuous_state_transitions + np.spacing(1)))

    plt.figure(figsize=(5, 5))
    plt.imshow(np.log(discrete_state_transitions + np.spacing(1)))

    return (
        design_matrix,
        predict_matrix,
        discrete_state_transitions,
        continuous_state_transitions,
        state_ind,
        zero_rates,
        is_training,
        state_names,
    )


def fit_switching_model(
    spikes,
    design_matrix,
    predict_matrix,
    discrete_state_transitions,
    continuous_state_transitions,
    state_ind,
    zero_rates,
    is_training,
    tolerance=1e-4,
    max_iter=3,
    fit_likelihood=True,
    fit_inital_conditions=False,
    fit_discrete_transition=False,
):

    n_time = spikes.shape[0]
    n_states = discrete_state_transitions.shape[0]
    n_state_bins = continuous_state_transitions.shape[0]

    causal_state_probabilities = np.zeros((n_time, n_states))
    acausal_state_probabilities = np.zeros((n_time, n_states))
    predictive_state_probabilities = np.zeros((n_time, n_states))

    coefficients_iter = []
    local_rates_iter = []
    non_local_rates_iter = []
    is_training_iter = []
    acausal_posterior_iter = []

    marginal_log_likelihoods = []
    n_iter = 0
    converged = False

    log_likelihood = np.zeros((n_time, n_state_bins))
    log_likelihood[:, state_ind == 1] = np.sum(
        scipy.stats.poisson.logpmf(spikes, zero_rates), axis=-1
    )[:, np.newaxis]

    while not converged and (n_iter < max_iter):

        # Likelihoods
        print("Likelihoods")
        if np.logical_or(n_iter == 0, fit_likelihood):
            coefficients = np.stack(
                [
                    fit_regression(
                        design_matrix,
                        is_training.astype(float),
                        s,
                    )
                    for s in tqdm(spikes.T)
                ],
                axis=1,
            )

            local_rates = np.exp(design_matrix @ coefficients)
            local_rates = np.clip(local_rates, a_min=1e-15, a_max=None)
            log_likelihood[:, state_ind == 0] = np.sum(
                scipy.stats.poisson.logpmf(spikes, local_rates), axis=-1
            )[:, np.newaxis]

            non_local_rates = np.exp(predict_matrix @ coefficients)
            non_local_rates = np.clip(non_local_rates, a_min=1e-15, a_max=None)
            for s, r in zip(spikes.T, non_local_rates.T):
                log_likelihood[:, state_ind == 2] += scipy.stats.poisson.logpmf(
                    s[:, np.newaxis], r[np.newaxis]
                )

        coefficients_iter.append(coefficients)
        local_rates_iter.append(local_rates)
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
            discrete_state_transitions = estimate_transition_matrix(
                causal_state_probabilities,
                predictive_state_probabilities,
                discrete_state_transitions,
                acausal_state_probabilities,
            )

        if fit_likelihood:
            is_training = acausal_posterior[:, 0]

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
        coefficients_iter,
        local_rates_iter,
        non_local_rates_iter,
        is_training_iter,
        acausal_posterior_iter,
    )
    return (
        predicted_state,
        acausal_posterior,
        acausal_state_probabilities,
        causal_posterior,
        marginal_log_likelihoods,
        *debug,
    )


def setup_nonlocal_hmm_model(
    is_ripple,
    spikes,
    position,
    env,
    sampling_frequency=500,
    df=15,
    no_spike_to_no_spike_prob=0.9999,
    no_spike_rate=1e-5,
):

    state_names = ["local", "no spike", "non-local"]

    n_states = len(state_names)

    initial_conditions = np.zeros((n_states,))
    initial_conditions[0] = 1.0

    is_training = ~is_ripple

    local_to_local_prob = (
        np.logical_and(~is_ripple[:-1], ~is_ripple[1:]).sum() / (~is_ripple[:-1]).sum()
    )

    non_local_to_non_local_prob = (
        np.logical_and(is_ripple[:-1], is_ripple[1:]).sum() / is_ripple[:-1].sum()
    )

    discrete_state_transitions = np.asarray(
        [
            [
                local_to_local_prob,
                (1 - local_to_local_prob) / 2,
                (1 - local_to_local_prob) / 2,
            ],
            [
                (1 - no_spike_to_no_spike_prob) / 2,
                no_spike_to_no_spike_prob,
                (1 - no_spike_to_no_spike_prob) / 2,
            ],
            [
                (1 - non_local_to_non_local_prob) / 2,
                (1 - non_local_to_non_local_prob) / 2,
                non_local_to_non_local_prob,
            ],
        ]
    )

    data = {"x": position}
    design_matrix = dmatrix(f"bs(x, df={df})", data)
    zero_rates = np.ones_like(spikes) * no_spike_rate / sampling_frequency
    predict_matrix = make_spline_predict_matrix(
        design_matrix.design_info, env.place_bin_centers_
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(np.log(discrete_state_transitions + np.spacing(1)))

    return (
        design_matrix,
        predict_matrix,
        discrete_state_transitions,
        zero_rates,
        is_training,
        state_names,
    )


def fit_hmm_model(
    spikes,
    design_matrix,
    predict_matrix,
    discrete_state_transitions,
    zero_rates,
    is_training,
    tolerance=1e-4,
    max_iter=3,
    fit_likelihood=True,
    fit_inital_conditions=False,
    fit_discrete_transition=False,
):
    n_time = spikes.shape[0]
    n_states = discrete_state_transitions.shape[0]

    marginal_log_likelihoods = []
    n_iter = 0
    converged = False

    log_likelihood = np.zeros((n_time, n_states))
    log_likelihood[:, 1] = np.sum(
        scipy.stats.poisson.logpmf(spikes, zero_rates), axis=-1
    )

    while not converged and (n_iter < max_iter):

        # Likelihoods
        print("Likelihoods")
        if np.logical_or(n_iter == 0, fit_likelihood):
            coefficients = np.stack(
                [
                    fit_regression(design_matrix, is_training.astype(float), s)
                    for s in tqdm(spikes.T)
                ],
                axis=1,
            )

            local_rates = np.exp(design_matrix @ coefficients)
            local_rates = np.clip(local_rates, a_min=1e-15, a_max=None)
            non_local_rates = np.max(local_rates, axis=0, keepdims=True) - local_rates

            log_likelihood[:, 0] = np.sum(
                scipy.stats.poisson.logpmf(spikes, local_rates), axis=-1
            )
            log_likelihood[:, 2] = np.sum(
                scipy.stats.poisson.logpmf(spikes, non_local_rates), axis=-1
            )

        # Expectation step
        print("Expectation Step")
        causal_posterior, predictive_distribution, marginal_log_likelihood = forward(
            initial_conditions, log_likelihood, discrete_state_transitions
        )
        acausal_posterior = smoother(
            causal_posterior, predictive_distribution, discrete_state_transitions
        )

        # Maximization step
        print("Maximization Step")

        if fit_discrete_transition:
            discrete_state_transitions = estimate_transition_matrix(
                causal_posterior,
                predictive_distribution,
                discrete_state_transitions,
                acausal_posterior,
            )

        if fit_likelihood:
            is_training = acausal_posterior[:, 0]

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

    non_local_rates = np.exp(predict_matrix @ coefficients)
    non_local_rates = np.clip(non_local_rates, a_min=1e-15, a_max=None)

    predicted_state = viterbi(
        initial_conditions, np.exp(log_likelihood), discrete_state_transitions
    )[0]

    return (
        predicted_state,
        acausal_posterior,
        causal_posterior,
        marginal_log_likelihoods,
        non_local_rates,
    )


def plot_hmm_model(
    time,
    position,
    spikes,
    non_local_rates,
    env,
    acausal_posterior,
    state_names,
    figsize=(20, 5),
    time_slice=(0, 10_000),
):
    _, axes = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=figsize)

    sliced_time = time[time_slice]

    neuron_sort_ind = np.argsort(
        env.place_bin_centers_[non_local_rates.argmax(axis=0)].squeeze()
    )
    spike_time_ind, neuron_ind = np.nonzero(spikes[time_slice][:, neuron_sort_ind])

    axes[0].plot(sliced_time, position[time_slice])

    axes[1].scatter(sliced_time[spike_time_ind], neuron_ind, s=1)
    h = axes[2].plot(sliced_time, acausal_posterior[time_slice])
    axes[2].legend(h, state_names)
    plt.xlim((sliced_time.min(), sliced_time.max()))


def plot_switching_model(
    time,
    position,
    spikes,
    non_local_rates,
    env,
    state_ind,
    acausal_state_probabilities,
    acausal_posterior,
    state_names,
    figsize=(20, 5),
    time_slice=(0, 10_000),
):

    _, axes = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=figsize)

    sliced_time = time[time_slice]

    t, x = np.meshgrid(time, env.place_bin_centers_)

    neuron_sort_ind = np.argsort(
        env.place_bin_centers_[non_local_rates.argmax(axis=0)].squeeze()
    )
    spike_time_ind, neuron_ind = np.nonzero(spikes[time_slice][:, neuron_sort_ind])

    axes[0].scatter(sliced_time[spike_time_ind], neuron_ind, s=1)
    h = axes[1].plot(sliced_time, acausal_state_probabilities)
    axes[1].legend(h, state_names)
    axes[2].pcolormesh(
        t,
        x,
        acausal_posterior[time_slice, state_ind == 2].T,
        vmin=0.0,
        vmax=0.01,
        cmap="bone_r",
    )
    axes[2].scatter(sliced_time, position[time_slice], s=1, color="magenta", zorder=2)
    plt.xlim((sliced_time.min(), sliced_time.max()))
