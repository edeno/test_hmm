import jax
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from patsy import dmatrix
from tqdm.autonotebook import tqdm

from src.estimate_transition import estimate_stationary_state_transition
from src.hmm import (
    check_converged,
    fit_poisson_regression,
    forward,
    make_spline_predict_matrix,
    smoother,
    viterbi,
)
from src.estimate_transition import make_discrete_state_transitions

jax.config.update("jax_platform_name", "cpu")

EPS = 1e-15


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

    discrete_state_transitions = make_discrete_state_transitions(
        is_ripple, no_spike_to_no_spike_prob
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
        initial_conditions,
        discrete_state_transitions,
        zero_rates,
        is_training,
        state_names,
    )


def fit_hmm_model(
    spikes,
    design_matrix,
    predict_matrix,
    initial_conditions,
    discrete_state_transitions,
    zero_rates,
    is_training,
    tolerance=1e-4,
    max_iter=20,
    fit_likelihood=True,
    fit_inital_conditions=False,
    fit_discrete_transition=False,
):
    n_time = spikes.shape[0]
    n_states = discrete_state_transitions.shape[0]

    marginal_log_likelihoods = []
    non_local_rates_iter = []
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
                    fit_poisson_regression(design_matrix, is_training.astype(float), s)
                    for s in tqdm(spikes.T)
                ],
                axis=1,
            )

            local_rates = np.exp(design_matrix @ coefficients)
            local_rates = np.clip(local_rates, a_min=EPS, a_max=None)
            non_local_rates = np.max(local_rates, axis=0, keepdims=True) - local_rates

            log_likelihood[:, 0] = np.sum(
                scipy.stats.poisson.logpmf(spikes, local_rates), axis=-1
            )
            log_likelihood[:, 2] = np.sum(
                scipy.stats.poisson.logpmf(spikes, non_local_rates), axis=-1
            )

            non_local_rates_iter.append(
                np.clip(np.exp(predict_matrix @ coefficients), a_min=EPS, a_max=None)
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
            discrete_state_transitions = estimate_stationary_state_transition(
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

    predicted_state = viterbi(
        initial_conditions, np.exp(log_likelihood), discrete_state_transitions
    )[0]

    return (
        predicted_state,
        acausal_posterior,
        causal_posterior,
        marginal_log_likelihoods,
        non_local_rates_iter,
        initial_conditions,
        discrete_state_transitions,
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
