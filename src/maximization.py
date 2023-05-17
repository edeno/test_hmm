from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import softmax
from scipy.optimize import minimize
from tqdm.auto import tqdm

from src.test_non_local_switching_model import load_data, setup_nonlocal_switching_model


def _normalize(u, axis=0, eps=1e-15):
    """Normalizes the values within the axis in a way that they sum up to 1.
    Args:
        u: Input array to normalize.
        axis: Axis over which to normalize.
        eps: Minimum value threshold for numerical stability.
    Returns:
        Tuple of the normalized values, and the normalizing denominator.
    """
    u = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = jnp.where(c == 0, 1, c)
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
    new_probs = probs * jnp.exp(ll - ll_max)
    new_probs, norm = _normalize(new_probs)
    log_norm = jnp.log(norm) + ll_max
    return new_probs, log_norm


def _predict(probs, A):
    return A.T @ probs


@partial(jax.jit)
def hmm_filter(
    initial_distribution,
    transition_matrix,
    log_likelihoods,
):
    r"""Forwards filtering
    Transition matrix may be either 2D (if transition probabilities are fixed) or 3D
    if the transition probabilities vary over time. Alternatively, the transition
    matrix may be specified via `transition_fn`, which takes in a time index $t$ and
    returns a transition matrix.
    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
    Returns:
        filtered posterior distribution
    """
    num_timesteps, num_states = log_likelihoods.shape

    def _step(carry, t):
        log_normalizer, predicted_probs = carry

        ll = log_likelihoods[t]

        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        log_normalizer += log_norm
        predicted_probs_next = _predict(filtered_probs, transition_matrix)

        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    carry = (0.0, initial_distribution)
    (log_normalizer, _), (filtered_probs, predicted_probs) = jax.lax.scan(
        _step, carry, jnp.arange(num_timesteps)
    )

    return log_normalizer, filtered_probs, predicted_probs


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

    initial_condition = np.array([np.log(np.average(spikes, weights=weights))])
    initial_condition = np.concatenate(
        [initial_condition, np.zeros(design_matrix.shape[1] - 1)]
    )

    res = minimize(neglogp, x0=initial_condition, method="BFGS", jac=dlike)

    return res.x


def centered_softmax_forward(y):
    """`softmax(x) = exp(x-c) / sum(exp(x-c))` where c is the last coordinate

    Example
    -------
    > y = np.log([2, 3, 4])
    > np.allclose(centered_softmax_forward(y), [0.2, 0.3, 0.4, 0.1])
    """
    if y.ndim == 1:
        y = jnp.append(y, 0)
    else:
        y = jnp.column_stack((y, jnp.zeros((y.shape[0],))))

    return softmax(y, axis=-1)


def centered_softmax_inverse(y):
    """`softmax(x) = exp(x-c) / sum(exp(x-c))` where c is the last coordinate

    Example
    -------
    > y = np.asarray([0.2, 0.3, 0.4, 0.1])
    > np.allclose(np.exp(centered_softmax_inverse(y)), np.asarray([2,3,4]))
    """
    return jnp.log(y[..., :-1]) - jnp.log(y[..., [-1]])


def create_initial_unconstrained_parameters(
    spikes,
    is_training,
    initial_conditions,
    discrete_state_transitions,
    design_matrix,
    estimate_initial_conditions=True,
    estimate_discrete_state_transitions=True,
    estimate_observation_coefficients=True,
    stationary_transition_matrix=True,
):
    """Creates the initial parameters for the optimization

    Parameters
    ----------
    spikes : np.ndarray, shape (n_time, n_neurons)
    is_training : np.ndarray, shape (n_time,)
    n_observation_coefficients : int
    n_neurons : int
    n_states : int

    Returns
    -------
    x0 : np.ndarray, shape (n_parameters,)
    """
    unconstrained_parameters = []
    parameter_labels = []

    if estimate_observation_coefficients:
        # n_observation_coefficients, n_neurons = design_matrix.shape[1], spikes.shape[1]
        # observation_coefficients = np.ones((n_observation_coefficients, n_neurons))
        # observation_coefficients[0] = spikes[is_training].mean(axis=0)

        # unconstrained_parameters.append(
        #     np.log(
        #         observation_coefficients
        #     ).ravel()  # n_observation_coefficients * n_neurons
        # )

        observation_coefficients = np.stack(
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

        observation_coefficients = (
            observation_coefficients.ravel()
        )  # n_observation_coefficients * n_neurons
        unconstrained_parameters.append(observation_coefficients)
        parameter_labels.append(["oc"] * len(observation_coefficients))

    if estimate_initial_conditions:
        unconstrained_ic = centered_softmax_inverse(
            initial_conditions
        )  # n_state_bins - 1
        unconstrained_parameters.append(unconstrained_ic)
        parameter_labels.append(["ic"] * len(unconstrained_ic))

    if estimate_discrete_state_transitions:
        if stationary_transition_matrix:
            unconstrained_st = centered_softmax_inverse(
                discrete_state_transitions
            ).ravel()  # n_states * (n_states - 1)
        else:
            pass

        unconstrained_parameters.append(unconstrained_st)
        parameter_labels.append(["st"] * len(unconstrained_st))

    return np.concatenate(unconstrained_parameters), np.concatenate(parameter_labels)


def transform_parameters(
    unconstrained_parameters,
    n_observation_coefficients,
    n_neurons,
    n_states,
    n_state_bins,
):

    n_total_observation_coefficients = n_observation_coefficients * n_neurons
    observation_coefficients = unconstrained_parameters[
        :n_total_observation_coefficients
    ].reshape((n_observation_coefficients, n_neurons))

    unconstrained_initial_distribution = unconstrained_parameters[
        n_total_observation_coefficients : n_total_observation_coefficients
        + n_state_bins
        - 1
    ]

    unconstrained_discrete_state_transitions = unconstrained_parameters[
        n_total_observation_coefficients + n_state_bins - 1 :
    ].reshape((n_states, n_states - 1))

    initial_distribution = centered_softmax_forward(unconstrained_initial_distribution)
    discrete_state_transitions = centered_softmax_forward(
        unconstrained_discrete_state_transitions
    )

    return observation_coefficients, initial_distribution, discrete_state_transitions


def transform_parameters(
    unconstrained_parameters,
    n_observation_coefficients,
    parameter_labels,
    n_neurons,
    n_states,
    n_state_bins,
):

    n_total_observation_coefficients = n_observation_coefficients * n_neurons
    observation_coefficients = unconstrained_parameters[
        :n_total_observation_coefficients
    ].reshape((n_observation_coefficients, n_neurons))

    unconstrained_initial_distribution = unconstrained_parameters[
        n_total_observation_coefficients : n_total_observation_coefficients
        + n_state_bins
        - 1
    ]

    unconstrained_discrete_state_transitions = unconstrained_parameters[
        n_total_observation_coefficients + n_state_bins - 1 :
    ].reshape((n_states, n_states - 1))

    initial_distribution = centered_softmax_forward(unconstrained_initial_distribution)
    discrete_state_transitions = centered_softmax_forward(
        unconstrained_discrete_state_transitions
    )

    return observation_coefficients, initial_distribution, discrete_state_transitions


@jax.jit
def neglogp(
    unconstrained_parameters,
    spikes,
    zero_rates,
    design_matrix,
    predict_matrix,
    continuous_state_transitions,
    state_ind,
    eps=1e-15,
):

    n_states = 3
    n_time, n_neurons = spikes.shape
    n_state_bins = len(state_ind)
    n_observation_coefficients = design_matrix.shape[1]

    (
        observation_coefficients,
        initial_distribution,
        discrete_state_transitions,
    ) = transform_parameters(
        unconstrained_parameters,
        n_observation_coefficients,
        n_neurons,
        n_states,
        n_state_bins,
    )

    local_rates = jnp.exp(
        design_matrix @ observation_coefficients
    )  # shape (n_time, n_neurons)
    local_rates = jnp.clip(local_rates, a_min=eps, a_max=None)

    observation_log_likelihood = jnp.zeros((n_time, n_state_bins))

    # Local state
    observation_log_likelihood = observation_log_likelihood.at[:, 0].set(
        jnp.sum(jax.scipy.stats.poisson.logpmf(spikes, local_rates), axis=-1)
    )

    # No-spike state
    observation_log_likelihood = observation_log_likelihood.at[:, 1].set(
        jnp.sum(jax.scipy.stats.poisson.logpmf(spikes, zero_rates), axis=-1)
    )

    # Non-local state
    non_local_rates = jnp.exp(predict_matrix @ observation_coefficients)
    non_local_rates = jnp.clip(non_local_rates, a_min=eps, a_max=None)
    for is_spike, rate in zip(spikes.T, non_local_rates.T):
        observation_log_likelihood = observation_log_likelihood.at[:, 2:].add(
            jax.scipy.stats.poisson.logpmf(is_spike[:, jnp.newaxis], rate[jnp.newaxis])
        )

    discrete_state_transitions_per_bin = discrete_state_transitions[
        jnp.ix_(state_ind, state_ind)
    ]

    transition_matrix = (
        discrete_state_transitions_per_bin * continuous_state_transitions
    )

    marginal_log_likelihood, _, _ = hmm_filter(
        initial_distribution, transition_matrix, observation_log_likelihood
    )

    return -1.0 * marginal_log_likelihood


dlike = jax.grad(neglogp)


@jax.jit
def get_acausal(
    unconstrained_parameters,
    spikes,
    zero_rates,
    design_matrix,
    predict_matrix,
    continuous_state_transitions,
    state_ind,
    eps=1e-15,
):

    n_states = 3
    n_time, n_neurons = spikes.shape
    n_state_bins = len(state_ind)
    n_observation_coefficients = design_matrix.shape[1]

    (
        observation_coefficients,
        initial_distribution,
        discrete_state_transitions,
    ) = transform_parameters(
        unconstrained_parameters,
        n_observation_coefficients,
        n_neurons,
        n_states,
        n_state_bins,
    )

    local_rates = jnp.exp(
        design_matrix @ observation_coefficients
    )  # shape (n_time, n_neurons)
    local_rates = jnp.clip(local_rates, a_min=eps, a_max=None)

    observation_log_likelihood = jnp.zeros((n_time, n_state_bins))

    # Local state
    observation_log_likelihood = observation_log_likelihood.at[:, 0].set(
        jnp.sum(jax.scipy.stats.poisson.logpmf(spikes, local_rates), axis=-1)
    )

    # No-spike state
    observation_log_likelihood = observation_log_likelihood.at[:, 1].set(
        jnp.sum(jax.scipy.stats.poisson.logpmf(spikes, zero_rates), axis=-1)
    )

    # Non-local state
    non_local_rates = jnp.exp(predict_matrix @ observation_coefficients)
    non_local_rates = jnp.clip(non_local_rates, a_min=eps, a_max=None)
    for is_spike, rate in zip(spikes.T, non_local_rates.T):
        observation_log_likelihood = observation_log_likelihood.at[:, 2:].add(
            jax.scipy.stats.poisson.logpmf(is_spike[:, jnp.newaxis], rate[jnp.newaxis])
        )

    discrete_state_transitions_per_bin = discrete_state_transitions[
        jnp.ix_(state_ind, state_ind)
    ]

    transition_matrix = (
        discrete_state_transitions_per_bin * continuous_state_transitions
    )

    marginal_log_likelihood, filtered_probs, predicted_probs = hmm_filter(
        initial_distribution, transition_matrix, observation_log_likelihood
    )

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_probs_next = carry
        t, filtered_probs, predicted_probs_next = args

        # Fold in the next state (Eq. 8.2 of Saarka, 2013)
        # If hard 0. in predicted_probs_next, set relative_probs_next as 0. to avoid NaN values
        relative_probs_next = jnp.where(
            jnp.isclose(predicted_probs_next, 0.0),
            0.0,
            smoothed_probs_next / predicted_probs_next,
        )
        smoothed_probs = filtered_probs * (transition_matrix @ relative_probs_next)
        smoothed_probs /= smoothed_probs.sum()

        return smoothed_probs, smoothed_probs

    # Run the HMM smoother
    carry = filtered_probs[-1]
    args = (
        jnp.arange(n_time - 2, -1, -1),
        filtered_probs[:-1][::-1],
        predicted_probs[1:][::-1],
    )
    _, rev_smoothed_probs = jax.lax.scan(_step, carry, args)

    # Reverse the arrays and return
    smoothed_probs = jnp.row_stack([rev_smoothed_probs[::-1], filtered_probs[-1]])

    return smoothed_probs


def test_non_local():
    is_ripple, spikes, position, env, time = load_data(work_computer=False)

    sampling_frequency = 500.0

    (
        design_matrix,
        predict_matrix,
        initial_conditions,
        discrete_state_transitions,
        continuous_state_transitions,
        state_ind,
        zero_rates,
        is_training,
        state_names,
    ) = setup_nonlocal_switching_model(
        is_ripple,
        spikes,
        position,
        env,
        df=7,
    )

    initial_conditions[0] = 1 - 1e-15
    initial_conditions[1:] = 1e-15 / len(initial_conditions[1:])

    x0, parameter_labels = create_initial_unconstrained_parameters(
        spikes,
        is_training,
        initial_conditions,
        discrete_state_transitions,
        design_matrix,
    )

    res = minimize(
        neglogp,
        x0=x0,
        method="BFGS",
        jac=dlike,
        args=(
            spikes,
            zero_rates,
            design_matrix,
            predict_matrix,
            continuous_state_transitions,
            state_ind,
        ),
        options={"disp": True, "maxiter": 20},
    )
