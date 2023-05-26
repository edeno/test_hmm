import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from patsy import dmatrix
from replay_trajectory_classification import Environment, RandomWalk, Uniform
from replay_trajectory_classification.classifier import joblib
from replay_trajectory_classification.core import atleast_2d
from replay_trajectory_classification.environments import get_n_bins
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from statsmodels.tsa.stattools import lagmat
from tqdm.autonotebook import tqdm
from patsy import build_design_matrices

from src.estimate_transition import (
    estimate_non_stationary_state_transition,
    estimate_stationary_state_transition,
    multinomial_gradient,
    multinomial_hessian,
    multinomial_neg_log_likelihood,
)
from src.hmm import (
    centered_softmax_forward,
    centered_softmax_inverse,
    check_converged,
    fit_poisson_regression,
)

jax.config.update("jax_platform_name", "cpu")

EPS = 1e-15

no_spike_self_transition = 0.999
NO_SPIKE_TRANSITIONS = np.array(
    [
        (1 - no_spike_self_transition) / 2,
        no_spike_self_transition,
        (1 - no_spike_self_transition) / 2,
    ]
)


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
    speed = np.asarray(position_info.nose_vel).astype(float)

    return is_ripple, spikes, position, speed, env, time


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


def gaussian_smooth(
    data: np.ndarray,
    sigma: float,
    sampling_frequency: float,
    axis: int = 0,
    truncate: float = 8.0,
) -> np.ndarray:
    """1D convolution of the data with a Gaussian.
    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is just a wrapper around scipy's
    `gaussian_filter1d`, The support is truncated at 8 standard deviations by default.

    Parameters
    ----------
    data : array_like
    sigma : float
    sampling_frequency : int
    axis : int, optional
    truncate : float, optional

    Returns
    -------
    smoothed_data : array_like
    """
    return gaussian_filter1d(
        data, sigma * sampling_frequency, truncate=truncate, axis=axis, mode="constant"
    )


def estimate_no_spike_times(
    spikes: np.ndarray,
    speed: None | np.ndarray = None,
    sampling_frequency: int = 500,
    sigma: float = 0.015,
    rate_threshold: float = 1e-5,
    speed_threshold: float = 4.0,
) -> np.ndarray:
    spike_rate = (
        gaussian_smooth(
            spikes.sum(axis=1).astype(float),
            sigma=sigma,
            sampling_frequency=sampling_frequency,
        )
        * sampling_frequency
    )
    is_no_spike = spike_rate < rate_threshold

    if speed is not None:
        is_low_speed = speed < speed_threshold
    else:
        is_low_speed = np.ones_like(spike_rate, dtype=bool)

    return np.logical_and(is_no_spike, is_low_speed)


def estimate_initial_discrete_transition(
    is_ripple: np.ndarray,
    spikes: np.ndarray,
    speed: np.ndarray,
    sampling_frequency: int = 500,
    no_spike_rate_threshold: float = 1e-3,
    no_spike_smoothing_sigma: float = 0.015,
    no_spike_speed_threshold: float = 4.0,
    speed_knots: None | np.ndarray = None,
    is_stationary: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    is_no_spike_time = estimate_no_spike_times(
        spikes,
        speed=speed,
        sampling_frequency=sampling_frequency,
        sigma=no_spike_smoothing_sigma,
        rate_threshold=no_spike_rate_threshold,
        speed_threshold=no_spike_speed_threshold,
    )
    is_local_time = np.logical_and(~is_ripple, ~is_no_spike_time)

    state_names = ["local", "no_spike", "non_local"]
    n_states = len(state_names)

    if is_stationary:
        discrete_state_transitions = np.zeros((n_states, n_states))

        is_states = {
            "local": is_local_time.astype(bool).squeeze(),
            "no_spike": is_no_spike_time.astype(bool).squeeze(),
            "non_local": is_ripple.astype(bool).squeeze(),
        }

        for from_state_ind, from_state in enumerate(state_names):
            for to_state_ind, to_state in enumerate(state_names):
                discrete_state_transitions[from_state_ind, to_state_ind] = (
                    np.logical_and(
                        is_states[from_state][:-1], is_states[to_state][1:]
                    ).sum()
                    / is_states[from_state][:-1].sum()
                )

        # if any is zero, set to small number
        # discrete_state_transitions = np.clip(discrete_state_transitions, 1e-16, 1.0 - 1e-16)

        discrete_state_transitions /= discrete_state_transitions.sum(
            axis=1, keepdims=True
        )

        discrete_transition_coefficients = None
        discrete_transition_design_matrix = None
    else:
        from_states = {
            "local": lagmat(is_local_time, maxlag=1).astype(bool).squeeze(),
            "no_spike": lagmat(is_no_spike_time, maxlag=1).astype(bool).squeeze(),
            "non_local": lagmat(is_ripple, maxlag=1).astype(bool).squeeze(),
        }

        if speed_knots is None:
            speed_knots = [1.0, 4.0, 16.0, 32.0, 64.0]

        formula = f"1 + cr(speed, knots={speed_knots}, constraints='center')"
        data = {"speed": lagmat(speed, maxlag=1)}
        discrete_transition_design_matrix = dmatrix(formula, data)

        n_time, n_coefficients = discrete_transition_design_matrix.shape

        response = np.zeros((n_time, n_states))
        response[is_local_time.astype(bool), 0] = 1.0
        response[is_no_spike_time.astype(bool), 1] = 1.0
        response[is_ripple.astype(bool), 2] = 1.0

        starting_prob = np.zeros(
            (
                n_coefficients,
                n_states - 1,
            )
        ).ravel()

        discrete_state_transitions = np.zeros((n_time, n_states, n_states))
        discrete_transition_coefficients = np.zeros(
            (n_coefficients, n_states, n_states - 1)
        )

        for from_state_ind, from_state in enumerate(state_names):
            is_from_state = from_states[from_state]
            res = minimize(
                multinomial_neg_log_likelihood,
                x0=starting_prob,
                jac=multinomial_gradient,
                hess=multinomial_hessian,
                args=(
                    discrete_transition_design_matrix[is_from_state],
                    response[is_from_state],
                ),
                method="Newton-CG",
                options={
                    "disp": True,
                    "maxiter": 100,
                },
            )
            discrete_transition_coefficients[:, from_state_ind, :] = res.x.reshape(
                (
                    n_coefficients,
                    n_states - 1,
                )
            )
            discrete_state_transitions[:, from_state_ind, :] = centered_softmax_forward(
                discrete_transition_design_matrix
                @ discrete_transition_coefficients[:, from_state_ind, :]
            )

    return (
        discrete_state_transitions,
        discrete_transition_coefficients,
        discrete_transition_design_matrix,
    )


def estimate_initial_discrete_transition2(
    is_ripple: np.ndarray,
    spikes: np.ndarray,
    speed: np.ndarray,
    sampling_frequency: int = 500,
    no_spike_rate_threshold: float = 1e-3,
    no_spike_smoothing_sigma: float = 0.015,
    no_spike_speed_threshold: float = 4.0,
    speed_knots: None | np.ndarray = None,
    is_stationary: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    is_no_spike_time = estimate_no_spike_times(
        spikes,
        speed=speed,
        sampling_frequency=sampling_frequency,
        sigma=no_spike_smoothing_sigma,
        rate_threshold=no_spike_rate_threshold,
        speed_threshold=no_spike_speed_threshold,
    )
    is_local_time = np.logical_and(~is_ripple, ~is_no_spike_time)

    state_names = ["local", "no_spike", "non_local"]
    n_states = len(state_names)

    if is_stationary:
        discrete_state_transitions = np.zeros((n_states, n_states))

        is_states = {
            "local": is_local_time.astype(bool).squeeze(),
            "no_spike": is_no_spike_time.astype(bool).squeeze(),
            "non_local": is_ripple.astype(bool).squeeze(),
        }

        for from_state_ind, from_state in enumerate(state_names):
            for to_state_ind, to_state in enumerate(state_names):
                discrete_state_transitions[from_state_ind, to_state_ind] = (
                    np.logical_and(
                        is_states[from_state][:-1], is_states[to_state][1:]
                    ).sum()
                    / is_states[from_state][:-1].sum()
                )

        discrete_state_transitions /= discrete_state_transitions.sum(
            axis=1, keepdims=True
        )

        # if any is zero, set to small number
        # discrete_state_transitions = np.clip(discrete_state_transitions, 1e-16, 1.0 - 1e-16)

        discrete_transition_coefficients = None
        discrete_transition_design_matrix = None
    else:
        if speed_knots is None:
            speed_knots = [1.0, 4.0, 16.0, 32.0, 64.0]

        n_time = spikes.shape[0]
        response = np.zeros((n_time, n_states))
        response[is_local_time.astype(bool), 0] = 1.0
        response[is_no_spike_time.astype(bool), 1] = 1.0
        response[is_ripple.astype(bool), 2] = 1.0

        data = {
            "speed": lagmat(speed, maxlag=1),
            "local": lagmat(is_local_time, maxlag=1).astype(bool).squeeze(),
            "no_spike": lagmat(is_no_spike_time, maxlag=1).astype(bool).squeeze(),
            "non_local": lagmat(is_ripple, maxlag=1).astype(bool).squeeze(),
        }

        discrete_state_transitions = np.zeros((n_time, n_states, n_states))
        discrete_transition_coefficients = []

        for from_state_ind, from_state in enumerate(state_names):
            formula = f"1 + {from_state} + bs(speed, knots={speed_knots})"
            discrete_transition_design_matrix = dmatrix(formula, data)
            n_coefficients = discrete_transition_design_matrix.shape[1]
            starting_prob = np.zeros(
                (
                    n_coefficients,
                    n_states - 1,
                )
            ).ravel()

            res = minimize(
                multinomial_neg_log_likelihood,
                x0=starting_prob,
                jac=multinomial_gradient,
                hess=multinomial_hessian,
                args=(
                    discrete_transition_design_matrix,
                    response,
                ),
                method="Newton-CG",
                options={
                    "disp": True,
                    "maxiter": 100,
                },
            )
            coef = res.x.reshape(
                (
                    n_coefficients,
                    n_states - 1,
                )
            )

            # remove the indicator of the from_state
            discrete_transition_coefficients.append(np.delete(coef, 1, axis=0))
            discrete_state_transitions[:, from_state_ind, :] = centered_softmax_forward(
                discrete_transition_design_matrix @ coef
            )
        discrete_transition_coefficients = np.stack(
            discrete_transition_coefficients, axis=1
        )

    return (
        discrete_state_transitions,
        discrete_transition_coefficients,
        np.delete(discrete_transition_design_matrix, 1, axis=1),
    )


def estimate_initial_discrete_transition3(
    is_ripple: np.ndarray,
    spikes: np.ndarray,
    speed: np.ndarray,
    sampling_frequency: int = 500,
    no_spike_rate_threshold: float = 1e-3,
    no_spike_smoothing_sigma: float = 0.015,
    no_spike_speed_threshold: float = 4.0,
    speed_knots: None | np.ndarray = None,
    is_stationary: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    is_no_spike_time = estimate_no_spike_times(
        spikes,
        speed=speed,
        sampling_frequency=sampling_frequency,
        sigma=no_spike_smoothing_sigma,
        rate_threshold=no_spike_rate_threshold,
        speed_threshold=no_spike_speed_threshold,
    )
    is_no_spike_time[is_ripple] = False
    is_local_time = np.logical_and(~is_ripple, ~is_no_spike_time)

    state_names = ["local", "no_spike", "non_local"]
    n_states = len(state_names)

    if is_stationary:
        discrete_state_transitions = np.zeros((n_states, n_states))

        is_states = {
            "local": is_local_time.astype(bool).squeeze(),
            "no_spike": is_no_spike_time.astype(bool).squeeze(),
            "non_local": is_ripple.astype(bool).squeeze(),
        }

        for from_state_ind, from_state in enumerate(state_names):
            for to_state_ind, to_state in enumerate(state_names):
                discrete_state_transitions[from_state_ind, to_state_ind] = (
                    np.logical_and(
                        is_states[from_state][:-1], is_states[to_state][1:]
                    ).sum()
                    / is_states[from_state][:-1].sum()
                )

        discrete_state_transitions /= discrete_state_transitions.sum(
            axis=1, keepdims=True
        )
        # if any is zero, set to small number
        # discrete_state_transitions = np.clip(
        #     discrete_state_transitions, a_min=1e-16, a_max=1.0 - 1e-16
        # )

        discrete_transition_coefficients = None
        discrete_transition_design_matrix = None
    else:
        is_states = {
            "local": is_local_time.astype(bool).squeeze(),
            "no_spike": is_no_spike_time.astype(bool).squeeze(),
            "non_local": is_ripple.astype(bool).squeeze(),
        }

        discrete_state_transitions = np.zeros((n_states, n_states))
        for from_state_ind, from_state in enumerate(state_names):
            for to_state_ind, to_state in enumerate(state_names):
                discrete_state_transitions[from_state_ind, to_state_ind] = (
                    np.logical_and(
                        is_states[from_state][:-1], is_states[to_state][1:]
                    ).sum()
                    / is_states[from_state][:-1].sum()
                )

        discrete_state_transitions /= discrete_state_transitions.sum(
            axis=1, keepdims=True
        )
        # if any is zero, set to small number
        discrete_state_transitions = np.clip(
            discrete_state_transitions, a_min=1e-16, a_max=1.0 - 1e-16
        )
        discrete_state_transitions /= discrete_state_transitions.sum(
            axis=1, keepdims=True
        )

        if speed_knots is None:
            speed_knots = [1.0, 4.0, 16.0, 32.0, 64.0]

        formula = f"1 + bs(speed, knots={speed_knots})"
        data = {"speed": lagmat(speed, maxlag=1)}
        discrete_transition_design_matrix = dmatrix(formula, data)

        n_time, n_coefficients = discrete_transition_design_matrix.shape

        discrete_transition_coefficients = np.zeros(
            (n_coefficients, n_states, n_states - 1)
        )
        discrete_transition_coefficients[0] = centered_softmax_inverse(
            discrete_state_transitions
        )

        discrete_state_transitions = discrete_state_transitions[np.newaxis] * np.ones(
            (n_time, n_states, n_states)
        )

    return (
        discrete_state_transitions,
        discrete_transition_coefficients,
        discrete_transition_design_matrix,
    )


def make_transition_from_diag(diag):
    n_states = len(diag)
    transition_matrix = diag * np.eye(n_states)
    off_diag = ((1.0 - diag) / (n_states - 1.0))[:, np.newaxis]
    transition_matrix += np.ones((n_states, n_states)) * off_diag - off_diag * np.eye(
        n_states
    )

    return transition_matrix


def estimate_initial_discrete_transition4(
    is_ripple: np.ndarray,
    spikes: np.ndarray,
    speed: np.ndarray,
    sampling_frequency: int = 500,
    no_spike_rate_threshold: float = 1e-3,
    no_spike_smoothing_sigma: float = 0.015,
    no_spike_speed_threshold: float = 4.0,
    speed_knots: None | np.ndarray = None,
    is_stationary: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_names = [
        "local",
        "no_spike",
        "non-local continuous",
        "non-local fragmented",
    ]
    n_states = len(state_names)

    if is_stationary:
        diag = np.array([0.90, 0.90, 0.90, 0.98])

        discrete_state_transitions = make_transition_from_diag(diag)

        discrete_transition_coefficients = None
        discrete_transition_design_matrix = None
    else:
        diag = np.array([0.90, 0.90, 0.90, 0.98])
        discrete_state_transitions = make_transition_from_diag(diag)

        if speed_knots is None:
            speed_knots = [1.0, 4.0, 16.0, 32.0, 64.0]

        formula = f"1 + bs(speed, knots={speed_knots})"
        data = {"speed": lagmat(speed, maxlag=1)}
        discrete_transition_design_matrix = dmatrix(formula, data)

        n_time, n_coefficients = discrete_transition_design_matrix.shape

        discrete_transition_coefficients = np.zeros(
            (n_coefficients, n_states, n_states - 1)
        )
        discrete_transition_coefficients[0] = centered_softmax_inverse(
            discrete_state_transitions
        )

        discrete_state_transitions = discrete_state_transitions[np.newaxis] * np.ones(
            (n_time, n_states, n_states)
        )

    return (
        discrete_state_transitions,
        discrete_transition_coefficients,
        discrete_transition_design_matrix,
    )


def setup_nonlocal_switching_model(
    is_ripple: np.ndarray,
    spikes: np.ndarray,
    position: np.ndarray,
    speed: np.ndarray,
    env: Environment,
    no_spike_rate_threshold: float = 1e-3,
    no_spike_smoothing_sigma: float = 0.015,
    no_spike_speed_threshold: float = 4.0,
    speed_knots: None | np.ndarray = None,
    sampling_frequency: int = 500,
    emission_knot_spacing: float = 8.0,
    no_spike_rate: float = 1e-10,
    is_stationary_discrete_transition: bool = False,
    include_no_spike_state: bool = True,
    rw_movement_var=6.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """Setup the non-local switching model.

    Parameters
    ----------
    is_ripple : np.ndarray
    spikes : np.ndarray
    position : np.ndarray
    speed : np.ndarray
    env : Environment
    no_spike_rate_threshold : float, optional
    no_spike_smoothing_sigma : float, optional
    is_stationary_discrete_transition : bool, optional
    speed_knots : None | np.ndarray, optional
    sampling_frequency : int, optional
    df : int, optional
    no_spike_rate : float, optional

    Returns
    -------
    emission_design_matrix : np.ndarray. shape (n_time, n_features)
    emission_predict_matrix : np.ndarray, shape (n_features,)
    initial_conditions : np.ndarray, shape (n_state_bins,)
    discrete_state_transitions : np.ndarray, shape (n_time, n_states, n_states)
    discrete_transition_coefficients : np.ndarray, shape (n_features, n_states, n_states - 1)
    discrete_transition_design_matrix : np.ndarray, shape (n_time, n_features)
    continuous_state_transitions : np.ndarray, shape (n_time, n_state_bins, n_state_bins)
    state_ind : np.ndarray, shape (n_state_bins,)
    no_spike_rates : np.ndarray, shape (n_neurons,)
    is_training : np.ndarray, shape (n_time,)
    state_names : list[str]
    """

    random_walk = RandomWalk(movement_var=rw_movement_var).make_state_transition([env])
    uniform = Uniform().make_state_transition([env])

    n_env_bins = env.place_bin_centers_.shape[0]

    if include_no_spike_state:
        state_names = [
            "local",
            "no_spike",
            "non-local continuous",
            "non-local fragmented",
        ]
        bin_sizes = [1, 1, n_env_bins, n_env_bins]
    else:
        state_names = ["local", "non-local continuous", "non-local fragmented"]
        bin_sizes = [1, n_env_bins, n_env_bins]

    n_states = len(state_names)

    state_ind = np.concatenate(
        [
            ind * np.ones((bin_size,), dtype=int)
            for ind, bin_size in enumerate(bin_sizes)
        ]
    )
    n_state_bins = len(state_ind)

    # Assume we start in the local state
    initial_conditions = np.zeros((n_state_bins,))
    initial_conditions[state_ind == 0] = 1.0

    # Estimate the discrete transition matrix
    (
        discrete_state_transitions,
        discrete_transition_coefficients,
        discrete_transition_design_matrix,
    ) = estimate_initial_discrete_transition4(
        is_ripple,
        spikes,
        speed,
        sampling_frequency=sampling_frequency,
        no_spike_rate_threshold=no_spike_rate_threshold,
        no_spike_smoothing_sigma=no_spike_smoothing_sigma,
        no_spike_speed_threshold=no_spike_speed_threshold,
        speed_knots=speed_knots,
        is_stationary=is_stationary_discrete_transition,
    )

    if not include_no_spike_state:
        discrete_state_transitions = np.delete(discrete_state_transitions, 1, axis=-1)
        discrete_state_transitions = np.delete(discrete_state_transitions, 1, axis=-2)
        if not is_stationary_discrete_transition:
            discrete_transition_coefficients = np.delete(
                discrete_transition_coefficients, 1, axis=-1
            )
            discrete_transition_coefficients = np.delete(
                discrete_transition_coefficients, 1, axis=-2
            )

        discrete_state_transitions /= discrete_state_transitions.sum(
            axis=-1, keepdims=True
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
                    if (
                        state_names[from_state] == "non-local continuous"
                        or state_names[from_state] == "non-local"
                    ):
                        continuous_state_transitions[inds] = random_walk
                    elif state_names[from_state] == "non-local fragmented":
                        continuous_state_transitions[inds] = uniform

    # Don't fit the place fields on ripples
    is_training = ~is_ripple

    emission_design_matrix = make_spline_design_matrix(
        position, env.place_bin_edges_, knot_spacing=emission_knot_spacing
    )
    emission_predict_matrix = make_spline_predict_matrix(
        emission_design_matrix.design_info, env.place_bin_centers_
    )

    n_neurons = spikes.shape[1]
    no_spike_rates = np.ones((n_neurons,)) * no_spike_rate / sampling_frequency

    return (
        emission_design_matrix,
        emission_predict_matrix,
        initial_conditions,
        discrete_state_transitions,
        discrete_transition_coefficients,
        discrete_transition_design_matrix,
        continuous_state_transitions,
        state_ind,
        no_spike_rates,
        is_training,
        state_names,
    )


def get_transition_matrix(
    continuous_state_transitions, discrete_state_transitions, state_ind, t
):
    if discrete_state_transitions.ndim == 2:
        # could consider caching this
        return (
            continuous_state_transitions
            * discrete_state_transitions[np.ix_(state_ind, state_ind)]
        )
    else:
        return (
            continuous_state_transitions
            * discrete_state_transitions[t][np.ix_(state_ind, state_ind)]
        )


def forward(
    initial_conditions: np.ndarray,
    log_likelihood: np.ndarray,
    discrete_state_transitions: np.ndarray,
    continuous_state_transitions: np.ndarray,
    state_ind: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Causal algorithm for computing the posterior distribution of the hidden states of a switching model

    Parameters
    ----------
    initial_conditions : np.ndarray, shape (n_states,)
    log_likelihood : np.ndarray, shape (n_time, n_states)
    discrete_state_transitions : np.ndarray, shape (n_time, n_states, n_states) or (n_states, n_states)
    continuous_state_transitions : np.ndarray, shape (n_state_bins, n_state_bins)
    state_ind : np.ndarray, shape (n_state_bins,)

    Returns
    -------
    causal_posterior : np.ndarray, shape (n_time, n_states)
        Causal posterior distribution
    predictive_distribution : np.ndarray, shape (n_time, n_states)
        One step predictive distribution
    marginal_likelihood : float

    """
    n_time = log_likelihood.shape[0]

    predictive_distribution = np.zeros_like(log_likelihood)
    causal_posterior = np.zeros_like(log_likelihood)
    max_log_likelihood = np.nanmax(log_likelihood, axis=1, keepdims=True)
    likelihood = np.exp(log_likelihood - max_log_likelihood)
    likelihood = np.clip(
        likelihood, a_min=np.nextafter(0.0, 1.0, dtype=np.float32), a_max=1.0
    )

    predictive_distribution[0] = initial_conditions
    causal_posterior[0] = initial_conditions * likelihood[0]
    norm = np.nansum(causal_posterior[0])
    marginal_likelihood = np.log(norm)
    causal_posterior[0] /= norm

    for t in range(1, n_time):
        # Predict
        predictive_distribution[t] = (
            get_transition_matrix(
                continuous_state_transitions, discrete_state_transitions, state_ind, t
            ).T
            @ causal_posterior[t - 1]
        )
        # Update
        causal_posterior[t] = predictive_distribution[t] * likelihood[t]
        # Normalize
        norm = np.nansum(causal_posterior[t])
        marginal_likelihood += np.log(norm)
        causal_posterior[t] /= norm

    marginal_likelihood += np.sum(max_log_likelihood)

    return causal_posterior, predictive_distribution, marginal_likelihood


def smoother(
    causal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    discrete_state_transitions: np.ndarray,
    continuous_state_transitions: np.ndarray,
    state_ind: np.ndarray,
) -> np.ndarray:
    """Acausal algorithm for computing the posterior distribution of the hidden states of a switching model

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_states)
    predictive_distribution : np.ndarray, shape (n_time, n_states)
        One step predictive distribution
    transition_matrix : np.ndarray, shape (n_states, n_states)

    Returns
    -------
    acausal_posterior, np.ndarray, shape (n_time, n_states)

    """
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
            get_transition_matrix(
                continuous_state_transitions, discrete_state_transitions, state_ind, t
            )
            @ relative_distribution
        )
        acausal_posterior[t] /= acausal_posterior[t].sum()

    return acausal_posterior


def viterbi(
    initial_conditions: np.ndarray,
    log_likelihood: np.ndarray,
    state_ind: np.ndarray,
    continuous_state_transitions: np.ndarray,
    discrete_state_transitions: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Calculate the most likely path through the hidden states.

    Parameters
    ----------
    initial_conditions : np.ndarray, shape (n_states,)
    log_likelihood : np.ndarray, shape (n_time, n_states)
    transition_matrix : np.ndarray, shape (n_states, n_states)

    Returns
    -------
    most_likely_path : np.ndarray, shape (n_time,)
    path_probability : float
    """

    EPS = 1e-15
    n_time, n_states = log_likelihood.shape

    log_initial_conditions = np.log(np.clip(initial_conditions, a_min=EPS, a_max=1.0))

    path_log_prob = np.ones_like(log_likelihood)
    back_pointer = np.zeros_like(log_likelihood, dtype=int)

    path_log_prob[0] = log_initial_conditions + log_likelihood[0]

    for time_ind in range(1, n_time):
        prior = path_log_prob[time_ind - 1] + np.log(
            np.clip(
                get_transition_matrix(
                    continuous_state_transitions,
                    discrete_state_transitions,
                    state_ind,
                    time_ind,
                ),
                a_min=EPS,
                a_max=1.0,
            )
        )
        for state_idx in range(n_states):
            back_pointer[time_ind, state_idx] = np.argmax(prior[state_idx])
            path_log_prob[time_ind, state_idx] = (
                prior[state_idx, back_pointer[time_ind, state_idx]]
                + log_likelihood[time_ind, state_idx]
            )

    # Find the best accumulated path prob in the last time bin
    # and then trace back the best path
    best_path = np.zeros((n_time,), dtype=int)
    best_path[-1] = np.argmax(path_log_prob[-1])
    for time_ind in range(n_time - 2, -1, -1):
        best_path[time_ind] = back_pointer[time_ind + 1, best_path[time_ind + 1]]

    return best_path, np.exp(np.max(path_log_prob[-1]))


def fit_switching_model(
    spikes,
    emission_design_matrix,
    emission_predict_matrix,
    initial_conditions,
    discrete_state_transitions,
    continuous_state_transitions,
    state_ind,
    no_spike_rates,
    is_training,
    env,
    discrete_transition_coefficients=None,
    discrete_transition_design_matrix=None,
    tolerance=1e-4,
    max_iter=20,
    fit_likelihood=False,
    fit_inital_conditions=False,
    fit_discrete_transition=True,
    concentration: float = 1.0,
    stickiness: float = 0.0,
    transition_regularization: float = 1e-5,
    debug: bool = False,
    log_likelihood: np.ndarray = None,
):
    """Fit non-local switching model to spike data.

    Parameters
    ----------
    spikes : np.ndarray, shape (n_time, n_neurons)
    design_matrix : np.ndarray, shape (n_time, n_features)
    predict_matrix : np.ndarray, shape (n_position_bins, n_features)
    initial_conditions : np.ndarray, shape (n_state_bins,)
    discrete_state_transitions : np.ndarray, shape (n_states, n_states)
    continuous_state_transitions : np.ndarray, shape (n_state_bins, n_state_bins)
    state_ind : np.ndarray, shape (n_state_bins,)
    no_spike_rates : np.ndarray, shape (n_neurons,)
    is_training : np.ndarray, shape (n_time,)
    discrete_transition_coefficients : np.ndarray, shape (n_features, n_states, n_states - 1), optional
    discrete_transition_design_matrix : np.ndarray, shape (n_time, n_features), optional
    tolerance : float, optional
    max_iter : int, optional
    fit_likelihood : bool, optional
    fit_inital_conditions : bool, optional
    fit_discrete_transition : bool, optional
    debug : bool, optional

    Returns
    -------
    predicted_state : np.ndarray, shape (n_time, n_states)
    acausal_posterior : np.ndarray, shape (n_time, n_state_bins)
    acausal_state_probabilities : np.ndarray, shape (n_time, n_states)
    causal_posterior : np.ndarray, shape (n_time, n_state_bins)
    marginal_log_likelihoods : np.ndarray, shape (n_iter,)
    initial_conditions : np.ndarray, shape (n_state_bins,)
    discrete_state_transitions : np.ndarray, shape (n_states, n_states)

    """

    n_time = spikes.shape[0]
    n_states = discrete_state_transitions.shape[-1]
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

    provided_likelihood = log_likelihood is not None

    if not provided_likelihood:
        log_likelihood = np.zeros((n_time, n_state_bins))
    else:
        coefficients = []
        local_rates = []
        non_local_rates = []

    include_no_spike_state = n_states > 3

    if include_no_spike_state:
        log_likelihood[:, state_ind == 1] = np.sum(
            scipy.stats.poisson.logpmf(spikes, no_spike_rates), axis=-1, keepdims=True
        )

    while not converged and (n_iter < max_iter):
        print(discrete_state_transitions)
        # Likelihoods
        print("Likelihoods")
        if np.logical_and(
            np.logical_or(n_iter == 0, fit_likelihood), not provided_likelihood
        ):
            coefficients = []
            local_rates = []
            non_local_rates = []

            for neuron_spikes in tqdm(spikes.T):
                coef = fit_poisson_regression(
                    emission_design_matrix,
                    is_training.astype(float),
                    neuron_spikes,
                    l2_penalty=1e-5,
                )
                coefficients.append(coef)

                local_rate = np.exp(emission_design_matrix @ coef)
                local_rate = np.clip(local_rate, a_min=EPS, a_max=None)
                local_rates.append(local_rate)
                log_likelihood[:, state_ind == 0] += scipy.stats.poisson.logpmf(
                    neuron_spikes, local_rate
                )[:, np.newaxis]

                non_local_rate = np.exp(emission_predict_matrix @ coef)
                non_local_rate[~env.is_track_interior_] = EPS
                non_local_rate = np.clip(non_local_rate, a_min=EPS, a_max=None)
                non_local_rates.append(non_local_rate)
                if include_no_spike_state:
                    # continuous
                    log_likelihood[:, state_ind == 2] += scipy.stats.poisson.logpmf(
                        neuron_spikes[:, np.newaxis], non_local_rate[np.newaxis]
                    )
                    log_likelihood[:, state_ind == 2][
                        :, ~env.is_track_interior_
                    ] = np.nan
                else:
                    # continuous
                    log_likelihood[:, state_ind == 1] += scipy.stats.poisson.logpmf(
                        neuron_spikes[:, np.newaxis], non_local_rate[np.newaxis]
                    )
                    log_likelihood[:, state_ind == 1][
                        :, ~env.is_track_interior_
                    ] = np.nan

            if include_no_spike_state:
                log_likelihood[:, state_ind == 3] = log_likelihood[:, state_ind == 2]
            else:
                log_likelihood[:, state_ind == 2] = log_likelihood[:, state_ind == 1]
            coefficients = np.stack(coefficients, axis=1)
            local_rates = np.stack(local_rates, axis=1)
            non_local_rates = np.stack(non_local_rates, axis=1)
            non_local_rates[~env.is_track_interior_, :] = np.nan

        coefficients_iter.append(coefficients)
        local_rates_iter.append(local_rates)
        non_local_rates_iter.append(non_local_rates)
        is_training_iter.append(is_training.astype(float))

        # Expectation step
        print("Expectation Step")
        causal_posterior, predictive_distribution, marginal_log_likelihood = forward(
            initial_conditions,
            log_likelihood,
            discrete_state_transitions,
            continuous_state_transitions,
            state_ind,
        )
        acausal_posterior = smoother(
            causal_posterior,
            predictive_distribution,
            discrete_state_transitions,
            continuous_state_transitions,
            state_ind,
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
            if (
                discrete_transition_coefficients is not None
                and discrete_transition_design_matrix is not None
            ):
                (
                    discrete_transition_coefficients,
                    discrete_state_transitions,
                ) = estimate_non_stationary_state_transition(
                    discrete_transition_coefficients,
                    discrete_transition_design_matrix,
                    causal_state_probabilities,
                    predictive_state_probabilities,
                    discrete_state_transitions,
                    acausal_state_probabilities,
                    concentration=concentration,
                    stickiness=stickiness,
                    transition_regularization=transition_regularization,
                )

            else:
                discrete_state_transitions = estimate_stationary_state_transition(
                    causal_state_probabilities,
                    predictive_state_probabilities,
                    discrete_state_transitions,
                    acausal_state_probabilities,
                    concentration=concentration,
                    stickiness=stickiness,
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

    print("Estimating predicted state")
    # predicted_state = viterbi(
    #     initial_conditions,
    #     log_likelihood,
    #     state_ind,
    #     continuous_state_transitions,
    #     discrete_state_transitions,
    # )[0]
    predicted_state = np.full((n_time,), np.nan)
    print("Done")

    debug = (
        (
            coefficients_iter,
            local_rates_iter,
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
        discrete_transition_coefficients,
        discrete_transition_design_matrix,
        non_local_rates,
        log_likelihood,
        *debug,
    )


def plot_switching_model(
    time,
    position,
    spikes,
    speed,
    non_local_rates,
    env,
    state_ind,
    acausal_state_probabilities,
    acausal_posterior,
    state_names,
    figsize=(20, 5),
    time_slice=None,
    posterior_max=0.25,
):
    if time_slice is None:
        time_slice = slice(0, len(time))

    _, axes = plt.subplots(
        4,
        1,
        sharex=True,
        constrained_layout=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": [2, 1, 3, 1]},
    )

    sliced_time = time[time_slice]

    t, x = np.meshgrid(sliced_time, env.place_bin_centers_)

    neuron_sort_ind = np.argsort(
        env.place_bin_centers_[np.nanargmax(non_local_rates, axis=0)].squeeze()
    )
    spike_time_ind, neuron_ind = np.nonzero(spikes[time_slice][:, neuron_sort_ind])

    conditional_non_local_acausal_posterior = (
        acausal_posterior[time_slice, state_ind == 2]
        + acausal_posterior[time_slice, state_ind == 3]
    ) / (
        acausal_state_probabilities[time_slice, [2]]
        + acausal_state_probabilities[time_slice, [3]]
    )
    conditional_non_local_acausal_posterior[:, ~env.is_track_interior_] = np.nan

    axes[0].scatter(sliced_time[spike_time_ind], neuron_ind, s=1)
    axes[0].set_ylabel("Neuron")

    h = axes[1].plot(sliced_time, acausal_state_probabilities[time_slice])
    axes[1].legend(h, state_names)
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim((0.0, 1.05))

    n_states = len(state_names)
    axes[2].pcolormesh(
        t,
        x,
        conditional_non_local_acausal_posterior.T,
        vmin=0.0,
        vmax=posterior_max,
        cmap="bone_r",
    )
    axes[2].scatter(sliced_time, position[time_slice], s=1, color="magenta", zorder=2)
    axes[2].set_ylabel("Position [cm]")
    axes[3].fill_between(sliced_time, speed[time_slice], color="lightgrey", zorder=2)
    axes[3].set_ylabel("Speed [cm / s]")
    plt.xlim((sliced_time.min(), sliced_time.max()))
    plt.xlabel("Time [ms]")


def non_local_state_density(acausal_posterior, state_ind, non_local_state_ind=2):
    return acausal_posterior[:, state_ind == non_local_state_ind] / acausal_posterior[
        :, state_ind == non_local_state_ind
    ].sum(axis=1)


import matplotlib.colors as colors
import copy


def plot_likelihood_ratio(
    time_slice,
    log_likelihood,
    acausal_posterior,
    acausal_state_probabilities,
    non_local_rates,
    spikes,
    position,
    env,
    time,
    state_ind,
    state_names,
    figsize=(10, 10),
    posterior_max=0.25,
):
    likelihood = np.exp(log_likelihood[time_slice, state_ind == 2])
    spike_time_ind, neuron_ind = np.nonzero(spikes[time_slice, :])
    is_spike = np.zeros_like(time[time_slice], dtype=bool)
    is_spike[spike_time_ind] = True
    likelihood[~is_spike, :] = np.nan
    likelihood[:, ~env.is_track_interior_] = np.nan

    likelihood_ratio = np.exp(
        log_likelihood[time_slice, state_ind == 2]
        - log_likelihood[time_slice, state_ind == 0]
    )
    likelihood_ratio[:, ~env.is_track_interior_] = np.nan

    conditional_non_local_acausal_posterior = (
        acausal_posterior[time_slice, state_ind == 2]
        + acausal_posterior[time_slice, state_ind == 3]
    ) / (
        acausal_state_probabilities[time_slice, [2]]
        + acausal_state_probabilities[time_slice, [3]]
    )
    conditional_non_local_acausal_posterior[:, ~env.is_track_interior_] = np.nan

    neuron_place_bin = env.place_bin_centers_[
        np.nanargmax(non_local_rates, axis=0)
    ].squeeze()

    t, x = np.meshgrid(time[time_slice], env.place_bin_centers_)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 3, 1]},
    )

    cmap = copy.deepcopy(plt.get_cmap("RdBu_r"))
    cmap.set_bad(color="lightgrey")
    h = axes[0].pcolormesh(
        t, x, likelihood_ratio.T, norm=colors.LogNorm(vmin=1 / 10, vmax=10), cmap=cmap
    )
    plt.colorbar(h, ax=axes[0])
    axes[0].scatter(time[time_slice], position[time_slice], color="magenta", s=1)
    axes[0].scatter(
        time[time_slice][spike_time_ind],
        neuron_place_bin[neuron_ind],
        color="black",
        s=10,
    )

    cmap = copy.deepcopy(plt.get_cmap("bone_r"))
    cmap.set_bad(color="lightgrey")
    h = axes[1].pcolormesh(
        t,
        x,
        conditional_non_local_acausal_posterior.T,
        cmap=cmap,
        vmin=0.0,
        vmax=posterior_max,
    )
    plt.colorbar(h, ax=axes[1])
    axes[1].scatter(time[time_slice], position[time_slice], color="magenta", s=1)

    axes[2].plot(
        time[time_slice], acausal_state_probabilities[time_slice], label=state_names
    )
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Prob.")
    axes[2].legend()
