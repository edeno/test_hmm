import jax
import jax.numpy as jnp
import numpy as np
from patsy import DesignInfo, DesignMatrix, build_design_matrices
from scipy.optimize import minimize
from scipy.special import softmax

EPS = 1e-15


def get_transition_matrix(transition_matrix, t):
    return transition_matrix[t] if transition_matrix.ndim == 3 else transition_matrix


def forward(
    initial_conditions: np.ndarray,
    log_likelihood: np.ndarray,
    transition_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Causal algorithm for computing the posterior distribution of the hidden states of a switching model

    Parameters
    ----------
    initial_conditions : np.ndarray, shape (n_states,)
    log_likelihood : np.ndarray, shape (n_time, n_states)
    transition_matrix : np.ndarray, shape (n_states, n_states) or (n_time, n_states, n_states)

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
        predictive_distribution[t] = (
            get_transition_matrix(transition_matrix, t).T @ causal_posterior[t - 1]
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
    transition_matrix: np.ndarray,
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
            get_transition_matrix(transition_matrix, t) @ relative_distribution
        )
        acausal_posterior[t] /= acausal_posterior[t].sum()

    return acausal_posterior


def estimate_initial_conditions(acausal_posterior: np.ndarray) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    acausal_posterior : np.ndarray, shape (n_time, n_states)
        Acausal posterior distribution

    Returns
    -------
    initial_conditions : np.ndarray, shape (n_states,)
        Estimated initial conditions

    """
    return acausal_posterior[0]


def estimate_transition_matrix(
    causal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    acausal_posterior: np.ndarray,
) -> np.ndarray:
    """Estimate the transition matrix

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_states)
        Causal posterior distribution
    predictive_distribution : np.ndarray, shape (n_time, n_states)
        One step predictive distribution
    transition_matrix : np.ndarray, shape (n_states, n_states)
        Current estimate of the transition matrix
    acausal_posterior : np.ndarray, shape (n_time, n_states)
        Acausal posterior distribution
    Returns
    -------
    new_transition_matrix, np.ndarray, shape (n_states, n_states)
        New estimate of the transition matrix

    """
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


def estimate_parameters_via_em(
    initial_conditions: np.ndarray,
    log_likelihood: np.ndarray,
    transition_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Use the expectation maximization algorithm to estimate the parameters of a switching model

    Parameters
    ----------
    initial_conditions : np.ndarray, shape (n_states,)
    log_likelihood : np.ndarray, shape (n_time, n_states)
    transition_matrix : np.ndarray, shape (n_states, n_states)

    Returns
    -------
    initial_conditions : np.ndarray, shape (n_states,)
    transition_matrix : np.ndarray, shape (n_states, n_states)
    acausal_posterior : np.ndarray, shape (n_time, n_states)
    marginal_likelihood : float

    """
    # Expectation
    causal_posterior, predictive_distribution, marginal_likelihood = forward(
        initial_conditions, log_likelihood, transition_matrix
    )
    acausal_posterior = smoother(
        causal_posterior, predictive_distribution, transition_matrix
    )

    # Maximization
    initial_conditions = estimate_initial_conditions(acausal_posterior)
    transition_matrix = estimate_transition_matrix(
        causal_posterior,
        predictive_distribution,
        transition_matrix,
        acausal_posterior,
    )

    return initial_conditions, transition_matrix, acausal_posterior, marginal_likelihood


def viterbi(
    initial_conditions: np.ndarray,
    log_likelihood: np.ndarray,
    transition_matrix: np.ndarray,
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

    log_state_transition = np.log(np.clip(transition_matrix, a_min=EPS, a_max=1.0))
    log_initial_conditions = np.log(np.clip(initial_conditions, a_min=EPS, a_max=1.0))

    path_log_prob = np.ones_like(log_likelihood)
    back_pointer = np.zeros_like(log_likelihood, dtype=int)

    path_log_prob[0] = log_initial_conditions + log_likelihood[0]

    for time_ind in range(1, n_time):
        prior = path_log_prob[time_ind - 1] + log_state_transition
        for state_ind in range(n_states):
            back_pointer[time_ind, state_ind] = np.argmax(prior[state_ind])
            path_log_prob[time_ind, state_ind] = (
                prior[state_ind, back_pointer[time_ind, state_ind]]
                + log_likelihood[time_ind, state_ind]
            )

    # Find the best accumulated path prob in the last time bin
    # and then trace back the best path
    best_path = np.zeros((n_time,), dtype=int)
    best_path[-1] = np.argmax(path_log_prob[-1])
    for time_ind in range(n_time - 2, -1, -1):
        best_path[time_ind] = back_pointer[time_ind + 1, best_path[time_ind + 1]]

    return best_path, np.exp(np.max(path_log_prob[-1]))


def hmm_information_criterion(
    log_likelihood: float, n_states: int, n_independent_parameters: int, n_time: int = 1
) -> tuple[float, float]:
    """Calculate the information criterion for a hidden Markov model

    Parameters
    ----------
    log_likelihood : float
    n_states : int
    n_independent_parameters : int
    n_time : int, optional

    Returns
    -------
    aic : float
    bic : float

    """
    n_parameters = n_states**2 + n_independent_parameters * n_states - 1
    aic = -2 * log_likelihood + 2 * n_parameters
    bic = -2 * log_likelihood + n_parameters * np.log(n_time)

    return aic, bic


def centered_softmax_forward(y):
    """`softmax(x) = exp(x-c) / sum(exp(x-c))` where c is the last coordinate

    Example
    -------
    > y = np.log([2, 3, 4])
    > np.allclose(centered_softmax_forward(y), [0.2, 0.3, 0.4, 0.1])
    """
    if y.ndim == 1:
        y = np.append(y, 0)
    else:
        y = np.column_stack((y, np.zeros((y.shape[0],))))

    return softmax(y, axis=-1)


def centered_softmax_inverse(y):
    """`softmax(x) = exp(x-c) / sum(exp(x-c))` where c is the last coordinate

    Example
    -------
    > y = np.asarray([0.2, 0.3, 0.4, 0.1])
    > np.allclose(np.exp(centered_softmax_inverse(y)), np.asarray([2,3,4]))
    """
    return np.log(y[..., :-1]) - np.log(y[..., [-1]])


def make_spline_predict_matrix(
    design_info: DesignInfo, place_bin_centers: np.ndarray
) -> DesignMatrix:
    """Make a design matrix for position bins

    Parameters
    ----------
    design_info : DesignInfo
    place_bin_centers : np.ndarray, shape (n_place_bins,)
        The center of each place bin
    """
    predict_data = {}
    predict_data[f"x"] = place_bin_centers

    return build_design_matrices([design_info], predict_data)[0]


def fit_poisson_regression(design_matrix, weights, spikes, l2_penalty=0.0):
    @jax.jit
    def neglogp(
        coefficients, spikes=spikes, design_matrix=design_matrix, weights=weights
    ):
        conditional_intensity = jnp.exp(design_matrix @ coefficients)
        conditional_intensity = jnp.clip(conditional_intensity, a_min=EPS, a_max=None)
        negative_log_likelihood = -1.0 * jnp.mean(
            weights * jax.scipy.stats.poisson.logpmf(spikes, conditional_intensity)
        )
        l2_penalty_term = l2_penalty * jnp.sum(coefficients[1:] ** 2)
        return negative_log_likelihood + l2_penalty_term

    dlike = jax.grad(neglogp)

    initial_condition = np.array([np.log(np.average(spikes, weights=weights))])
    initial_condition = np.concatenate(
        [initial_condition, np.zeros(design_matrix.shape[1] - 1)]
    )

    res = minimize(
        neglogp,
        x0=initial_condition,
        method="BFGS",
        jac=dlike,
    )

    return res.x


def check_converged(
    loglik: float, previous_loglik: float, tolerance: float = 1e-4
) -> tuple[bool, bool]:
    """
    We have converged if the slope of the log-likelihood function falls below 'threshold',
    i.e., |f(t) - f(t-1)| / avg < threshold,
    where avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log lik at iteration t.
    'threshold' defaults to 1e-4.

    This stopping criterion is from Numerical Recipes in C p423
    """
    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + np.spacing(1)) / 2

    is_increasing = loglik - previous_loglik >= -1e-3
    is_converged = (delta_loglik / avg_loglik) < tolerance

    return is_converged, is_increasing


@jax.jit
def jax_forward(
    initial_conditions: jnp.ndarray,
    log_likelihood: jnp.ndarray,
    transition_matrix: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Causal algorithm for computing the posterior distribution of the hidden states of a switching model

    Parameters
    ----------
    initial_conditions : np.ndarray, shape (n_states,)
    log_likelihood : np.ndarray, shape (n_time, n_states)
    transition_matrix : np.ndarray, shape (n_states, n_states) or (n_time, n_states, n_states)

    Returns
    -------
    causal_posterior : np.ndarray, shape (n_time, n_states)
        Causal posterior distribution
    predictive_distribution : np.ndarray, shape (n_time, n_states)
        One step predictive distribution
    marginal_likelihood : float

    """
    n_time = log_likelihood.shape[0]

    predictive_distribution = jnp.zeros_like(log_likelihood)
    causal_posterior = jnp.zeros_like(log_likelihood)
    max_log_likelihood = jnp.max(log_likelihood, axis=1, keepdims=True)
    likelihood = jnp.exp(log_likelihood - max_log_likelihood)
    likelihood = jnp.clip(likelihood, a_min=1e-15, a_max=1.0)

    predictive_distribution.at[0].set(initial_conditions)
    causal_posterior.at[0].set(initial_conditions * likelihood[0])
    norm = jnp.nansum(causal_posterior[0])
    marginal_likelihood = jnp.log(norm)
    causal_posterior.at[0].set(causal_posterior[0] / norm)

    for t in range(1, n_time):
        # Predict
        predictive_distribution.at[t].set(
            get_transition_matrix(transition_matrix, t).T @ causal_posterior[t - 1]
        )
        # Update
        causal_posterior.at[t].set(predictive_distribution[t] * likelihood[t])
        # Normalize
        norm = jnp.nansum(causal_posterior[t])
        marginal_likelihood += jnp.log(norm)
        causal_posterior.at[t].set(norm)

    marginal_likelihood += jnp.sum(max_log_likelihood)

    return causal_posterior, predictive_distribution, marginal_likelihood
