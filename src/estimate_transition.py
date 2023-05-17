import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import log_softmax
from jax.scipy.special import gammaln
from scipy.optimize import minimize


def estimate_joint_distribution(
    causal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    acausal_posterior: np.ndarray,
) -> np.ndarray:
    """Estimate the joint_distribution of latents given the observations

    p(x_t, x_{t+1} | O_{1:T})

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_states)
        Causal posterior distribution
    predictive_distribution : np.ndarray, shape (n_time, n_states)
        One step predictive distribution
    transition_matrix : np.ndarray, shape (n_time, n_states, n_states) or shape (n_states, n_states)
        Current estimate of the transition matrix
    acausal_posterior : np.ndarray, shape (n_time, n_states)
        Acausal posterior distribution

    Returns
    -------
    joint_distribution : np.ndarray, shape (n_time - 1, n_states, n_states)

    """
    relative_distribution = np.where(
        np.isclose(predictive_distribution[1:], 0.0),
        0.0,
        acausal_posterior[1:] / predictive_distribution[1:],
    )[:, np.newaxis]

    if transition_matrix.ndim == 2:
        # Add a singleton dimension for the time axis if the transition matrix is stationary
        joint_distribution = (
            transition_matrix[np.newaxis]
            * causal_posterior[:-1, :, np.newaxis]
            * relative_distribution
        )
    else:
        joint_distribution = (
            transition_matrix[:-1]
            * causal_posterior[:-1, :, np.newaxis]
            * relative_distribution
        )
        # joint_distribution = (
        #         np.einsum("ts,st,t->tsst", transition_matrix[:-1], causal_posterior[:-1], relative_distribution)
        #     )

    return joint_distribution


@jax.jit
def jax_centered_log_softmax_forward(y: jnp.ndarray) -> jnp.ndarray:
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

    return log_softmax(y, axis=-1)


@jax.jit
def multinomial_neg_log_likelihood(
    coefficients: jnp.ndarray,
    design_matrix: jnp.ndarray,
    response: jnp.ndarray,
    l2_penalty: float = 1e-10,
) -> float:
    """Negative expected complete log likelihood of the transition model.

    Parameters
    ----------
    coefficients : jnp.ndarray, shape (n_coefficients * n_states - 1)
    design_matrix : jnp.ndarray, shape (n_samples, n_coefficients)
    response : jnp.ndarray, shape (n_samples, n_states)

    Returns
    -------
    negative_expected_complete_log_likelihood : float

    """
    # Reshape flattened coefficients to shape (n_coefficients, n_states - 1)
    n_coefficients = design_matrix.shape[1]
    coefficients = coefficients.reshape((n_coefficients, -1))

    # The last state probability can be inferred from the other state probabilities
    # since the probabilities must sum to 1
    # shape (n_samples, n_states)
    log_probs = jax_centered_log_softmax_forward(design_matrix @ coefficients)

    # Average cross entropy over samples
    # Average is used instead of sum to make the negative log likelihood
    # invariant to the number of samples
    n_samples = response.shape[0]
    neg_log_likelihood = -1.0 * jnp.sum(response * log_probs) / n_samples

    # Penalize the size (squared magnitude) of the coefficients
    # Don't penalize the intercept for identifiability
    l2_penalty_term = l2_penalty * jnp.sum(coefficients[1:] ** 2)

    return neg_log_likelihood + l2_penalty_term


multinomial_gradient = jax.grad(multinomial_neg_log_likelihood)
multinomial_hessian = jax.hessian(multinomial_neg_log_likelihood)


def get_transition_prior(concentration: float, stickiness: float, n_states: int):
    return concentration * np.ones((n_states,)) + stickiness * np.eye(n_states)


def estimate_non_stationary_state_transition(
    transition_coefficients: np.ndarray,
    design_matrix: np.ndarray,
    causal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    acausal_posterior: np.ndarray,
    concentration: float = 1.0,
    stickiness: float = 0.0,
    transition_regularization: float = 1e-5,
    optimization_method: str = "Newton-CG",
    maxiter: None | int = 100,
    disp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the non-stationary state transition model.

    Parameters
    ----------
    transition_coefficients : np.ndarray, shape (n_coefficients, n_states, n_states - 1)
        Initial estimate of the transition coefficients
    design_matrix : np.ndarray, shape (n_time, n_coefficients)
    causal_posterior : np.ndarray, shape (n_time, n_states)
        Causal posterior distribution
    predictive_distribution : np.ndarray, shape (n_time, n_states)
        One step predictive distribution
    transition_matrix : np.ndarray, shape (n_time, n_states, n_states)
        Current estimate of the transition matrix
    acausal_posterior : np.ndarray, shape (n_time, n_states)
        Acausal posterior distribution
    optimization_method : str, optional
        See scipy.optimize.minimize for available methods
    maxiter : int, optional
        Maximum number of iterations for the optimization
    disp : bool, optional
        Whether to print convergence messages for the optimization

    Returns
    -------
    estimated_transition_coefficients : jnp.ndarray, shape (n_coefficients, n_states, n_states - 1)
    estimated_transition_matrix : jnp.ndarray, shape (n_time, n_states, n_states)

    """
    # p(x_t, x_{t+1} | O_{1:T})
    joint_distribution = estimate_joint_distribution(
        causal_posterior,
        predictive_distribution,
        transition_matrix,
        acausal_posterior,
    )

    n_coefficients, n_states = transition_coefficients.shape[:2]
    estimated_transition_coefficients = np.zeros(
        (n_coefficients, n_states, (n_states - 1))
    )

    n_time = design_matrix.shape[0]
    estimated_transition_matrix = np.zeros((n_time, n_states, n_states))

    alpha = get_transition_prior(concentration, stickiness, n_states)

    # Estimate the transition coefficients for each state
    for from_state, row_alpha in enumerate(alpha):

        result = minimize(
            dirichlet_neg_log_likelihood,
            x0=transition_coefficients[:, from_state].ravel(),
            method=optimization_method,
            jac=dirichlet_gradient,
            hess=dirichlet_hessian,
            args=(
                design_matrix[:-1],
                joint_distribution[:, from_state, :],
                row_alpha,
                transition_regularization,
            ),
            options={"disp": disp, "maxiter": maxiter},
        )

        estimated_transition_coefficients[:, from_state, :] = result.x.reshape(
            (n_coefficients, n_states - 1)
        )

        linear_predictor = (
            design_matrix @ estimated_transition_coefficients[:, from_state, :]
        )
        estimated_transition_matrix[:, from_state, :] = jnp.exp(
            jax_centered_log_softmax_forward(linear_predictor)
        )

    # # if any is zero, set to small number
    # estimated_transition_matrix = np.clip(
    #     estimated_transition_matrix, 1e-16, 1.0 - 1e-16
    # )

    return estimated_transition_coefficients, estimated_transition_matrix


def estimate_stationary_state_transition(
    causal_posterior: np.ndarray,
    predictive_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    acausal_posterior: np.ndarray,
    stickiness: float = 0.0,
    concentration: float = 1.0,
):
    # p(x_t, x_{t+1} | O_{1:T})
    joint_distribution = estimate_joint_distribution(
        causal_posterior,
        predictive_distribution,
        transition_matrix,
        acausal_posterior,
    )

    # Dirichlet prior for transition probabilities
    n_states = acausal_posterior.shape[1]
    alpha = get_transition_prior(concentration, stickiness, n_states)

    # alpha needs to be >= 1.0 for the new transition matrix to be greater than 0.0
    assert np.all(alpha >= 1.0)

    new_transition_matrix = joint_distribution.sum(axis=0) + alpha - 1.0
    new_transition_matrix /= new_transition_matrix.sum(axis=-1, keepdims=True)

    # if any is zero, set to small number
    # new_transition_matrix = np.clip(
    #     new_transition_matrix, a_min=1e-16, a_max=1.0 - 1e-16
    # )

    return new_transition_matrix


@jax.jit
def dirichlet_neg_log_likelihood(
    coefficients: jnp.ndarray,
    design_matrix: jnp.ndarray,
    response: jnp.ndarray,
    alpha=1.0,
    l2_penalty: float = 1e-5,
) -> float:
    """Negative expected complete log likelihood of the transition model.

    Parameters
    ----------
    coefficients : jnp.ndarray, shape (n_coefficients * (n_states - 1))
    design_matrix : jnp.ndarray, shape (n_samples, n_coefficients)
    response : jnp.ndarray, shape (n_samples, n_states)
    alpha : jnp.ndarray, shape (n_states,)

    Returns
    -------
    negative_expected_complete_log_likelihood : float

    """
    n_coefficients = design_matrix.shape[1]

    # shape (n_coefficients, n_states - 1)
    coefficients = coefficients.reshape((n_coefficients, -1))

    # shape (n_samples, n_states)
    log_probs = jax_centered_log_softmax_forward(design_matrix @ coefficients)

    # Dirichlet prior
    n_samples = response.shape[0]
    prior = (alpha - 1.0) / n_samples

    neg_log_likelihood = -1.0 * jnp.sum((response + prior) * log_probs) / n_samples
    l2_penalty_term = l2_penalty * jnp.sum(coefficients[1:] ** 2)

    return neg_log_likelihood + l2_penalty_term


dirichlet_gradient = jax.grad(dirichlet_neg_log_likelihood)
dirichlet_hessian = jax.hessian(dirichlet_neg_log_likelihood)