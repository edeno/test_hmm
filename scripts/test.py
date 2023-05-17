from src.test_non_local_model import (
    fit_switching_model,
    load_data,
    setup_nonlocal_switching_model,
)

is_ripple, spikes, position, env, time = load_data(work_computer=False)


(
    design_matrix_switching,
    predict_matrix_switching,
    initial_conditions_switching,
    discrete_state_transitions_switching,
    continuous_state_transitions_switching,
    state_ind_switching,
    zero_rates_switching,
    is_training_switching,
    state_names_switching,
) = setup_nonlocal_switching_model(
    is_ripple,
    spikes,
    position,
    env,
)


(
    predicted_state_switching,
    acausal_posterior_switching,
    acausal_state_probabilities_switching,
    causal_posterior_switching,
    marginal_log_likelihoods_switching,
    initial_conditions_switching,
    discrete_state_transitions_switching,
    coefficients_iter,
    local_rates_iter,
    non_local_rates_iter,
    is_training_iter,
    acausal_posterior_iter,
) = fit_switching_model(
    spikes,
    design_matrix_switching,
    predict_matrix_switching,
    initial_conditions_switching,
    discrete_state_transitions_switching,
    continuous_state_transitions_switching,
    state_ind_switching,
    zero_rates_switching,
    is_training_switching,
)
