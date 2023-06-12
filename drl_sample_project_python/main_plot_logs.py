from drl_lib.to_do.logs_analysis import plot_reward_lengh

monte_carlo_es = 'monte_carlo_es_logs.json'
on_policy_first_visit_monte_carlo_control = 'on_policy_first_visit_monte_carlo_control_logs.json'
off_policy_monte_carlo_control = 'off_policy_monte_carlo_control_logs.json'
sarsa = 'sarsa_logs.json'
q_learning = 'q_learning_logs.json'
expected_sarsa = 'expected_sarsa_logs.json'
# list_of_logs = [monte_carlo_es, on_policy_first_visit_monte_carlo_control, off_policy_monte_carlo_control]
list_of_logs = [sarsa]
for logs in list_of_logs:
    plot_reward_lengh("logs", logs, 500)