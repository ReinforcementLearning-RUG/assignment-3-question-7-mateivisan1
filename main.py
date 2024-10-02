from rl_mdp.model_free_prediction.monte_carlo_evaluator import MCEvaluator
from rl_mdp.model_free_prediction.td_evaluator import TDEvaluator
from rl_mdp.model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator
from rl_mdp.util import create_mdp, create_policy_1, create_policy_2


def main() -> None:
    mdp = create_mdp()

    policy_1 = create_policy_1()
    policy_2 = create_policy_2()

    mc_evaluator = MCEvaluator(env=mdp)
    # policy 1
    print("MCEvaluator --> Evaluating Policy 1...")
    value_function_mc_1 = mc_evaluator.evaluate(policy_1, num_episodes=1000)
    print("Value function for Policy 1:")
    print(value_function_mc_1)
    # policy 2
    print("\nMCEvaluator --> Evaluating Policy 2...")
    value_function_mc_2 = mc_evaluator.evaluate(policy_2, num_episodes=1000)
    print("Value function for Policy 2:")
    print(value_function_mc_2)
    
    print("-----------------------------------------------")

    
    td_evaluator = TDEvaluator(env=mdp, alpha=0.1)
    # policy 1
    print("\nTD0Evaluator --> Evaluating Policy 1...")
    value_function_td0_1 = td_evaluator.evaluate(policy_1, num_episodes=1000)
    print("Value function for Policy 1:")
    print(value_function_td0_1)
    # policy 2
    print("\nTD0Evaluator --> Evaluating Policy 2...")
    value_function_td_2 = td_evaluator.evaluate(policy_2, num_episodes=1000)
    print("Value function for Policy 2:")
    print(value_function_td_2)
    
    print("-----------------------------------------------")
    
    td_evaluator_lamda = TDLambdaEvaluator(env=mdp, alpha=0.1, lambd=0.5)
    # policy 1
    print("\nTDlamdEvaluator --> Evaluating Policy 1...")
    value_function_tdlamd_1 = td_evaluator_lamda.evaluate(policy_1, num_episodes=1000)
    print("Value function for Policy 1:")
    print(value_function_tdlamd_1)
    # policy 2
    print("\nTDlamdEvaluator --> Evaluating Policy 2...")
    value_function_tdlamd_2 = td_evaluator_lamda.evaluate(policy_2, num_episodes=1000)
    print("Value function for Policy 2:")
    print(value_function_tdlamd_2)
    
    print("-----------------------------------------------")    

if __name__ == "__main__":
    main()
