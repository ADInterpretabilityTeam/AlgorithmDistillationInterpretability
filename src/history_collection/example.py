''' example for collecting some histories from a given environment  '''
import numpy as np
np.random.seed(42)
from stable_baselines3 import A2C

from history_generation import get_learning_histories
from bandits import alternating_bandit 

n_arms = 10
train_tasks = [alternating_bandit(n_arms, odd=True) for i in range(10)]  
test_tasks = [alternating_bandit(n_arms, odd=False) for i in range(10)]  

model_init = lambda env : A2C('MlpPolicy', env, verbose=1)
histories = get_learning_histories(model_init, train_tasks)


# %%

# let's train a transformer on these sequences now 

