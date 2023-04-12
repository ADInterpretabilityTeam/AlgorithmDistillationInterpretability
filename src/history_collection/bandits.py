import numpy as np
np.random.seed(42)
import random 
from gym_bandits.bandit import BanditEnv

# I am not sure what the paper means with the distribution of the reward 
# quoting: " Reward is more likely distributed under odd arms 95% of the time during training."
def alternating_bandit(n_arms, odd, p_ratio=1-0.95):
	''' returns a bandit with alternating payout probabilities depending on the arm 
		if odd then the odd arms will have a higher payout probability, with p 1-p_ratio
	'''
	delta = (-p_ratio - np.sqrt(2*p_ratio)) / (2*p_ratio -4)
	alternating_pdist = lambda n, odd=True : [
		random.uniform(0, 0.5+delta) if i%2==int(odd) else random.uniform(0.5-delta, 1) for i in range(n)
	]
	return BanditEnv(p_dist=alternating_pdist(n_arms, odd=odd), r_dist=[1]*n_arms)

