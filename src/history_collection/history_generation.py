''' utilities for collecting histories from the training run of a sb3 algorithm '''
from stable_baselines3.common.callbacks import BaseCallback
from typing import List, Dict
import random
import gym 


class HistoryDataset:
	''' Container for training histories for multiple tasks 
		very basic for now, intended to be used as dataset soon '''
	def __init__(self) -> None: 
		self.histories = []  # container of task histories 

	def append(self, task_history: List):
		assert isinstance(task_history, list)
		self.histories.append(task_history)

	def __getitem__(self, i):
		return self.histories[i]

	def sample(self, l, cross_task=False) -> List[Dict]:
		if cross_task:
			raise NotImplemented
		task_hist = random.choice(self.histories) 
		s = random.randint(0, len(task_hist)-l)
		return task_hist[s:s+l]
		
class HistoryCallback(BaseCallback):
	''' callback for extracting keys from local variables during sb3 model training '''
	def __init__(self, collect_keys=['obs_tensor', 'actions', 'rewards', 'dones'], verbose: int = 0):
		super().__init__(verbose)
		self.collect_keys = collect_keys
		self.history = []

	def _on_step(self) -> bool:
		self.history.append({k: self.locals[k] for k in self.collect_keys})
		return True 

	
def get_learning_histories(model_init, tasks: List[gym.Env]) -> HistoryDataset:
	''' trains model specified by model_init on tasks provided, 
	collects histories in a HistoryDataset object '''
	learning_histories = HistoryDataset()
	for i, env in enumerate(tasks):
		print('solving task', i) 
		env.reset()
		model = model_init(env)
		cb = HistoryCallback() 
		model.learn(total_timesteps=1000, callback=cb) 
		learning_histories.append(cb.history)
		# what does this look like when the episodes are longer than 1? 
		# presumably cb.history is just multiepisode 
	return learning_histories



