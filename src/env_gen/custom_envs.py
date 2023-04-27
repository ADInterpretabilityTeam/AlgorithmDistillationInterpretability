import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gymnasium as gym
from .utils import *
import numpy as np
import networkx as nx

 
def sample_transition_rules(n_states, n_actions, rng, mcmc=True):
    # Generates matrix of the form [n_states, n_actions, n_states]
    # Start by created a strongly-connected directed graph (a loop)
    T = np.zeros((n_states, n_actions, n_states,))
    adj_matrix = np.eye(n_states)
    adj_matrix = np.concatenate((adj_matrix[1:, :], adj_matrix[:1, :]), axis=0)
    adj_matrix = permute_graph(adj_matrix, rng)
    T[:, 0, :] = adj_matrix
    # For the other actions, find a random state that we can go to 
    for a in range(1, n_actions):
        for s in range(n_states):
            # Create a random edge
            sp = rng.integers(0, n_states)
            T[s, a, sp] = 1.0
            # 
    # Use MCMC to get a random SC digraph (hopefully close to a uniform sample)
    if mcmc:
        for i in range(100000):  # Not possible to calculate in general what the mixing time is here
            rand_prev_state = rng.integers(0, n_states)
            rand_next_state = rng.integers(0, n_states)
            rand_act = rng.integers(0, n_actions)
            # Zero out the current transition, and set it to be random
            curr_transition = np.argmax(T[rand_prev_state, rand_act, :])
            T[rand_prev_state, rand_act, curr_transition] = 0
            T[rand_prev_state, rand_act, rand_next_state] = 1
            # Check if still connected
            visited = np.zeros(n_states, dtype=bool)
            dfs(T.sum(1), rand_prev_state, visited)
            is_connected = visited[curr_transition]
            # If no longer connected, then revert
            if not is_connected:
                T[rand_prev_state, rand_act, curr_transition] = 1
                T[rand_prev_state, rand_act, rand_next_state] = 0
    # We can also add some stochasticity to make the environment a bit harder
    # Normalize the probabilities
    T = T / np.sum(T, axis=2)[:, :, None]
    return T


def sample_observation_rules(n_states):
    # How should we sample observations
    obs = np.eye(n_states)
    # Optionally, we can rotate the observations using a random orthonormal matrix to make it harder for the policies
    return obs


def sample_reward_rules(n_states, n_actions, n_rewards, rng, prob_use_old_state=1.0, prob_use_new_state=0.0, prob_use_action=0.33, prob_use_flag=0.33):
    # Rules are of the form (old_state, new_state, action, probability, value, flag)
    rules = []
    for nr in range(n_rewards):
        rule = [-1, -1, -1, 0, 0, -1]
        # Repeat until the rule is conditional on either the old state, new_state, or action
        while rule[0] == -1 and rule[1] == -1 and rule[2] == -1:
            if rng.random() < prob_use_old_state:
                rule[0] = rng.integers(0, n_states)
            if rng.random() < prob_use_new_state:
                rule[1] = rng.integers(0, n_states)
            if rng.random() < prob_use_action:
                rule[2] = rng.integers(0, n_actions)
        # Decide whether or not to make reward conditional on hidden variable
        if rng.random() < prob_use_flag:
            rule[5] = 1.0  # rng.choice([1.0, 1.0, 1.0, 0.0])
        # Reward is always 1, but probability of reward can change
        rule[3] = 1.0  # rng.choice([.2, .8, 1.0, 1.0])
        rule[4] = 1.0
        rules.append(rule)
    # Make sure that two rules with the same old_state don't have the same action
    # or two rules with the same action don't have the same old state
    for i in range(len(rules)):
        for j in range(i+1, len(rules)):
            if rules[i][0]  == rules[j][0]:  # If rules share the same old state
                if rules[i][2] == -1 or rules[i][2] == rules[j][2]:  # If there are identical rules, re-run method
                    return sample_reward_rules(n_states, n_actions, n_rewards, rng, prob_use_old_state, prob_use_new_state, prob_use_action, prob_use_flag)
    return rules


def sample_flag_rules(n_states, n_actions, n_flag_rules, rng, prob_use_old_state=1.0, prob_use_new_state=0.0, prob_use_action=0.33):
    # Rules are of the form (old_state, new_state, action, new flag value)
    flagrules = []
    for nr in range(n_flag_rules):
        flagrule = [-1, -1, -1, 1.0]
        while flagrule[0] == -1 and flagrule[1] == -1 and flagrule[2] == -1:
            if rng.random() < prob_use_old_state:
                flagrule[0] = rng.integers(0, n_states)
                # Check if there exists a sta
            if rng.random() < prob_use_new_state:
                flagrule[1] = rng.integers(0, n_states)
            if rng.random() < prob_use_action:
                flagrule[2] = rng.integers(0, n_actions)
        flagrule[3] = 1  # np.random.choice([1, 1, 1, 0])
        flagrules.append(flagrule)
    return flagrules


class GeneralTask(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, n_states, n_actions, n_rewards, n_flags, max_steps, render_mode="rgb_array", seed=None, prob_use_action=0.33, prob_use_flag=0.33):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_rewards = n_rewards
        self.n_flags = n_flags
        self.max_steps = max_steps
        # create rng
        if seed is None:
            seed = np.random.randint(0, 2000)
        self.rng = np.random.default_rng(seed=seed)
        # Sample environment properties
        self.transition = sample_transition_rules(n_states, n_actions, self.rng)
        self.observations = sample_observation_rules(n_states)
        self.reward_rules = sample_reward_rules(n_states, n_actions, n_rewards, self.rng, prob_use_action=prob_use_action, prob_use_flag=prob_use_flag)
        # Sample flag rules such that they have no overlap with the reward_rules
        # There must be no two flag and reward rules that share the same old state
        while True:        
            self.flag_rules = sample_flag_rules(n_states, n_actions, n_flags, self.rng, prob_use_action=prob_use_action)
            # Resample flag rules if there is a flag rule with the same old state as a reward rule
            repeat = False
            for reward_rule in self.reward_rules:
                for flag_rule in self.flag_rules:
                    if reward_rule[0] == flag_rule[0]:
                        repeat = True
            if not repeat:
                break
        # Set environment options
        self.observation_space = gym.spaces.Box(0, 1.0, (n_states,))
        self.action_space = gym.spaces.Discrete(n_actions)
        # Set environment render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return self.observations[self.current_state]

    @property
    def steps_remaining(self):
        return self.max_steps - self.current_step

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.current_state = np.random.randint(0, self.n_states) # Choose new starting position
        self.current_flag = 0  # Reset flag
        self.current_step = 0

        return self._get_obs(), {}

    def step(self, action):
        # Sample the next state from the transition matrix
        t_probs = self.transition[self.current_state, action]
        next_state = np.random.choice(self.n_states, 1, p=t_probs)[0]  # Sample s' from categorical dist T[s, a]
        # Check to update flag
        for flag_old_state, flag_next_state, flag_action, flag_value in self.flag_rules[::-1]:
            # Check if the rule meets the desired criteria
            if flag_old_state == -1 or flag_old_state == self.current_state:
                if flag_next_state == -1 or flag_next_state == next_state:
                    if flag_action == -1 or flag_action == action:
                        # Set the new flag value and break
                        # Newer rules override older ones so we break when we meet the first rule that applies
                        self.current_flag = flag_value
                        break
        # Calculate rewards
        reward = 0
        for reward_old_state, reward_next_state, reward_action, reward_prob, reward_value, reward_flag in self.reward_rules[::-1]:
            if reward_old_state == -1 or reward_old_state == self.current_state:
                if reward_next_state == -1 or reward_next_state == next_state:
                    if reward_action == -1 or reward_action == action:
                        if reward_flag == -1 or reward_flag == self.current_flag:
                            if np.random.random() < reward_prob:
                                reward = reward_value
        # Calculate remaining timesteps and if done
        self.current_state = next_state
        self.current_step += 1
        done = self.current_step == self.max_steps
        obs = self._get_obs()
        # returns obs, reward, terminated, truncated, info
        return obs, reward, done, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # Draw transition matrix as a graph
        adj_matrix = self.transition.sum(1) != 0
        graph = nx.DiGraph(incoming_graph_data=adj_matrix)
        # Label current state, reward states, and flag states
        color_map = ["lightblue" for _ in range(self.n_states)]
        for reward_rule in self.reward_rules:
            reward_node = reward_rule[0]
            if reward_node != -1:
                color_map[reward_node] = "lightgreen"
        for flag_rule in self.flag_rules:
            flag_node = flag_rule[0]
            if flag_node != -1:
                color_map[flag_node] = "salmon"
        edge_colors = [0 for i in range(self.n_states)]
        edge_colors[self.current_state] = 5
        # Draw image and return image array
        fig, ax = plt.subplots(figsize=(6, 6))
        nx.draw_circular(graph, ax=ax, with_labels=True, node_color=color_map, font_weight='bold', edgecolors="darkgoldenrod", linewidths=edge_colors, node_size=800)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


class MultiArmedBandit(GeneralTask):

    def __init__(self, n_arms, n_steps, seed=None):
        super().__init__(1, n_arms, n_arms, 0, n_steps, prob_use_action=1.0, prob_use_flag=0.0, seed=seed)
        # We need to reinit probs for the rewards
        for reward_rule in self.reward_rules:   
            reward_rule[3] = self.rng.random()


class NavigationBandit(GeneralTask):

    def __init__(self, n_states, n_steps, seed=None):
        super().__init__(n_states, 2, n_states, 0, n_steps, prob_use_action=0.0, prob_use_flag=0.0, seed=seed)
        # We need to reinit probs for the rewards
        for reward_rule in self.reward_rules:
            reward_rule[3] = self.rng.random()
        # We need to re-init transition


class DarkRoom(GeneralTask):
    
    def __init__(self, n_states, n_actions, n_steps, seed=None):
        # Never has a flag in the reward rule
        super().__init__(n_states, n_actions, 1, 0, n_steps, prob_use_flag=0.0, seed=seed)


class DarkKeyDoor(GeneralTask):
    
    def __init__(self, n_states, n_actions, n_steps, seed=None):
        # Always has a flag in the reward rule
        super().__init__(n_states, n_actions, 1, 1, n_steps, prob_use_flag=1.0, seed=seed)


class RandomTask(GeneralTask):
    
    def __init__(self, n_states, n_actions, n_steps, seed=None):
        # 50% Dark Room, 50% Dark Key Door
        super().__init__(n_states, n_actions, 1, 1, n_steps, prob_use_flag=0.5, seed=seed)