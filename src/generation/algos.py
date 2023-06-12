import numpy as np
import heapq


def value_iteration(env, discount, theta=1e-5):
    # Convert the reward rules into a reward matrix
    R = np.zeros_like(env.transition)
    for (
        reward_old_state,
        reward_next_state,
        reward_action,
        reward_prob,
        reward_value,
        reward_flag,
    ) in env.reward_rules:
        assert reward_flag == -1  # Make sure the env is fully observable
        if reward_old_state == -1:
            reward_old_state = slice(None)
        if reward_next_state == -1:
            reward_next_state = slice(None)
        if reward_action == -1:
            reward_action = slice(None)
        R[reward_old_state, reward_action, reward_next_state] = reward_prob * reward_value
    # Perform value iteration
    values = np.zeros(env.n_states)
    while True:
        values_old = np.copy(values)
        for s in range(env.n_states):
            Q = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                for s_prime in range(env.n_states):
                    p = env.transition[s, a, s_prime]
                    r = R[s, a, s_prime]
                    Q[a] += p * (r + discount * values_old[s_prime])
            values[s] = np.max(Q)
        delta_V = np.max(np.abs(values - values_old))
        if delta_V < theta:
            break
    # Calculate optimal policy
    policy = np.zeros(env.n_states)
    for s in range(env.n_states):
        Q = np.zeros(env.n_actions)
        for a in range(env.n_actions):
            for s_prime in range(env.n_states):
                p = env.transition[s, a, s_prime]
                r = R[s, a, s_prime]
                Q[a] += p * (r + discount * values_old[s_prime])
        policy[s] = np.argmax(Q)
    return values, policy


def q_learning(env, n_episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon):
    Q = np.ones((env.n_states, env.n_actions))
    reward_over_time = []
    # Iterate until we have enough epsiodes
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        state = np.argmax(obs)
        done = False
        total_reward = 0
        
        while not done:
            # Choose action using epsilon greedy strategy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
            # Execute action update Q table
            obs, reward, done, _, _ = env.step(action)
            new_state = np.argmax(obs)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            total_reward += reward
            state = new_state
        
        epsilon = max(min_alpha, epsilon_decay * epsilon)
        reward_over_time.append(total_reward)

    return np.array(reward_over_time)


def sarsa_learning(env, n_episodes, alpha, gamma, lambda_, epsilon, epsilon_decay, epsilon_min):
    Q = np.ones((env.n_states, env.n_actions))
    
    reward_over_time = []
    # Iterate until we have enough epsiodes
    for episode in range(n_episodes):
        e_traces = np.zeros_like(Q)
        
        # Get initial state
        obs, _ = env.reset(seed=episode)
        state = np.argmax(obs)
        
        # Get first action
        if np.random.rand() < epsilon:
            action = env.action_space.sample() # Explore
        else:
            action = np.argmax(Q[state, :]) # Exploit
        
        done = False
        total_reward = 0
        
        while not done:

            obs, reward, done, _, _ = env.step(action)
            new_state = np.argmax(obs)
            
            if np.random.rand() < epsilon:
                new_action = env.action_space.sample() # Explore
            else:
                new_action = np.argmax(Q[new_state, :]) # Exploit
            
            target = reward + gamma * Q[new_state, new_action] * (1 - done)
            td_error = target - Q[state, action]
            e_traces[state, action] += 1
            
            Q += alpha * td_error * e_traces
            e_traces *= gamma * lambda_
            
            state = new_state
            action = new_action
            total_reward += reward
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        reward_over_time.append(total_reward)

    return np.array(reward_over_time)


def dynaq_learning(env, n_episodes, alpha, gamma, epsilon, epsilon_decay, epsilon_min, n_planning_steps):
    n_states = env.n_states
    n_actions = env.n_actions
    q_values = 12 * np.ones((n_states, n_actions)) # Initializing to one encourages much faster exploration
    model = {}
    ep_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        state = np.argmax(obs)
        done = False
        
        total_reward = 0

        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_values[state])
            
            obs, reward, done, _, _ = env.step(action)
            next_state = np.argmax(obs)
            
            q_values[state, action] += alpha * (reward + gamma * np.max(q_values[next_state]) - q_values[state, action])

            # Update the model
            if state not in model:
                model[state] = {}
            model[state][action] = (reward, next_state)

            # Planning loop
            for _ in range(n_planning_steps):
                random_state = np.random.choice(list(model.keys()))
                random_action = np.random.choice(list(model[random_state].keys()))
                random_reward, random_next_state = model[random_state][random_action] 
                q_values[random_state, random_action] += alpha * (random_reward + gamma * np.max(q_values[random_next_state]) - q_values[random_state, random_action])

            state = next_state
            total_reward += reward

        ep_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    return np.array(ep_rewards)


def prioritized_sweeping(env, n_episodes, alpha, gamma, epsilon, epsilon_decay, epsilon_min, theta, n_planning_steps):
    n_states = env.n_states
    n_actions = env.n_actions
    q_values =  12  * np.ones((n_states, n_actions)) # Initializing to one encourages much faster exploration
    model = {}
    ep_rewards = []
    priority_queue = []
    seen_state_action_pairs = set()
    
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        state = np.argmax(obs)
        done = False        
        total_reward = 0

        while not done:
            # Compute next environment step
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_values[state])
            seen_state_action_pairs.add((state, action))
            obs, reward, done , _, _ = env.step(action)
            next_state = np.argmax(obs)
            # Update the model
            if state not in model:
                model[state] = {}
            model[state][action] = (reward, next_state)
            # Calculate priority and update queue
            max_q_next = np.max(q_values[next_state]) if not done else 0
            priority = np.abs(reward + gamma * max_q_next - q_values[state, action])
            if priority > theta:
                heapq.heappush(priority_queue, (-priority, state, action))
            # Planning loop
            for i in range(n_planning_steps):
                if len(priority_queue) == 0:
                    break

                _, s, a = heapq.heappop(priority_queue)
                r, ns = model[s][a]

                q_values[s, a] += alpha * (r + gamma * np.max(q_values[ns]) - q_values[s][a])

                for s_prime, a_prime in seen_state_action_pairs:
                    r_prime, ns_prime = model[s_prime][a_prime]
                    if ns_prime != s:
                        continue
                    
                    max_q_ns_prime = np.max(q_values[ns_prime])
                    priority_prime = np.abs(r_prime + gamma * max_q_ns_prime - q_values[s_prime][a_prime])

                    if priority_prime > theta:
                        heapq.heappush(priority_queue, (-priority_prime, s_prime, a_prime))

            state = next_state
            total_reward += reward

        ep_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    return np.array(ep_rewards)
