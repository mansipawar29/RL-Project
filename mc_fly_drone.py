#! /usr/bin/env python

import rospy
import time
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
import random
import gym
import numpy as np

env = gym.make('FrozenLake-v0')

def drone_motion():
    P = env.env.P
    reward_goal, reward_holes, reward_others = 1, -1, -0.04
    # default goal AAA
    goal, hole = 2, [1, 3, 13]

    for s in range(len(P)):
        for a in range(len(P[s])):
            for t in range(len(P[s][a])):
                values = list(P[s][a][t])
                if values[1] == goal:
                    values[2] = reward_goal
                    values[3] = False
                elif values[1] in hole:
                    values[2] = reward_holes
                    values[3] = False
                else:
                    values[2] = reward_others
                    values[3] = False
                if s in hole or s == goal:
                    values[2] = 0
                    values[3] = True
                P[s][a][t] = tuple(values)

    # change transition function
    prob_action, prob_drift_one, prob_drift_two = 0.8, 0.1, 0.1
    for s in range(len(P)):
        for a in range(len(P[s])):
            for t in range(len(P[s][a])):
                if P[s][a][t][0] == 1.0:
                    continue
                values = list(P[s][a][t])
                if t == 0:
                    values[0] = prob_drift_one
                elif t == 1:
                    values[0] = prob_action
                elif t == 2:
                    values[0] = prob_drift_two
                P[s][a][t] = tuple(values)
    env.env.P = P


    """
    def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
        print(title)
        for s in range(len(P)):
            v = V[s]
            print("| ", end="")
            if np.all([done for action in P[s].values() for _, _, _, done in action]):
                print("".rjust(9), end=" ")
            else:
                print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
            if (s + 1) % n_cols == 0: print("|")
    """
    def generate_trajectory(select_action, Q, epsilon, env, max_steps=200):
        done, trajectory = False, []
        while not done:
            state = env.reset()
            for t in count():
                action = select_action(state, Q, epsilon)
                next_state, reward, done, _ = env.step(action)
                experience = (state, action, reward, next_state, done)
                trajectory.append(experience)
                if done:
                    break
                if t >= max_steps - 1:
                    trajectory = []
                    break
                state = next_state
        return np.array(trajectory, np.object)

    def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), 'edge')
        return values

    def value_iteration(P, gamma=1.0, theta=1e-10):
        V = np.zeros(len(P), dtype=np.float64)
        while True:
            Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
            for s in range(len(P)):
                for a in range(len(P[s])):
                    for prob, next_state, reward, done in P[s][a]:
                        Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
                break
            V = np.max(Q, axis=1)
        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return Q, V, pi

    def print_action_value_function(Q, 
                                    optimal_Q=None, 
                                    action_symbols=('<', '>'), 
                                    prec=3, 
                                    title='Action-value function:'):
        vf_types=('',) if optimal_Q is None else ('', '*', 'err')
        headers = ['s',] + [' '.join(i) for i in list(itertools.product(vf_types, action_symbols))]
        states = np.arange(len(Q))[..., np.newaxis]
        arr = np.hstack((states, np.round(Q, prec)))
        if not (optimal_Q is None):
            arr = np.hstack((arr, np.round(optimal_Q, prec), np.round(optimal_Q-Q, prec)))
        
    def get_policy_metrics(env, gamma, pi, goal_state, optimal_Q, 
                        n_episodes=100, max_steps=200):
        random.seed(123); np.random.seed(123) ; env.seed(123)
        reached_goal, episode_reward, episode_regret = [], [], []
        for _ in range(n_episodes):
            state, done, steps = env.reset(), False, 0
            episode_reward.append(0.0)
            episode_regret.append(0.0)
            while not done and steps < max_steps:
                action = pi(state)
                regret = np.max(optimal_Q[state]) - optimal_Q[state][action]
                episode_regret[-1] += regret
                
                state, reward, done, _ = env.step(action)
                episode_reward[-1] += (gamma**steps * reward)
                
                steps += 1

            reached_goal.append(state == goal_state)
        results = np.array((np.sum(reached_goal)/len(reached_goal)*100, 
                            np.mean(episode_reward), 
                            np.mean(episode_regret)))
        return results
    """
    def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Optimal policy:'):
        print(title)
        arrs = {k:v for k,v in enumerate(action_symbols)}
        for s in range(len(P)):
            a = pi(s)
            print("| ", end="")
            if s == goal:
                print("GOAL".rjust(6), end=" ")
            
            if np.all([done for action in P[s].values() for _, _, _, done in action]):
                if s != goal:
                    print("HOLE".rjust(6), end=" ")

                
            else:
                #print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
                print(arrs[a].rjust(6), end=" ")
            if (s + 1) % n_cols == 0: print("|")
    """
                
    
    def drone_policy_path(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Optimal policy:'):
        drone_policy = []
        
        arrs = {k:v for k,v in enumerate(action_symbols)}
        for s in range(len(P)):
            a = pi(s)
            if s == goal:
                drone_policy.append('goal')
            
            if np.all([done for action in P[s].values() for _, _, _, done in action]):
                if s != goal:
                    drone_policy.append('hole')  
            else:            
                drone_policy.append(arrs[a])    
        return drone_policy
                

    def mc_control(env,
                gamma=0.99,
                init_alpha=0.5,
                min_alpha=0.01,
                alpha_decay_ratio=0.5,
                init_epsilon=1.0,
                min_epsilon=0.1,
                epsilon_decay_ratio=0.9,
                n_episodes=5000,
                max_steps=200,
                first_visit=True):
        nS, nA = env.observation_space.n, env.action_space.n
        discounts = np.logspace(0, 
                                max_steps, 
                                num=max_steps, 
                                base=gamma, 
                                endpoint=False) 
        alphas = decay_schedule(init_alpha, 
                            min_alpha, 
                            alpha_decay_ratio, 
                            n_episodes)
        epsilons = decay_schedule(init_epsilon, 
                                min_epsilon, 
                                epsilon_decay_ratio, 
                                n_episodes)
        pi_track = []
        Q = np.zeros((nS, nA), dtype=np.float64)
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
            if np.random.random() > epsilon \
            else np.random.randint(len(Q[state]))

        for e in tqdm(range(n_episodes), leave=False):
            
            trajectory = generate_trajectory(select_action,
                                            Q,
                                            epsilons[e],
                                            env, 
                                            max_steps)
            visited = np.zeros((nS, nA), dtype=np.bool)
            for t, (state, action, reward, _, _) in enumerate(trajectory):
                if visited[state][action] and first_visit:
                    continue
                visited[state][action] = True
                
                n_steps = len(trajectory[t:])
                G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
                Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])

            Q_track[e] = Q
            pi_track.append(np.argmax(Q, axis=1))

        V = np.max(Q, axis=1)
        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return Q, V, pi, Q_track, pi_track



    #from tqdm import tqdm_notebook as tqdm
    from tqdm import tqdm
    from itertools import cycle, count
    import itertools
    from tabulate import tabulate
    SEEDS = (12, 34, 56, 78, 90)
    n_episodes=3000
    gamma = 0.95

    Q_mcs, V_mcs, Q_track_mcs = [], [], []
    for seed in tqdm(SEEDS, desc='All seeds', leave=True):
        random.seed(seed); np.random.seed(seed) ; env.seed(seed)
        Q_mc, V_mc, pi_mc, Q_track_mc, pi_track_mc = mc_control(env, gamma=gamma, n_episodes=n_episodes)
        Q_mcs.append(Q_mc) ; V_mcs.append(V_mc) ; Q_track_mcs.append(Q_track_mc)
    Q_mc, V_mc, Q_track_mc = np.mean(Q_mcs, axis=0), np.mean(V_mcs, axis=0), np.mean(Q_track_mcs, axis=0)
    del Q_mcs ; del V_mcs ; del Q_track_mcs


    optimal_Q, optimal_V, optimal_pi = value_iteration(P, gamma=gamma)

    goal_state = 9
    gamma = 0.95
    n_episodes = 100000
    P = env.env.P
    n_cols, svf_prec, err_prec, avf_prec=4, 4, 2, 3
    action_symbols=('<', 'v', '>', '^')
    limit_items, limit_value = 5, 0.01
    cu_limit_items, cu_limit_value, cu_episodes = 10, 0.0, 1000
    d = drone_policy_path(pi_mc, P, action_symbols=action_symbols, n_cols=n_cols)
    policy = np.reshape(d, (4,4))

    def next_state(action, state):

        if action == 'v': #1
            return [state[0]+1, state[1]]
        if action == '<': #2
            return [state[0], state[1]-1]
        if action == '^': #3
            return [state[0]-1, state[1]]
        if action == '>': #4
            return [state[0], state[1]+1]
        
    def find_goal(city_map):
        for i in range(city_map.shape[0]):
            for j in range(city_map.shape[1]):
                if city_map[i][j] == 'goal':
                    return [i,j]


    def drone_flying_plan(policy):
        drone_state = [0,0]
        motion_plan = []
        motion_plan.append(drone_state) 
        while(drone_state != find_goal(policy)):
            action = policy[drone_state[0]][drone_state[1]]
            drone_state = next_state(action,drone_state)
            motion_plan.append(drone_state)
        return motion_plan

    def drone_motion(flying_plan, policy):
        actual_heading = 0
        motion = []
        motion_corr = []
        for step in flying_plan:
            if policy[step[0]][step[1]] == 'v':
                motion.append(1) 
            if policy[step[0]][step[1]] == '<':
                motion.append(1)
            if policy[step[0]][step[1]] == '^':
                motion.append(-1)
            if policy[step[0]][step[1]] == '>':
                motion.append(-1)
    
        for m in motion:
            if m == actual_heading:
                motion_corr.append(0)
                actual_heading_temp = 0
            if  m != actual_heading:
                motion_corr.append(m)
                actual_heading_temp = m
            actual_heading = actual_heading_temp
        return motion_corr   

    def drone_motion_dev(flying_plan, policy):
        motion = []
        for step in flying_plan:
            if policy[step[0]][step[1]] == 'v':
                motion.append(1) 
            if policy[step[0]][step[1]] == '<':
                motion.append(2)
            if policy[step[0]][step[1]] == '^':
                motion.append(3)
            if policy[step[0]][step[1]] == '>':
                motion.append(0)
        return motion  


    flying_plan = drone_flying_plan(policy)
    m = drone_motion(flying_plan, policy)
    return m

class MoveDroneClass(object):

    def __init__(self):

        self.ctrl_c = False
        self.rate = rospy.Rate(1)

    def publish_once_in_cmd_vel(self, cmd):
        """
        This is because publishing in topics sometimes fails teh first time you publish.
        In continuos publishing systems there is no big deal but in systems that publish only
        once it IS very important.
        """
        while not self.ctrl_c:
            connections = self._pub_cmd_vel.get_num_connections()
            if connections > 0:
                self._pub_cmd_vel.publish(cmd)
                rospy.loginfo("Publish in cmd_vel...")
                break
            else:
                self.rate.sleep()

    # function that stops the drone from any movement
    def stop_drone(self):
        rospy.loginfo("Stopping...")
        self._move_msg.linear.x = 0.0
        self._move_msg.angular.z = 0.0
        self.publish_once_in_cmd_vel(self._move_msg)

    # function that makes the drone turn 90 degrees
    def turn_drone(self, move):
        rospy.loginfo("Turning...")
        self._move_msg.linear.x = 0.0
        self._move_msg.angular.z = -0.5 * move *2 # 
        self.publish_once_in_cmd_vel(self._move_msg)

    # function that makes the drone move forward
    def move_forward_drone(self):
        rospy.loginfo("Moving forward...")
        self._move_msg.linear.x = 0.2 * 3
        self._move_msg.angular.z = 0.0
        self.publish_once_in_cmd_vel(self._move_msg)

    def move_drone(self, motion):
        actual_heading = 0
        r = rospy.Rate(5)
        self._pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self._move_msg = Twist()
        self._pub_takeoff = rospy.Publisher('/drone/takeoff', Empty, queue_size=1)
        self._takeoff_msg = Empty()
        self._pub_land = rospy.Publisher('/drone/land', Empty, queue_size=1)
        self._land_msg = Empty()
        sideSeconds = 3.3
        turnSeconds = 1.5  

          # ===================DRONE TAKEOFF===========================================
        i = 0
        while not i == 2:
            self._pub_takeoff.publish(self._takeoff_msg)
            rospy.loginfo('Taking off...')
            time.sleep(1)
            i += 1
         # ==========================================================================
        
        for move in motion:
            print("motion ::::", move) #, "turning time : ", turning_time)

            if move == 0:
                self.move_forward_drone()
                time.sleep(sideSeconds)
                actual_heading = move

            if move != 0:
                
                self.turn_drone(move)
                time.sleep(turnSeconds)
                self.move_forward_drone()
                r.sleep()
                time.sleep(sideSeconds)
                actual_heading = move
            r.sleep()

        # ===================DRONE STOP AND LAND=====================================
        self.stop_drone()
        i = 0
        while not i == 3:
            self._pub_land.publish(self._land_msg)
            rospy.loginfo('Landing...')
            time.sleep(1)
            i += 1
        # =============================================================================


if __name__ == '__main__':
    motion = drone_motion()
    print(motion)
    rospy.init_node('move_drone')
    move_drone = MoveDroneClass()
    try:
        move_drone.move_drone(motion)
    except rospy.ROSInterruptException:
        pass




