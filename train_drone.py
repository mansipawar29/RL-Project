#! /usr/bin/env python
# environment inspired by Morvan
import rospy
import time
import pandas as pd
import numpy as np
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from gazebo_connection import GazeboConnection

#seting environemt and training parameters 
N_STATES = 5   # states of environment (available position of the drone)
ACTIONS = ['fly_left', 'fly_right']     # available actions in dron environment
EPSILON = 0.5   # greedy policy
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
EPISODES = 8   # number episodes which the drone is going to "play"
TRANSITION_TIME = 0.1     # transition time from one state to other

gazebo = GazeboConnection()

"""
Q table - the simple "brain" of the drone.
During the training the Q table is going to be updated 
"""
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    return table

"""
For each of the state the drone is, the brain of the drone (agent) decides
about the action to take - based on updated Q table.
"""
def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = np.argmax(state_actions) 
    return action_name

"""
Feedback from the environment.
The drone (agent) receives the rewards 0 or 1.
"""
def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'fly_right':    
        if S == N_STATES - 2:  
            S_ = 'goal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    if A == 'fly_left':
        R = 0
        if S == 0:
            S_ = S  
        else:
            S_ = S - 1
    return S_, R

"""
Update of the environent.
Environment/drone transits to other state.
Update of the terminal.
"""
def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['_']*(N_STATES-1) + ['Goal']   
    if S == 'goal':
        interaction = 'Episode :: %s:: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction))
        time.sleep(2)
        print('\r                                ')

        gazebo.resetSim()
    else:
        env_list[S] = 'X'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction))
        time.sleep(TRANSITION_TIME)


"""
Class for definition of drone movements.
Learning function implementation (Bellman equation).
"""
class MoveSquareClass(object):
    
    def __init__(self):

        self.ctrl_c = False
        self.rate = rospy.Rate(10)


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
                #rospy.loginfo("Publish in cmd_vel...")
                break
            else:
                self.rate.sleep()
                
    # function that stops the drone from any movement
    def stop_drone(self):
        #rospy.loginfo("Stopping...")
        self._move_msg.linear.x = 0.0
        self._move_msg.angular.z = 0.0
        self.publish_once_in_cmd_vel(self._move_msg)
            
    # function that makes the drone turn 90 degrees
    def turn_drone(self):
        #rospy.loginfo("Turning...")
        self._move_msg.linear.x = 0.0
        self._move_msg.angular.z = 1.0
        self.publish_once_in_cmd_vel(self._move_msg)
        
    # function that makes the drone move forward
    def move_forward_drone(self):
        #rospy.loginfo("Moving forward...")
        self._move_msg.linear.y = 0.5
        self._move_msg.angular.z = 0.0
        self.publish_once_in_cmd_vel(self._move_msg)
        #self.stop_drone()
        #time.sleep(10)

    def move_forward_drone_opposite(self):
        #rospy.loginfo("Moving opposite...")
        self._move_msg.linear.y = -0.5
        self._move_msg.angular.z = 0.0
        self.publish_once_in_cmd_vel(self._move_msg)
        #self.stop_drone()
        #time.sleep(10)
        
    """
    Implementation of learning process for the drone(agent).

    """
    def learn(self):

        # helper variables
        r = rospy.Rate(1)
        
        # define the different publishers and messages that will be used
        self._pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self._move_msg = Twist()
        self._pub_takeoff = rospy.Publisher('/drone/takeoff', Empty, queue_size=1)
        self._takeoff_msg = Empty()
        self._pub_land = rospy.Publisher('/drone/land', Empty, queue_size=1)
        self._land_msg = Empty()

        q_table = build_q_table(N_STATES, ACTIONS)
        for episode in range(EPISODES):
        
            # make the drone takeoff
            i=0
            while not i == 3:
                self._pub_takeoff.publish(self._takeoff_msg)

                time.sleep(1)
                i += 1
            
            # define the seconds to move in each side of the square (which is taken from the goal) and the seconds to turn
            sideSeconds = 1
            turnSeconds = 1.8


            step_counter = 0
            S = 0
            is_terminated = False
            update_env(S, episode, step_counter)
            
            #########################
            ### LEARNING PROCESS ####
            #########################
            while not is_terminated:

                A = choose_action(S, q_table)
                              
                if A == 'fly_right':
                    self.move_forward_drone()
                    time.sleep(sideSeconds)
       
                if A == 'fly_left':
                    self.move_forward_drone_opposite()
                    time.sleep(sideSeconds)


                S_, R = get_env_feedback(S, A)  # take action & get next state and reward
                q_predict = q_table.loc[S, A]
                if S_ != 'goal':
                    q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
                else:
                    q_target = R     # next state is terminal
                    is_terminated = True    # terminate this episode
                #########################
                ### BELLMAN EQUATION ###
                #########################
                q_table.loc[S, A] += ALPHA * (q_target - q_predict)  
                S = S_  # move to next state

                update_env(S, episode, step_counter+1)
                step_counter += 1


        gazebo.resetSim()
       
        self.stop_drone()
        i=0
        while not i == 3:
            self._pub_land.publish(self._land_msg)
            time.sleep(1)
            i += 1
        
        return q_table
      
if __name__ == '__main__':
    rospy.init_node('move_square')
    move_square = MoveSquareClass()
    try:
        q_table = move_square.learn()
        print('\rQ(a,s)::')
        print('-----------------')
        print(q_table)
    except rospy.ROSInterruptException:
        pass

