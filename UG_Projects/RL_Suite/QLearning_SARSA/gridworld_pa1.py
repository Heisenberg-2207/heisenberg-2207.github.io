from math import floor
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DOWN = 1
UP = 0
LEFT = 2
RIGHT = 3

def row_col_to_seq(row_col, num_cols):  #Converts state number to row_column format
    return row_col[:,0] * num_cols + row_col[:,1]

def seq_to_col_row(seq, num_cols): #Converts row_column format to state number
    r = floor(seq / num_cols)
    c = seq - r * num_cols
    return np.array([[r, c]])

class GridWorld:
    """
    Creates a gridworld object to pass to an RL algorithm.
    Parameters
    ----------
    num_rows : int
        The number of rows in the gridworld.
    num_cols : int
        The number of cols in the gridworld.
    start_state : numpy array of shape (1, 2), np.array([[row, col]])
        The start state of the gridworld (can only be one start state)
    goal_states : numpy arrany of shape (n, 2)
        The goal states for the gridworld where n is the number of goal
        states.
    """
    def __init__(self, num_rows, num_cols, start_state, goal_states, wind = False, max_steps = 100):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.start_state = start_state
        self.goal_states = goal_states
        self.obs_states = None
        self.bad_states = None
        self.num_bad_states = 0
        self.p_good_trans = None
        self.bias = None
        self.r_step = None
        self.r_goal = None
        self.r_dead = None
        self.gamma = 1 # default is no discounting
        self.wind = wind

        self.max_steps = max_steps


    def render_world(self, state = -1, message = "Gridworld"):
        matrix = np.zeros((10, 10)) 

        matrix[self.obs_states[:, 0], self.obs_states[:, 1]] = 1
        matrix[self.bad_states[:, 0], self.bad_states[:, 1]] = 2
        matrix[self.restart_states[:, 0], self.restart_states[:, 1]] = 3
        matrix[self.goal_states[:, 0], self.goal_states[:, 1]] = 4
        matrix[self.start_state[:, 0], self.start_state[:, 1]] = 5
        if(state != -1): 
            state = seq_to_col_row(state,num_cols=10)
            matrix[state[0][0],state[0][1]] = 6

        matrix = np.flipud(matrix)
        plt.figure(figsize=(5,5))
        plt.title(message)
        # Define the colors for your custom colormap
        colors = ['#20908C', '#FFA500', '#BFBF00', '#FF0000', '#008000', '#0000FF', '#FFFFFF']

        # Create the custom colormap
        cmap = ListedColormap(colors)

        # Use the custom colormap in your plot
        plt.pcolor(matrix, edgecolors='k', linewidths=2, cmap=cmap)
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='#20908C', edgecolor='black', linewidth=2, label='Empty state'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#FFA500', edgecolor='black', linewidth=2, label='Obstruction'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#BFBF00', edgecolor='black', linewidth=2, label='Bad State'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#FF0000', edgecolor='black', linewidth=2, label='Restart State'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#008000', edgecolor='black', linewidth=2, label='Goal State'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#0000FF', edgecolor='black', linewidth=2, label='Start State'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#FFFFFF', edgecolor='black', linewidth=2, label='Current State')
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        

    def add_obstructions(self, obstructed_states=None, bad_states=None, restart_states=None):

        self.obs_states = obstructed_states
        self.bad_states = bad_states
        if bad_states is not None:
            self.num_bad_states = bad_states.shape[0]
        else:
            self.num_bad_states = 0
        self.restart_states = restart_states
        if restart_states is not None:
            self.num_restart_states = restart_states.shape[0]
        else:
            self.num_restart_states = 0

    def add_transition_probability(self, p_good_transition, bias):

        self.p_good_trans = p_good_transition
        self.bias = bias

    def add_rewards(self, step_reward, goal_reward, bad_state_reward=None, restart_state_reward = None):

        self.r_step = step_reward
        self.r_goal = goal_reward
        self.r_bad = bad_state_reward
        self.r_restart = restart_state_reward


    def create_gridworld(self):

        self.num_actions = 4
        self.num_states = self.num_cols * self.num_rows# +1
        self.start_state_seq = row_col_to_seq(self.start_state, self.num_cols)
        self.goal_states_seq = row_col_to_seq(self.goal_states, self.num_cols)

        # rewards structure
        self.R = self.r_step * np.ones((self.num_states, 1))
        #self.R[self.num_states-1] = 0
        self.R[self.goal_states_seq] = self.r_goal

        for i in range(self.num_bad_states):
            if self.r_bad is None:
                raise Exception("Bad state specified but no reward is given")
            bad_state = row_col_to_seq(self.bad_states[i,:].reshape(1,-1), self.num_cols)
            #print("bad states", bad_state)
            self.R[bad_state, :] = self.r_bad
        for i in range(self.num_restart_states):
            if self.r_restart is None:
                raise Exception("Restart state specified but no reward is given")
            restart_state = row_col_to_seq(self.restart_states[i,:].reshape(1,-1), self.num_cols)
            #print("restart_state", restart_state)
            self.R[restart_state, :] = self.r_restart

        # probability model
        if self.p_good_trans == None:
            raise Exception("Must assign probability and bias terms via the add_transition_probability method.")

        self.P = np.zeros((self.num_states,self.num_states,self.num_actions))
        for action in range(self.num_actions):
            for state in range(self.num_states):


                # check if the state is the goal state or an obstructed state - transition to end
                row_col = seq_to_col_row(state, self.num_cols)
                if self.obs_states is not None:
                    end_states = np.vstack((self.obs_states, self.goal_states))
                else:
                    end_states = self.goal_states

                if any(np.sum(np.abs(end_states-row_col), 1) == 0):
                    self.P[state, state, action] = 1

                # else consider stochastic effects of action
                else:
                    for dir in range(-1,2,1):

                        direction = self._get_direction(action, dir)
                        next_state = self._get_state(state, direction)
                        if dir == 0:
                            prob = self.p_good_trans
                        elif dir == -1:
                            prob = (1 - self.p_good_trans)*(self.bias)
                        elif dir == 1:
                            prob = (1 - self.p_good_trans)*(1-self.bias)

                        self.P[state, next_state, action] += prob

                # make restart states transition back to the start state with
                # probability 1
                if self.restart_states is not None:
                    if any(np.sum(np.abs(self.restart_states-row_col),1)==0):
                        next_state = row_col_to_seq(self.start_state, self.num_cols)
                        self.P[state,:,:] = 0
                        self.P[state,next_state,:] = 1
        return self

    def _get_direction(self, action, direction):

        left = [2,3,1,0]
        right = [3,2,0,1]
        if direction == 0:
            new_direction = action
        elif direction == -1:
            new_direction = left[action]
        elif direction == 1:
            new_direction = right[action]
        else:
            raise Exception("getDir received an unspecified case")
        return new_direction

    def _get_state(self, state, direction):

        row_change = [-1,1,0,0]
        col_change = [0,0,-1,1]
        row_col = seq_to_col_row(state, self.num_cols)
        row_col[0,0] += row_change[direction]
        row_col[0,1] += col_change[direction]

        # check for invalid states
        if self.obs_states is not None:
            if (np.any(row_col < 0) or
                np.any(row_col[:,0] > self.num_rows-1) or
                np.any(row_col[:,1] > self.num_cols-1) or
                np.any(np.sum(abs(self.obs_states - row_col), 1)==0)):
                next_state = state
            else:
                next_state = row_col_to_seq(row_col, self.num_cols)[0]
        else:
            if (np.any(row_col < 0) or
                np.any(row_col[:,0] > self.num_rows-1) or
                np.any(row_col[:,1] > self.num_cols-1)):
                next_state = state
            else:
                next_state = row_col_to_seq(row_col, self.num_cols)[0]

        return next_state

    def reset(self):
      self.steps = 0
      return int(self.start_state_seq)

    def step(self, state, action):
        self.done = False
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        if state in self.goal_states_seq:
            self.done = True
        
        p, r = 0, np.random.random()
        for next_state in range(self.num_states):

            p += self.P[state, next_state, action]

            if r <= p:
                break

        if(self.wind and np.random.random() < 0.4):

          arr = self.P[next_state, :, 3]
          next_next = np.where(arr == np.amax(arr))
          next_next = next_next[0][0]
          return next_next, self.R[next_next], self.done
        else:
          return next_state, self.R[next_state], self.done
        


def world(world_num):
    num_cols = 10
    num_rows = 10
    obstructions = np.array([[0,7],[1,1],[1,2],[1,3],[1,7],[2,1],[2,3],
                            [2,7],[3,1],[3,3],[3,5],[4,3],[4,5],[4,7],
                            [5,3],[5,7],[5,9],[6,3],[6,9],[7,1],[7,6],
                            [7,7],[7,8],[7,9],[8,1],[8,5],[8,6],[9,1]])
    bad_states = np.array([[1,9],[4,2],[4,4],[7,5],[9,9]])
    restart_states = np.array([[3,7],[8,2]])
    goal_states = np.array([[0,9],[2,2],[8,7]])

    if world_num == 0:
        start_state = np.array([[0,4]])
        wind = False
        p_good_transition = 1
    elif world_num == 1:
        start_state = np.array([[0,4]])
        wind = False
        p_good_transition = 0.7
    elif world_num == 2:
        start_state = np.array([[0,4]])
        wind = True
        p_good_transition = 1
    elif world_num == 3:
        start_state = np.array([[3,6]])
        wind = False
        p_good_transition = 1
    elif world_num == 4:
        start_state = np.array([[3,6]])
        wind = False
        p_good_transition = 0.7
    elif world_num == 5:
        start_state = np.array([[3,6]])
        wind = True
        p_good_transition = 1
    else:
        print("Invalid world number")


    # create model
    gw = GridWorld(num_rows=num_rows,
                num_cols=num_cols,
                start_state=start_state,
                goal_states=goal_states, wind = wind)
    gw.add_obstructions(obstructed_states=obstructions,
                        bad_states=bad_states,
                        restart_states=restart_states)
    gw.add_rewards(step_reward=-1,
                goal_reward=10,
                bad_state_reward=-6,
                restart_state_reward=-100)
    gw.add_transition_probability(p_good_transition,
                                bias=0.5)
    return gw.create_gridworld()

def plot_Q(Q, world_num = -1, message = "Q plot"):
    
    Q = np.flipud(Q.reshape(10,10,4))
    
    plt.figure(figsize=(10,10))
    plt.title(message)
    plt.pcolor(Q.max(-1), edgecolors='k', linewidths=2)
    plt.colorbar()
    def x_direct(a):
        if a in [UP, DOWN]:
            return 0
        return 1 if a == RIGHT else -1
    def y_direct(a):
        if a in [RIGHT, LEFT]:
            return 0
        return 1 if a == UP else -1
    policy = Q.argmax(-1)
    policyx = np.vectorize(x_direct)(policy)
    policyy = np.vectorize(y_direct)(policy)
    idx = np.indices(policy.shape)
    plt.quiver(idx[1].ravel()+0.5, idx[0].ravel()+0.5, policyx.ravel(), policyy.ravel(), pivot="middle", color='red')
    
    plt.savefig('world_' + str(world_num) + '_Q_plot.png')
    plt.show()


