import matplotlib.pyplot as plt
import numpy as np
plt.ion()

class Environment:
    
    def __init__(self, grid_size, layout):
        self.grid = self.build_grid(grid_size, layout)
        self.start_range = layout[0]
        self.agent_loc_x, self.agent_loc_y = 0, 0
        self.n_actions = 9
        self.observation_shape = (32, 32)
        
    def action_sample(self):
        return np.random.randint(9)
<<<<<<< HEAD

    def sample_state(self):
        return(np.random.randint(32), np.random.randint(32))
=======
>>>>>>> 0bcc2835e1a952f53fe8e6b5d0190dc5bef3d7b7
        
    def build_grid(self, grid_size, layout):
        def generate_row(indices, grid_size, cat):
            row = []
            ind1, ind2 = indices
            if cat == 's':
                for _ in range(0, ind1):
                    row.append(State('i'))
                row.extend([State(cat) for _ in range(ind1, ind2+1)])
                for _ in range(ind2+1, grid_size):
                    row.append(State('i'))
            elif cat == 't':
                for _ in range(0, ind1):
                    row.append(State('i'))
                row.append(State('b'))
                for _ in range(ind1+1, grid_size-1):
                    row.append(State('n'))
                row.append(State('t'))
            else:
                for _ in range(0, ind1):
                    row.append(State('i'))
                row.append(State('b'))
                for _ in range(ind1+1, ind2):
                    row.append(State('n'))
                row.append(State('b'))
                for _ in range(ind2+1, grid_size):
                    row.append(State('i'))
            return row, ind2
        grid = []
        for i, indices in enumerate(layout):
            if i == 0:
                row, last_col = generate_row(indices, grid_size, 's')
            elif i > grid_size - 5:
                row, last_col = generate_row(indices, grid_size, 't')
            else:
                row, last_col = generate_row(indices, grid_size, 'n')
            grid.append(row)
        return grid
    
    def get_state_matrix(self):
        n = len(self.grid)
<<<<<<< HEAD
        matrix = np.zeros((n+1, n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = self.grid[i][j].get_rep()
        matrix[-1][0], matrix[-1][1], matrix[-1][2] = self.car_vel_up, self.car_vel_left, self.car_vel_right
=======
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = self.grid[i][j].get_rep()
>>>>>>> 0bcc2835e1a952f53fe8e6b5d0190dc5bef3d7b7
        return matrix
        
    def reset(self):
        self.agent_loc_x = np.random.randint(self.start_range[0], self.start_range[1])
        self.agent_loc_y = 0
        self.car_vel_up = 0
        self.car_vel_left = 0
        self.car_vel_right = 0
        self.grid[self.agent_loc_y][self.agent_loc_x].agent = True
        state = self.get_state_matrix()
        self.grid[self.agent_loc_y][self.agent_loc_x].agent = False
        return state
    
    def step(self, action):
        if action == 0:
            action = (0, 0, 0)
        elif action == 1:
            action = (1, 0, 0)
        elif action == 2:
            action = (-1, 0, 0)
        elif action == 3:
            action = (0, 1, 0)
        elif action == 4:
            action = (0, -1, 0)
        elif action == 5:
            action = (0, 0, 1)
        elif action == 6:
            action = (0, 0, -1)
        elif action == 7:
            action = (1, 1, 0)
        elif action == 8:
            action = (-1, -1, 0)
<<<<<<< HEAD
        if np.random.random() < 0.1:
            action = (0, 0, 0)
        del_up, del_left, del_right = action
        prev_loc = [self.agent_loc_y, self.agent_loc_x]
=======
        del_up, del_left, del_right = action
>>>>>>> 0bcc2835e1a952f53fe8e6b5d0190dc5bef3d7b7
        self.car_vel_up += del_up
        self.car_vel_left += del_left
        self.car_vel_right += del_right
        self.agent_loc_y += self.car_vel_up
        self.agent_loc_x += (self.car_vel_right - self.car_vel_left)
        try:
            curr_state = self.grid[self.agent_loc_y][self.agent_loc_x]
<<<<<<< HEAD
            new_loc = [self.agent_loc_y, self.agent_loc_x]
=======
>>>>>>> 0bcc2835e1a952f53fe8e6b5d0190dc5bef3d7b7
        except:
            done = True
            reward = -100
            state = self.reset()
            return state, reward, done
        curr_state.agent = True
        reward = -1
<<<<<<< HEAD
        if self.trajectory_collision(prev_loc[1], new_loc[1], prev_loc[0], new_loc[0]):
            done = False
            next_state = self.reset()
        else:
            if curr_state.category == 'i':
                done = False
                next_state = self.reset()
            elif curr_state.category == 't':
                done = True
                reward = 100
            else:
                done = False
        state = self.get_state_matrix()
        curr_state.agent = False
        return state, reward, done 

    def trajectory_collision(self, prev_loc, new_loc, prev_row, new_row):
        iter_x = [j for j in range(min(prev_loc, new_loc), max(prev_loc, new_loc) + 1)]
        if prev_loc == new_loc:
            iter_x = [prev_loc for _ in range(prev_row, new_row + 1)]
        iter_y = [j for j in range(prev_row, new_row + 1)]
        for y, x in zip(iter_y, iter_x):
            if self.grid[y][x].category == 'i':
                return True
        return False
=======
        if curr_state.category == 'i':
            done = False
            next_state = self.reset()
        elif curr_state.category == 't':
            done = True
            reward = 100
        else:
            done = False
        state = self.get_state_matrix()
        curr_state.agent = False
        return state, reward, done 
>>>>>>> 0bcc2835e1a952f53fe8e6b5d0190dc5bef3d7b7
    
    def render(self, s):
        plt.imshow(s, cmap='Accent')
        plt.draw()
        plt.pause(0.001)
        plt.clf()


class State:
    
    def __init__(self, category):
#         category = 'b'/'t'/'s'/'n'/'i'
#         b = border
#         t = terminal
#         s = starting
#         n = normal
#         i = inaccessible
        self.category = category
        self.agent = False
        
    def get_rep(self):
        k = 175
        if self.category == 's':
            k = 220
        elif self.category == 't':
            k = 220
        elif self.category == 'i':
            k = 0
        if self.agent:
            k = 255
        return k