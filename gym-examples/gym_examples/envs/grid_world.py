import numpy as np
import pygame
import csv
from csv import writer

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 70}

    def __init__(self, render_mode=None): #, size=20):
        #self.size = size  # The size of the square grid
        self.window_size = 500 #512  # The size of the PyGame window
        #self.reward = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=float),
                "angular": spaces.Box(0, 361, shape=(1, ), dtype=int),
                #"target": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
                #"pedestrian": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
                #"pedestrian2": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
                #"static": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
                #"static2": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
                #"static3": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
                "sensor": spaces.Box(0, 201, shape=(171, ), dtype=int),
                "goal": spaces.Box(0, 201, shape=(1,), dtype=int),
                "goal_angle": spaces.Box(0, 361, shape=(1, ), dtype=int)
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(3)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            #0: np.array([0, 1, 0]), #np.array([1, 0]),
            0: np.array([0.7, 0.7, 5]), #np.array([0, 1]),
            1: np.array([0.7, 0.7, -5]), #np.array([-1, 0]),
            2: np.array([0, 0, 0]),
            2: np.array([0.5, 0.5, 0])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, 
                "angular": self._agent_angle,
                #"target": self._target_location, 
                #"pedestrian": self._pedestrian_location, 
                #"pedestrian2": self._pedestrian2_location,
                #"static": self._static_location,
                #"static2": self._static2_location, 
                #"static3": self._static3_location, 
                "sensor": self._sensor_grid,
                "goal": self._goal,
                "goal_angle": self._goal_angle,
                }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        #self.reward = 0
        self.success = False
        
        self._sensor_grid = self.np_random.integers(0, 1, size=171, dtype=int)

        # render no overlap and inside the frame
        self._static_location = self.np_random.integers(0, self.window_size - 35, size=2, dtype=int)
        self._static2_location = self.np_random.integers(0, self.window_size - 35, size=2, dtype=int)
        while np.linalg.norm(self._static2_location - self._static_location) <= 80:
            self._static2_location = self.np_random.integers(0, self.window_size - 35, size=2, dtype=int)
        self._static3_location = self.np_random.integers(0, self.window_size - 35, size=2, dtype=int)
        while np.linalg.norm(self._static3_location - self._static_location) <= 80 or np.linalg.norm(self._static3_location - self._static2_location) <= 80:
            self._static3_location = self.np_random.integers(0, self.window_size - 35, size=2, dtype=int)


        # Choose the agent's location uniformly at random
        #self._agent_location = self.np_random.integers(0, self.window_size, size=2, dtype=int)
        #self._agent_angle = self.np_random.integers(0, 361, size=1, dtype=int)
        self._target_location = self.np_random.integers(self.window_size/2-150, self.window_size/2+150, size=2, dtype=int) #self.np_random.integers(0, self.window_size, size=2, dtype=int)
        #self._pedestrian_location = self.np_random.integers(0, self.window_size, size=2, dtype=int)
        #self._pedestrian2_location = self.np_random.integers(0, self.window_size, size=2, dtype=int)

        self._target_side = np.random.choice(range(4), 1)
        self._agent_location = self._target_location + self.np_random.integers(25, 51, size=2, dtype=int)        
        if self._target_side == 0:
            self._target_location[1] = self.np_random.integers(50, 100, size=1, dtype=int)
            self._agent_location[1] = self.np_random.integers(0, 1, size=1, dtype=int)
        elif self._target_side == 1:
            self._target_location[0] = self.np_random.integers(self.window_size-100, self.window_size-50, size=1, dtype=int)
            self._agent_location[0] = self.np_random.integers(self.window_size-20, self.window_size-20+1, size=1, dtype=int)
        elif self._target_side == 2:
            self._target_location[1] = self.np_random.integers(self.window_size-100, self.window_size-50, size=1, dtype=int)
            self._agent_location[1] = self.np_random.integers(self.window_size-20, self.window_size-20+1, size=1, dtype=int)
        elif self._target_side ==3:
            self._target_location[0] = self.np_random.integers(50, 100, size=1, dtype=int)
            self._agent_location[0] = self.np_random.integers(0, 1, size=1, dtype=int)

        #while np.linalg.norm(self._agent_location - self._target_location) >= 170 or np.linalg.norm(self._agent_location - self._target_location) <= 50:
        #    self._agent_location = self._target_location + self.np_random.integers(-100, 100, size=2, dtype=int)


        self._pedestrian_location = self.gen_ped(self._target_side)
        self._pedestrian2_location = self.gen_ped2(self._target_side)
        #self._pedestrian_direction = -(self._pedestrian_location[1] - self.window_size/2) / (self._pedestrian_location[0] - self.window_size/2)
        #self._calc_angle((self._agent_location + 10), self._agent_angle * np.pi / 180-5*np.pi/12)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        #self._agent_location = self._agent_location.astype(float)
        self.center = (self._agent_location + 10)
        # agent location or center?
        self._agent_angle = self.np_random.integers(0, 361, size=1, dtype=int)
        while abs(self._agent_angle - ((np.arctan2(self._target_location[1]-self.center[1], self._target_location[0]-self.center[0])*180/np.pi) % 360)) >= 45:
            self._agent_angle = self.np_random.integers(0, 361, size=1, dtype=int)
        self.radian = self._agent_angle * np.pi / 180


        '''while np.array_equal(self._pedestrian_location, self._agent_location):
            self._pedestrian_location = self.np_random.integers(
                0, self.window_size, size=2, dtype=int
            )'''
        '''while np.array_equal(self._pedestrian2_location, self._agent_location):
            self._pedestrian2_location = self.np_random.integers(
                0, self.window_size, size=2, dtype=int
            )'''
        
        '''
        self._dist_to_name = {
            0: "self._pedestrian_location",
            1: "self._pedestrian2_location",
            2: "self._static_location",
            3: "self._static2_location",
            4: "self._static3_location"
        }'''
        self._p_i = int(np.random.choice([0,1], 1))
        self._p_ip = int(np.random.choice([0,1], 1))

        self._dist = [self._pedestrian_location, self._pedestrian2_location, self._static_location, self._static2_location, self._static3_location]

        self._cangle = self._calc_angle(self.center, self.radian)
        for i in range(len(self._dist)):
            #d = self._distance(self._agent_location, self._dist[i])
            d = np.linalg.norm(self._agent_location - self._dist[i])
            if d <= 200:
                r = np.arctan2(self._dist[i][1]-self.center[1], self._dist[i][0]-self.center[0])
                deg = r*180/np.pi % 360
                diff = self._agent_angle - deg
                if abs(diff) <= 75:
                    #print(self._dist_to_name[i], self._agent_angle, deg, abs(diff))
                    if deg < 0:
                        self._sensor_grid[75+int(abs(diff))] = d
                    else:
                        self._sensor_grid[75-int(diff)] = d

        self._i1 = 0
        self._i2 = 0
        self._goal = self.np_random.integers(0, 1, size=1, dtype=int)
        self._goal[0] = int(np.linalg.norm(self._agent_location - self._target_location))
        #self._pre_goal = self._goal
        self._count = 0
        self._goal_angle = np.array([int(abs(self._agent_angle - ((np.arctan2(self._target_location[1]-self.center[1], self._target_location[0]-self.center[0])*180/np.pi) % 360)))])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # np.cos(), np.sin(), 
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        p = {
            0:[0.5, 0.2, 0.3],
            1:[0.3, 0.2, 0.5],
        }
        
        rec = np.array(np.random.choice([-1, 0, 1], 1, p=p[self._p_i]))#self.np_random.integers(-1, 2, size=1, dtype=int)
        dom = self.np_random.integers(0, 2, size=1, dtype=int)
        self._target_direction = {
            0: np.concatenate([rec, dom]),
            1: np.concatenate([-1*dom, rec]),
            2: np.concatenate([rec, -1*dom]),
            3: np.concatenate([dom, rec]),
        }

        rec_ped = np.array(np.random.choice([-2, 0, 2], 1, p=p[self._p_ip]))#self.np_random.integers(-1, 2, size=1, dtype=int)
        dom_ped = self.np_random.integers(0, 2, size=1, dtype=int)
        self._ped_direction = {
            0: np.concatenate([rec_ped, dom_ped]),
            1: np.concatenate([-1*dom_ped, rec_ped]),
            2: np.concatenate([rec_ped, -1*dom_ped]),
            3: np.concatenate([dom_ped, rec_ped]),
        }

        direction_angle = self._action_to_direction[int(action)][:2]
        #if action != 0:
        #one_step = self.np_random.integers(2, 3, size=2, dtype=int)

        self._pre_agent_location = [self._agent_location[0], self._agent_location[1]]
        self._agent_angle = (self._agent_angle + self._action_to_direction[int(action)][-1]) % 360
        # We use `np.clip` to make sure we don't leave the grid
        direction = np.zeros(2) #np.random.normal(0.4, 0.1, 2)
        direction[0] = direction_angle[0]*np.cos(np.pi/180*(360-self._agent_angle))
        direction[1] = -direction_angle[1]*np.sin(np.pi/180*(360-self._agent_angle))
        #print(direction)
        #print(self._action_to_direction[int(action)])
        #direction = np.array([np.cos(np.pi/180*(360-self._agent_angle)), -np.sin(np.pi/180*(360-self._agent_angle))])
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.window_size - 20
        )
        self._dagent = np.linalg.norm(self._pre_agent_location - self._agent_location)
        #print(self._dagent)

        self._angle_to_center = abs(self._agent_angle - ((np.arctan2(self.window_size/2-self.center[1], self.window_size/2-self.center[0])*180/np.pi) % 360))
        #print(self._angle_to_center)

        # change their walking behaviors
        target_direction = self._target_direction[int(self._target_side)] #self._action_to_direction[self.action_space.sample()][:2]
        if self._target_location[0] == 0 or self._target_location[1] == 0 or self._target_location[0] >= self.window_size-20 or self._target_location[1] >= self.window_size-20:
            target_direction = np.array([0, 0]) #???
        self._target_location = np.clip(
            self._target_location + target_direction, 0, self.window_size - 20
        )

        '''
        
            if self._i2 % 2 == 0:
                if self._pedestrian_location[0] - self.window_size/2 > 0:
                    rec_ped = np.array(np.random.choice([-2, 0, 2], 1, p=[0.8, 0.2, 0.]))
                elif self._pedestrian_location[0] - self.window_size/2 < 0:
                    rec_ped = np.array(np.random.choice([-2, 0, 2], 1, p=[0., 0.2, 0.8]))
            else:
                if self._pedestrian_location[1] - self.window_size/2 > 0:
                    rec_ped = np.array(np.random.choice([-2, 0, 2], 1, p=[0.8, 0.2, 0.]))
                elif self._pedestrian_location[1] - self.window_size/2 < 0:
                    rec_ped = np.array(np.random.choice([-2, 0, 2], 1, p=[0., 0.2, 0.8]))
            #self._pedestrian_location[]
            #extra = (self._pedestrian_location[1] - self.window_size/2) / (self._pedestrian_location[0] - self.window_size/2) /4
        if self._pedestrian_location[0] == 0 or self._pedestrian_location[1] == 0 or self._pedestrian_location[0] >= self.window_size-20 or self._pedestrian_location[1] >= self.window_size-20:
            self._pedestrian_direction = (self._pedestrian_location[1] - self.window_size/2) / np.linalg.norm(self._pedestrian_direction - [self.window_size/2, self.window_size/2])
        '''
        if self.boundary(self._pedestrian_location, (int(self._target_side)+2+self._i1)%4):
            self._p_pi = int(np.random.choice([0,1], 1))
            self._pedestrian_location = self.gen_ped((int(self._target_side)+2+self._i1)%4)
            self._i1 += 2
        pedestrian_direction = self._ped_direction[(int(self._target_side)+2+self._i1)%4] #self._action_to_direction[self.action_space.sample()][:2]
        self._pedestrian_location = np.clip(
            self._pedestrian_location + pedestrian_direction, 0, self.window_size - 20
        )

        if self.boundary(self._pedestrian2_location, (int(self._target_side)+3+self._i2)%4):
            self._p_pi = int(np.random.choice([0,1], 1))
            self._pedestrian2_location = self.gen_ped((int(self._target_side)+3+self._i2)%4)
            self._i2 += 2    
        pedestrian2_direction = self._ped_direction[(int(self._target_side)+3+self._i2)%4] #self._action_to_direction[self.action_space.sample()][:2]
        self._pedestrian2_location = np.clip(
            self._pedestrian2_location + pedestrian2_direction, 0, self.window_size - 20
        )

        self.center = (self._agent_location + 10)
        self.radian = self._agent_angle * np.pi / 180

        self._sensor_grid = self.np_random.integers(0, 1, size=171, dtype=int)

        self._dist = [self._pedestrian_location, self._pedestrian2_location, self._static_location, self._static2_location, self._static3_location]
        self._cangle = self._calc_angle(self.center, self.radian)
        for i in range(len(self._dist)):
            #d = self._distance(self._agent_location, self._dist[i])
            d = np.linalg.norm(self._agent_location - self._dist[i])
            if d <= 200:
                r = np.arctan2(self._dist[i][1]-self.center[1], self._dist[i][0]-self.center[0])
                deg = r*180/np.pi % 360
                diff = self._agent_angle - deg
                if abs(diff) <= 85:
                    #print(self._dist_to_name[i], self._agent_angle, deg, abs(diff))
                    if deg < 0:
                        self._sensor_grid[85+int(abs(diff))] = d
                    else:
                        self._sensor_grid[85-int(diff)] = d

        #print(self._sensor_grid)
        self._pre_goal = int(self._goal[0])

        if np.linalg.norm(self._agent_location - self._target_location) <= 200 and abs(self._agent_angle - ((np.arctan2(self._target_location[1]-self.center[1], self._target_location[0]-self.center[0])*180/np.pi) % 360)) <= 80:
            #self._goal = np.array(np.linalg.norm(self._agent_location - self._target_location), dtype=int)
            self._goal[0] = int(np.linalg.norm(self._agent_location - self._target_location))

        else:
            self._goal[0] = 0
        self._dgoal = int(self._goal[0]) - self._pre_goal

        '''
        if self._goal == 0 and (self._agent_angle%90 <= 20 or self._agent_angle%90 >= 70):
            print(self._count)
            self._count += 1'''

        self._count += 1
        self._goal_angle = np.array([int(abs(self._agent_angle - ((np.arctan2(self._target_location[1]-self.center[1], self._target_location[0]-self.center[0])*180/np.pi) % 360)))])
        #print(self._goal_angle)

        success = False
        # An episode is done iff the agent has reached the target
        #terminated = np.array_equal(self._agent_location, self._target_location)
        if self._goal == 0:
            terminated = True

        elif (self._target_location[0] == 0 or self._target_location[1] == 0 or self._target_location[0] == self.window_size-20 or self._target_location[1] == self.window_size-20) and np.linalg.norm(self._agent_location - self._target_location) <= 50 and self._goal > 0:
            #print("completed", self._target_location)
            success = True
            #print("success!", self._target_location)
            terminated = True
            #print("target location", self._target_location)
            #print("distance to agent", np.linalg.norm(self._agent_location - self._target_location))
            #print("angle", abs(self._agent_angle - ((np.arctan2(self._target_location[1]-self.center[1], self._target_location[0]-self.center[0])*180/np.pi) % 360)))
        #elif abs(self._agent_location[0] - self._target_location[0]) <= 3 and abs(self._agent_location[1] - self._target_location[1]) <= 3:
        #    terminated = True
        else:
            terminated = False
        reward = self.calc_rewards(success, terminated) #1 if terminated else 0  # Binary sparse rewards
        #print(reward)
        #print(self._angle_to_center)
        observation = self._get_obs()
        info = self._get_info()

        '''
        if terminated:
            with open('follow.csv', 'a') as f:
                w = writer(f)
                w.writerow([self._count])
                f.close()
            with open('reward.csv', 'a') as f:
                w = writer(f)
                w.writerow([reward])
                f.close()'''

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def calc_rewards(self, succ, term):
        #if term:
        #    self.reward += 50 
            #print("+ terminated")
        #    return self.reward
        #if self._goal == 0 and (self._agent_location[0] <= 10 or self._agent_location[1] <= 10 or self._agent_location[0] >= self.window_size-30 or self._agent_location[1] >= self.window_size-30):
        #    if self._angle_to_center <= 90 or self._angle_to_center >= 360-90:
        #        self.reward += +1
                #print("+ angle towards center", self._angle_to_center)
        #    else: #elif self._angle_to_center >= 90 or self._angle_to_center >= 360-90:
        #        self.reward += -1
                #print("- angle outward")
            #if (self._agent_location[0] <= 30 or self._agent_location[1] <= 30 or self._agent_location[0] >= self.window_size-30 or self._agent_location[1] >= self.window_size-30) and (self._angle_to_center <= 90 or self._angle_to_center >= 360-90):
            #    self.reward += 7
            #elif (self._agent_location[0] <= 30 or self._agent_location[1] <= 30 or self._agent_location[0] >= self.window_size-30 or self._agent_location[1] >= self.window_size-30) and (self._angle_to_center <= 180 or self._angle_to_center >= 360-180):
            #    self.reward += 1
            #elif np.linalg.norm(self._agent_location - [0, 0]) <= 3 or np.linalg.norm(self._agent_location - [0, self.window_size]) <= 23 or np.linalg.norm(self._agent_location - [self.window_size, 0]) <= 23 or np.linalg.norm(self._agent_location - [self.window_size, self.window_size]) <= 23:
            #    self.reward += -35
            #elif (self._agent_location[0] <= 30 or self._agent_location[0] >= self.window_size-30 or self._agent_location[1] <= 30 or self._agent_location[1] >= self.window_size-30) and (self._agent_angle%90 <= 20 or self._agent_angle%90 >= 70):
            #    self.reward += -25
        #elif self._goal >= 0:
        #if not term:
            #if self._dgoal < 0:
            #    self.reward += 2
                    #print("+ closer")
        #    self.reward += 1
        #    self.reward += self._collision()  
        #if term:
        #    self.reward += self._count
        ret = 0
        if not term and self._goal_angle <= 30:
            ret += 5
            ret += self._collision()
        elif succ:
            ret += 200
        else:
            ret += -500
        return ret

    def _collision(self):
        ret = 0
        if np.linalg.norm(self.center - (self._pedestrian_location+10)) <= (10 + 14):
            ret = -100
        elif np.linalg.norm(self.center - (self._pedestrian_location+10)) <= (10 + 14 + 3):
            ret = -10
        if np.linalg.norm(self.center - (self._pedestrian2_location+10)) <= (10 + 14):
            ret = -100
        elif np.linalg.norm(self._agent_location - (self._pedestrian2_location+10)) <= (10 + 14 + 3):
            ret = -10
        if np.linalg.norm(self._agent_location - (self._static_location+35/2)) <= (10 + 24):
            ret = -100
        elif np.linalg.norm(self._agent_location - (self._static_location+35/2)) <= (10 + 24 + 3):
            ret = -10
        if np.linalg.norm(self._agent_location - (self._static2_location+35/2)) <= (10 + 24):
            ret = -100
        elif np.linalg.norm(self._agent_location - (self._static2_location+35/2)) <= (10 + 24 + 3):
            ret = -10
        if np.linalg.norm(self._agent_location - (self._static3_location+35/2)) <= (10 + 24):
            ret = -100
        elif np.linalg.norm(self._agent_location - (self._static3_location+35/2)) <= (10 + 24 + 3):
            ret = -10 
        #print("0 no collision")
        return ret
    
    def gen_ped(self, side):
        ret = self.np_random.integers(self.window_size/2-150, self.window_size/2+150, size=2, dtype=int)

        if side==0:
            ret[1] = self.np_random.integers(self.window_size-100, self.window_size, size=1, dtype=int)     
        elif side==1:
            ret[0] = self.np_random.integers(0, 100, size=1, dtype=int)
        elif side==2:
            ret[1] = self.np_random.integers(0, 100, size=1, dtype=int)
        elif side==3:
            ret[0] = self.np_random.integers(self.window_size-100, self.window_size, size=1, dtype=int)
        return ret

    def gen_ped2(self, side):
        ret = self.np_random.integers(self.window_size/2-150, self.window_size/2+150, size=2, dtype=int)

        if side==0:
            ret[0] = self.np_random.integers(0, 100, size=1, dtype=int)
        elif side==1:
            ret[1] = self.np_random.integers(0, 100, size=1, dtype=int)
        elif side==2:
            ret[0] = self.np_random.integers(self.window_size-100, self.window_size, size=1, dtype=int)
        elif side==3:
            ret[1] = self.np_random.integers(self.window_size-100, self.window_size, size=1, dtype=int)
        return ret

    def boundary(self, pos, side):
        if side == 0:
            return pos[1] >= self.window_size - 20
        elif side == 1:
            return pos[0] == 0
        elif side == 2:
            return pos[1] == 0
        elif side == 3:
            return pos[0] >= self.window_size - 20

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        '''pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels '''
        
        #orig = (center[0], center[1]-10)
        #orig2 = (orig[0], orig[1]-50)
        pygame.draw.rect(
            canvas,
            (0,0,0),
            pygame.Rect(
                self._static_location,
                (35, 35),
            ),
            width=1,
        )

        pygame.draw.rect(
            canvas,
            (0,0,0),
            pygame.Rect(
                self._static2_location,
                (35, 35),
            ),
            width=1,
        )

        pygame.draw.rect(
            canvas,
            (0,0,0),
            pygame.Rect(
                self._static3_location,
                (35, 35),
            ),
            width=1,
        )

        # First we draw the target
        '''pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                self._target_location,
                (20, 20),
            ),
        )'''

        pygame.draw.circle(
            canvas,
            (255, 0,0),
            (self._target_location + 10),
            10
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 10), #0.5),
            10,
        )
        
        pygame.draw.aaline(
            canvas,
            (0,0,255),
            self.center,
            self._calc_angle(self.center, self.radian),
        )
        
        pygame.draw.aaline(
            canvas,
            (0,0,255),
            self.center,
            self._calc_angle(self.center, self.radian-5*np.pi/12),
        )

        pygame.draw.aaline(
            canvas,
            (0,0,255),
            self.center,
            self._calc_angle(self.center, self.radian+5*np.pi/12)
        )

        pygame.draw.aaline(
            canvas,
            (0,0,255),
            self.center,
            self._calc_angle(self.center, self.radian+np.pi/4)
        )

        pygame.draw.aaline(
            canvas,
            (0,0,255),
            self.center,
            self._calc_angle(self.center, self.radian-np.pi/4)
        )

        pygame.draw.aaline(
            canvas,
            (0,0,255),
            self.center,
            self._calc_angle(self.center, self.radian+np.pi/12)
        )

        pygame.draw.aaline(
            canvas,
            (0,0,255),
            self.center,
            self._calc_angle(self.center, self.radian-np.pi/12)
        )

        pygame.draw.rect(
            canvas,
            (0, 0, 0),
            pygame.Rect(
                self._pedestrian_location,
                (20, 20),
            ),
            width=1,
        )
        
        
        pygame.draw.rect(
            canvas,
            (0, 0, 0),
            pygame.Rect(
                self._pedestrian2_location,
                (20, 20),
            ),
            width=1,
        )
        
        # Finally, add some gridlines
        '''for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )'''

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def _calc_angle(self, center, a, dist=200):
        x = center[0] + np.cos(a) * dist
        y = center[1] + np.sin(a) * dist
        return (x, y)

    def _insideRect(self, x, y, x1, x2, y1, y2):
        if x > x1 and x < x2 and y > y1 and y < y2:
            return True
        return False

    def _distance(self, pt1, pt2):
        return np.sqrt(pow(pt2[0]-pt1[0], 2) + pow(pt2[1]-pt1[1],2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()