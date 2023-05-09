import numpy as np
import pygame
import csv
from csv import writer

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 70}

    def __init__(self, render_mode=None):
        self.window_size = 400

        self.observation_space = spaces.Dict(
            {
                "angular": spaces.Box(0, 361, shape=(1, ), dtype=int),
                "sensor": spaces.Box(-85, 87, shape=(5,), dtype=int),
                "goal": spaces.Box(0, 202, shape=(1,), dtype=int),
                "goal_angle": spaces.Box(-85, 87, shape=(1, ), dtype=int),
                "sensor_dist": spaces.Box(0,202, shape=(5,), dtype=int)
            }
        )

        self.action_space = spaces.Discrete(3)

        self._action_to_direction = {
            0: np.array([0.7, 0.7, 5]), 
            1: np.array([0.7, 0.7, -5]), 
            2: np.array([0, 0, 0]),
            2: np.array([0.5, 0.5, 0])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
                "angular": self._agent_angle,
                "sensor": self._sensor_grid,
                "goal": self._goal,
                "goal_angle": self._goal_angle,
                "sensor_dist": self._sensor_dist,
                }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._sensor_grid = self.np_random.integers(86, 87, size=5, dtype=int)
        self._sensor_dist = self.np_random.integers(201, 202, size=5, dtype=int)

        self._static_location = self.np_random.integers(0, self.window_size - 35, size=2, dtype=int)
        self._static2_location = self.np_random.integers(0, self.window_size - 35, size=2, dtype=int)
        while np.linalg.norm(self._static2_location - self._static_location) <= 80:
            self._static2_location = self.np_random.integers(0, self.window_size - 35, size=2, dtype=int)
        self._static3_location = self.np_random.integers(0, self.window_size - 35, size=2, dtype=int)
        while np.linalg.norm(self._static3_location - self._static_location) <= 80 or np.linalg.norm(self._static3_location - self._static2_location) <= 80:
            self._static3_location = self.np_random.integers(0, self.window_size - 35, size=2, dtype=int)

        self._target_location = self.np_random.integers(self.window_size/2-150, self.window_size/2+150, size=2, dtype=int) #self.np_random.integers(0, self.window_size, size=2, dtype=int)

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

        self._pedestrian_location = self.gen_ped(self._target_side)
        self._pedestrian2_location = self.gen_ped2(self._target_side)

        self.center = (self._agent_location + 10)

        self._agent_angle = self.np_random.integers(0, 361, size=1, dtype=int)
        while abs(self._agent_angle - ((np.arctan2(self._target_location[1]-self.center[1], self._target_location[0]-self.center[0])*180/np.pi) % 360)) >= 45:
            self._agent_angle = self.np_random.integers(0, 361, size=1, dtype=int)
        self.radian = self._agent_angle * np.pi / 180

        self._p_i = int(np.random.choice([0,1], 1))
        self._p_ip = int(np.random.choice([0,1], 1))

        self._dist = [self._pedestrian_location, self._pedestrian2_location, self._static_location, self._static2_location, self._static3_location]

        #self._cangle = self._calc_angle(self.center, self.radian)
        for i in range(len(self._dist)):
            d = np.linalg.norm(self._agent_location - self._dist[i])
            if d <= 200:
                r = np.arctan2(self._dist[i][1]-self.center[1], self._dist[i][0]-self.center[0])
                deg = r*180/np.pi % 360
                diff = self._agent_angle - deg
                if abs(diff) <= 75:
                    self._sensor_grid[i] = diff
                    self._sensor_dist[i] = d

        self._i1 = 0
        self._i2 = 0

        self._goal = self.np_random.integers(201, 202, size=1, dtype=int)
        if int(np.linalg.norm(self._agent_location - self._target_location)) < 201:
            self._goal[0] = int(np.linalg.norm(self._agent_location - self._target_location))

        self._count = 0
        self._goal_angle = np.array([int(self._agent_angle - ((np.arctan2(self._target_location[1]-self.center[1], self._target_location[0]-self.center[0])*180/np.pi) % 360))])

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action):
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

        self._pre_agent_location = [self._agent_location[0], self._agent_location[1]]
        self._agent_angle = (self._agent_angle + self._action_to_direction[int(action)][-1]) % 360
        direction = np.zeros(2) 
        direction[0] = direction_angle[0]*np.cos(np.pi/180*(360-self._agent_angle))
        direction[1] = -direction_angle[1]*np.sin(np.pi/180*(360-self._agent_angle))
        
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.window_size - 20
        )
        #self._dagent = np.linalg.norm(self._pre_agent_location - self._agent_location)
        #self._angle_to_center = abs(self._agent_angle - ((np.arctan2(self.window_size/2-self.center[1], self.window_size/2-self.center[0])*180/np.pi) % 360))

        target_direction = self._target_direction[int(self._target_side)] #self._action_to_direction[self.action_space.sample()][:2]
        if self._target_location[0] == 0 or self._target_location[1] == 0 or self._target_location[0] >= self.window_size-20 or self._target_location[1] >= self.window_size-20:
            target_direction = np.array([0, 0]) #???
        self._target_location = np.clip(
            self._target_location + target_direction, 0, self.window_size - 20
        )

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

        self._sensor_grid = self.np_random.integers(86, 87, size=5, dtype=int)
        self._sensor_dist = self.np_random.integers(201, 202, size=5, dtype=int)

        self._dist = [self._pedestrian_location, self._pedestrian2_location, self._static_location, self._static2_location, self._static3_location]
        #self._cangle = self._calc_angle(self.center, self.radian)

        for i in range(len(self._dist)):
            d = np.linalg.norm(self._agent_location - self._dist[i])
            if d <= 200:
                r = np.arctan2(self._dist[i][1]-self.center[1], self._dist[i][0]-self.center[0])
                deg = r*180/np.pi % 360
                diff = self._agent_angle - deg
                if abs(diff) <= 75:
                    self._sensor_grid[i] = diff
                    self._sensor_dist[i] = d

        #self._pre_goal = int(self._goal[0])

        if np.linalg.norm(self._agent_location - self._target_location) <= 200 and abs(self._agent_angle - ((np.arctan2(self._target_location[1]-self.center[1], self._target_location[0]-self.center[0])*180/np.pi) % 360)) <= 80:
            self._goal[0] = int(np.linalg.norm(self._agent_location - self._target_location))

        else:
            self._goal[0] = 201
        #self._dgoal = int(self._goal[0]) - self._pre_goal

        self._count += 1

        self._goal_angle = np.array([int(self._agent_angle - ((np.arctan2(self._target_location[1]-self.center[1], self._target_location[0]-self.center[0])*180/np.pi) % 360))])

        success = False
        if self._goal > 200:
            terminated = True
        elif (self._target_location[0] == 0 or self._target_location[1] == 0 or self._target_location[0] == self.window_size-20 or self._target_location[1] == self.window_size-20) and np.linalg.norm(self._agent_location - self._target_location) <= 50 and self._goal < 201:
            success = True
            terminated = True
        else:
            terminated = False
        
        
        if terminated:
            with open('num_steps.csv', 'a') as f:
                w = writer(f)
                w.writerow([self._count])
                f.close()
        

        reward = self.calc_rewards(success, terminated)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, False, info
    
    def calc_rewards(self, succ, term):
        ret = 0
        if not term:
            ret += self._collision()
            if abs(self._goal_angle) <= 30:
                ret += 3
        elif succ:
            ret += 0
        else:
            ret += -300
        return ret

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

        # draw static obstacles
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
        # draw target
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
        # draw FOV sector
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
        
        # generating two pedestrians
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

        # bounding box for generating target
        '''
        pygame.draw.rect(
            canvas,
            (255, 0,0),
            pygame.Rect(
                (100, 100),
                (200,200),
            ),
            width=1,
        )
        '''

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
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# helper functions
    #detect collision or danger
    def _collision(self):
        ret = 0
        
        if np.linalg.norm(self.center - (self._pedestrian_location+10)) <= (10 + 14 + 3):
            ret = -15 #4
        elif np.linalg.norm(self.center - (self._pedestrian_location+10)) <= (10 + 14):
            ret = -25 #6

        if np.linalg.norm(self._agent_location - (self._pedestrian2_location+10)) <= (10 + 14 + 3):
            ret = -15
        elif np.linalg.norm(self.center - (self._pedestrian2_location+10)) <= (10 + 14):
            ret = -25

        if np.linalg.norm(self._agent_location - (self._static_location+35/2)) <= (10 + 24 + 3):
            ret = -15
        elif np.linalg.norm(self._agent_location - (self._static_location+35/2)) <= (10 + 24):
            ret = -25

        if np.linalg.norm(self._agent_location - (self._static2_location+35/2)) <= (10 + 24 + 3):
            ret = -15
        elif np.linalg.norm(self._agent_location - (self._static2_location+35/2)) <= (10 + 24):
            ret = -25

        if np.linalg.norm(self._agent_location - (self._static3_location+35/2)) <= (10 + 24 + 3):
            ret = -15
        elif np.linalg.norm(self._agent_location - (self._static3_location+35/2)) <= (10 + 24):
            ret = -25
        return ret
    # generate pedestrian #1
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
    # generate pedestrian #2
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