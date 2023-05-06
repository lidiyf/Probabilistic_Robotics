import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 70}

    def __init__(self, render_mode=None): #, size=20):
        #self.size = size  # The size of the square grid
        self.window_size = 400 #512  # The size of the PyGame window
        self.reward = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
                "angular": spaces.Box(0, 361, shape=(1, ), dtype=int),
                #"target": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
                #"pedestrian": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
                #"pedestrian2": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
                #"static": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
                #"static2": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
                #"static3": spaces.Box(0, self.window_size - 1, shape=(2,), dtype=int),
                "sensor": spaces.Box(0, 201, shape=(151, ), dtype=int),
                "goal": spaces.Box(0, 201, shape=(1,), dtype=int)
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
            0: np.array([0, 1, 0]), #np.array([1, 0]),
            1: np.array([3, 3, 3]), #np.array([0, 1]),
            2: np.array([3, 3, -3]), #np.array([-1, 0]),
            #3: (0, 0, 0),
            #3: np.array([0, -1]),
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
        self.reward = 0
        
        self._sensor_grid = self.np_random.integers(0, 1, size=151, dtype=int)

        # render no overlap and inside the frame
        self._static_location = self.np_random.integers(0, self.window_size - 35, size=2, dtype=int)
        self._static2_location = self.np_random.integers(0, self.window_size - 35, size=2, dtype=int)
        self._static3_location = self.np_random.integers(0, self.window_size - 35, size=2, dtype=int)


        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.window_size, size=2, dtype=int)
        self._agent_angle = self.np_random.integers(0, 361, size=1, dtype=int)

        self._calc_angle((self._agent_location + 10), self._agent_angle * np.pi / 180-5*np.pi/12)

        self.center = (self._agent_location + 10)
        self.radian = self._agent_angle * np.pi / 180
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location + self.np_random.integers(-20, 20, size=2, dtype=int)
        while np.linalg.norm(self._agent_location - self._target_location) >= 170 or np.linalg.norm(self._agent_location - self._target_location) <= 10  or abs(self._agent_angle - ((np.arctan2(self._target_location[1]-self.center[1], self._target_location[0]-self.center[0])*180/np.pi) % 360)) >= 65:
            self._target_location = self._agent_location + self.np_random.integers(-20, 20, size=2, dtype=int)
        # agent location or center?
        
        self._pedestrian_location = self.np_random.integers(0, self.window_size, size=2, dtype=int)
        while np.array_equal(self._pedestrian_location, self._agent_location):
            self._pedestrian_location = self.np_random.integers(
                0, self.window_size, size=2, dtype=int
            )
        self._pedestrian2_location = self.np_random.integers(0, self.window_size, size=2, dtype=int)
        while np.array_equal(self._pedestrian2_location, self._agent_location):
            self._pedestrian2_location = self.np_random.integers(
                0, self.window_size, size=2, dtype=int
            )
        
        '''
        self._dist_to_name = {
            0: "self._pedestrian_location",
            1: "self._pedestrian2_location",
            2: "self._static_location",
            3: "self._static2_location",
            4: "self._static3_location"
        }'''

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

        self._goal = self.np_random.integers(0, 1, size=1, dtype=int)
        self._goal[0] = int(np.linalg.norm(self._agent_location - self._target_location))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # np.cos(), np.sin(), 
        # Map the action (element of {0,1,2,3}) to the direction we walk in

        direction = self._action_to_direction[int(action)][:2]
        if action != 0:
            direction = (int(direction[0]*np.cos(np.pi/180*self._agent_angle)), int(direction[1]*np.sin(np.pi/180*self._agent_angle)))

        self._agent_angle = (self._agent_angle + self._action_to_direction[int(action)][-1]) % 360


        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.window_size - 20
        )

        # change their walking behaviors
        target_direction = np.array([1, 1])#self._action_to_direction[self.action_space.sample()][:2]
        self._target_location = np.clip(
            self._target_location + target_direction, 0, self.window_size - 20
        )

        pedestrian_direction = self._action_to_direction[self.action_space.sample()][:2]
        self._pedestrian_location = np.clip(
            self._pedestrian_location + pedestrian_direction, 0, self.window_size - 20
        )
        pedestrian2_direction = self._action_to_direction[self.action_space.sample()][:2]
        self._pedestrian2_location = np.clip(
            self._pedestrian2_location + pedestrian2_direction, 0, self.window_size - 20
        )

        self.center = (self._agent_location + 10)
        self.radian = self._agent_angle * np.pi / 180

        self._sensor_grid = self.np_random.integers(0, 1, size=151, dtype=int)

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

        if np.linalg.norm(self._agent_location - self._target_location) <= 200 and abs(self._agent_angle - ((np.arctan2(self._target_location[1]-self.center[1], self._target_location[0]-self.center[0])*180/np.pi) % 360)) <= 95:
            #self._goal = np.array(np.linalg.norm(self._agent_location - self._target_location), dtype=int)
            self._goal[0] = int(np.linalg.norm(self._agent_location - self._target_location))

        else:
            self._goal[0] = 0

        # An episode is done iff the agent has reached the target
        #terminated = np.array_equal(self._agent_location, self._target_location)
        if self._target_location[0] == 0 or self._target_location[1] == 0 or self._target_location[0] >= self.window_size-20 or self._target_location[1] >= self.window_size-20:
            terminated = True
        else:
            terminated = False
        reward = self.calc_rewards() #1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def _collision(self):
        if np.linalg.norm(self._agent_location - (self._pedestrian_location+10)) <= (10 + 10):
            return -25
        elif np.linalg.norm(self._agent_location - (self._pedestrian_location+10)) <= (10 + 10 + 5):
            return -5
        if np.linalg.norm(self._agent_location - (self._pedestrian2_location+10)) <= (10 + 10):
            return -25
        elif np.linalg.norm(self._agent_location - (self._pedestrian2_location+10)) <= (10 + 10 + 5):
            return -5
        if np.linalg.norm(self._agent_location - (self._static_location+35/2)) <= (10 + 35/2):
            return -25
        elif np.linalg.norm(self._agent_location - (self._static_location+35/2)) <= (10 + 35/2 + 5):
            return -5
        if np.linalg.norm(self._agent_location - (self._static2_location+35/2)) <= (10 + 35/2):
            return -25
        elif np.linalg.norm(self._agent_location - (self._static2_location+35/2)) <= (10 + 35/2 + 5):
            return -5
        if np.linalg.norm(self._agent_location - (self._static3_location+35/2)) <= (10 + 35/2):
            return -25
        elif np.linalg.norm(self._agent_location - (self._static3_location+35/2)) <= (10 + 35/2 + 5):
            return -5
        return 0
    
    def calc_rewards(self):
        if self._goal == 0:
            self.reward = -50
        elif self._goal >= 0:
            self.reward += 15
        
        self.reward += self._collision()
        
        return self.reward

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