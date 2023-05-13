# Probabilistic Robotics Final Project

To run the latest version of AgileBot
1. Clone tina's branch
   ```sh
   git clone -b tina https://github.com/lidiyf/Probabilistic_Robotics
   ```
2. Install required packages
   ```sh
   pip install -r requirements.txt
   ```
   or 
   Install separately
   ```sh
   pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
   ```
   ```sh
   pip install gymnasium
   ```
   ```sh
   pip install tqdm
   ```
   ```sh
   pip install rich
   ```
   ```sh
   pip install pygame
   ```
3. To test our PPO moodel
   ```sh
   python gym-examples/test.py
   ```
   or to run training
   ```sh
   python gym-examples/env.py   
   ```