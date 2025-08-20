# 

# Reinforcement Learning Algorithms and Experiments

This repository contains a comprehensive reinforcement learning (RL) project implemented as part of a coursework assignment. The goal was to explore foundational RL algorithms, compare their performance across various environments, and study advanced techniques including Deep RL. (report.pdf)


## Project Overview

The project is divided into three main parts:

### Task 1: Classical RL in Grid World Environments

- Built a **5×5 grid world** using Pygame with obstacles, lava tiles, and goal states.

- Implemented and compared:
  
  - Random Agent
    
    <img src="https://github.com/rojinakashefi/ReinforcementLearning/blob/main/photos/1.png" title="" alt="1.png" width="410">
  
  - Fixed Agent (hardcoded behavior)
    
    <img src="https://github.com/rojinakashefi/ReinforcementLearning/blob/main/photos/2.png" title="" alt="2.png" width="416">
    
    
  
  - Q-learning
  
  - SARSA
    
    ![3.png](https://github.com/rojinakashefi/ReinforcementLearning/blob/main/photos/3.png)
    
    

### Task 2: Temporal Difference vs Monte Carlo

- Compared **SARSA**, **Q-learning**, and **Monte Carlo** in four environments:
  
  - Empty Room
  
  - Room with Lava
  
  - Cliff Environment
  
  - Monster Environment
    
    <img src="https://github.com/rojinakashefi/ReinforcementLearning/blob/main/photos/4.png" title="" alt="4.png" width="396">

- Studied the effects of:
  
  - On-policy vs Off-policy learning
  
  - Learning rate variation
    
    ![5.png](https://github.com/rojinakashefi/ReinforcementLearning/blob/main/photos/5.png)
  
  - Epsilon (exploration) settings and decay
    
    ![6.png](https://github.com/rojinakashefi/ReinforcementLearning/blob/main/photos/6.png)
  
  - Model-based learning via Dyna-Q
    
    ![7.png](https://github.com/rojinakashefi/ReinforcmentLearning/blob/main/photos/7.png)

### Task 3: Function Approximation and Deep RL

- Implemented Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) with a neural network.

- Compared tabular vs deep Q-learning.

- Tested in:
  
  - Empty Room
  
  - Room with Multiple Monsters

- Techniques used:
  
  - Experience Replay
  
  - Target Networks
  
  - Double DQN, Dueling DQN, Distributional DQN
  
  - REINFORCE, Actor-Critic, PPO with entropy regularization

![Screenshot 2025-08-20 at 6.46.53 PM.png](https://github.com/rojinakashefi/ReinforcementLearning/blob/main/photos/8.png)

----

This project is © 2025 KU Leuven and may not be used without permission.
