## Author: core CEM code taken from John Lathrop and modified by Sorina Lupu.

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import Callable
from nn import MLP

def clip_vector(vector: np.ndarray, hypercube: np.ndarray) -> np.ndarray:
    return np.clip(vector, hypercube[:, 0], hypercube[:, 1])

class CEM:
    def __init__(self,
                 action_dim: int,
                 action_lims: np.ndarray,
                 horizon: int,
                 num_particles: int,
                 num_iterations: int,
                 num_elite: int,
                 mean: np.ndarray,
                 cov: np.ndarray,
                 smoothing: float = 0.5,
                 max_std_stop: float = 0.01,
                 seed: int = 0):

        self.horizon = horizon # number of steps to plan for.
        self.num_particles = num_particles # number of particles to sample.
        assert num_iterations >= 1
        self.num_iterations = num_iterations # number of iterations to run.
        self.num_elite = int(num_elite/100.0 * num_particles) # number of elite particles to keep.
        self.mean = mean # (horizon x action_dim)
        self.cov = cov # (horizon x action_dim x action_dim)
        self.cov_noise = 1e-0
        self.generator = np.random.default_rng(seed)
        self.smoothing = smoothing
        self.max_std_stop = max_std_stop
        self.action_dim = action_dim
        self.action_lims = action_lims
        self.action_folders = 'actions'
        if not os.path.exists(self.action_folders):
            os.makedirs(self.action_folders)

    def optimize(self, initial_state: np.ndarray, dynamics_model: MLP, reward: Callable, goal: float):
        best_actions = np.zeros((self.num_iterations, self.horizon, self.action_dim))
        best_action = np.zeros((self.horizon, self.action_dim))
        best_value = -np.inf
        mean = self.mean
        cov = self.cov

        pbar = tqdm(range(self.num_iterations), desc='CEM Progress')

        for iter_nb in pbar:
            # Generate all particles at once
            particles = np.array([
                clip_vector(self.generator.multivariate_normal(self.mean[i, :], self.cov[i, :, :]),
                          self.action_lims)
                for i in range(self.horizon)
            ])  # (horizon x action_dim)
            particles = np.tile(particles, (self.num_particles, 1, 1))  # (num_particles x horizon x action_dim)

            # Initialize states for all particles.
            states = np.tile(initial_state, (self.num_particles, 1))  # (num_particles x state_dim)
            values = np.zeros(self.num_particles)

            # Process all particles in parallel for each timestep
            for t in range(self.horizon):
                # Get actions for current timestep for all particles
                actions = particles[:, t, :]  # (num_particles x action_dim)

                if t == 0:
                    past_actions = actions
                else:
                    past_actions = particles[:, :t-1, :] # (num_particles x action_dim)

                # Prepare input for dynamics model
                inputs = np.hstack([states, actions])  # (num_particles x (state_dim + action_dim))

                # Get next states for all particles
                next_states = dynamics_model.forward(inputs).detach().numpy()  # (num_particles x state_dim)

                # Compute rewards for all particles
                rewards = np.array([reward(states[i], actions[i], past_actions[i], goal) for i in range(len(states))])
                values += rewards

                # Update states for next iteration
                states = next_states.copy()

            # Average values over horizon
            values = values / float(self.horizon)

            # Select elite particles
            elite_indices = np.argsort(values)[-self.num_elite:]
            elite_particles = particles[elite_indices]  # (num_elite x horizon x action_dim)

            # Update mean and covariance
            mean = np.mean(elite_particles, axis=0)  # (horizon x action_dim)
            cov = self.computecovariance(elite_particles, mean)

            # Update best action if better value found
            if values[elite_indices[-1]] > best_value:
                best_value = values[elite_indices[-1]]
                best_action = elite_particles[-1]

            self.mean = mean
            self.cov = cov + np.array([self.cov_noise * np.eye(self.action_dim) for _ in range(self.horizon)])
            best_actions[iter_nb] = best_action

            # Check convergence
            stds = np.sqrt(np.linalg.eigvalsh(cov))
            if np.amax(stds) <= self.max_std_stop:
                print("Converged at iteration", iter_nb)
                break

        return best_action, best_actions, best_value

    def computecovariance(self, particles, mean):
        # particles is (num_particles x horizon x action_dim)
        # mean is (horizon x action_dim)
        centered_particles = particles - np.tile(mean, (particles.shape[0], 1, 1))  # (num_particles x horizon x action_dim)
        cov = np.array([np.mean([np.outer(particle, particle) for particle in centered_particles[:, i, :]], axis=0)
                       for i in range(particles.shape[1])])
        return cov
