import torch
import numpy as np
from tqdm import tqdm

import mujoco
import numpy as np
from mujoco import viewer
from robot_stand_up.nn import MLP
from robot_stand_up.cem import CEM

DURATION = 200 # seconds
VISUALIZE = True
PREDEFINED_CONFIG = True # Choice between predefined config (see the xml) or random.
DURATION_RANDOM_POLICY = 1000 # seconds

# Planner parameters.
HORIZON_PLANNER = 10
NUM_PARTICLES = 100
NUM_ITERATIONS = 100
NUM_ELITE = 100

def reward_L1(state: np.ndarray, goal: float = 0.2) -> float:
    # TODO: add in the gravity component.
    # z_component = state[..., 0]  # TODO: add for both single states and batches
    z_component = state[0] # '0' corresponds to the z-component.
    # TODO: add the control action penalty.
    return -np.abs(z_component - goal)

if __name__ == "__main__":
    # Create MuJoCo env.    
    model = mujoco.MjModel.from_xml_path("xmls/scene_mjx_fullcollisions_flat_terrain.xml")
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    if VISUALIZE:
        v = viewer.launch_passive(model, data)

    # Initialize robot in a random or desired configuration.
    if PREDEFINED_CONFIG == True:
        init_q = np.array(model.keyframe("on_the_belly").qpos)
        init_ctrl = np.array(model.keyframe("on_the_belly").ctrl)
    else:
        init_q = np.random.random(data.qpos.shape)
        init_ctrl = np.random.random(data.ctrl.shape)
    data.qpos = init_q
    data.ctrl = init_ctrl
    mujoco.mj_step(model, data)
    
    action = init_ctrl.copy()
    
    # Initialize dynamics model: Model(S, A) -> S'.
    dynamics_model = MLP(input_size=len(data.qpos[2: ]) + len(data.qvel) + len(data.ctrl),
                        hidden_size=200,
                        output_size=len(data.qpos[2: ]) + len(data.qvel))
    dynamics_model_inputs = [] # States and actions.
    dynamics_model_outputs = [] # Next states.

    # Planner.
    planner = CEM(
                  action_dim=data.ctrl.shape[0],
                  action_lims=model.actuator_ctrlrange,
                  horizon=HORIZON_PLANNER,
                  num_particles=NUM_PARTICLES,
                  num_iterations=NUM_ITERATIONS,
                  num_elite=NUM_ELITE,
                  mean=np.zeros((HORIZON_PLANNER, data.ctrl.shape[0])), # (horizon x action_dim)
                  cov=np.array([np.eye(data.ctrl.shape[0]) for _ in range(HORIZON_PLANNER)]) # (horizon x action_dim x action_dim)
                 )

    # Run a random policy for a while to collect data.
    state = np.hstack([data.qpos[2: ],
                       data.qvel])
    next_state = np.zeros_like(state) # TODO: figure out if putting zeros is ok.
    pbar = tqdm(range(DURATION_RANDOM_POLICY), desc='Duration Random Policy Progress')
    for i in pbar:
        action = init_ctrl + np.random.normal(0, 0.5, data.ctrl.shape)
        data.ctrl = action

        mujoco.mj_step(model, data)

        next_state = np.hstack([data.qpos[2: ],
                                data.qvel])
        dynamics_model_input = np.hstack([state, action])
        dynamics_model_inputs.append(dynamics_model_input)
        dynamics_model_outputs.append(next_state)
        state = next_state.copy()

        if VISUALIZE is True:
            if v.is_running():
                v.sync()

    t = 0
    while t < DURATION: # Loop forever.

        # Get state [qpos [z, quat, joints], qvel]
        state = np.hstack([data.qpos[2: ],
                           data.qvel])

        # Take action (optimal from CEM).
        for i in range(HORIZON_PLANNER):
            data.ctrl = action

            # Step the simulation.
            mujoco.mj_step(model, data)

            # Get next true state.
            next_state = np.hstack([data.qpos[2: ],
                                    data.qvel])

            # Learn dynamics.
            dynamics_model_input = np.hstack([state, action])
            dynamics_model_inputs.append(dynamics_model_input)
            dynamics_model_outputs.append(next_state)

        X_train = np.array(dynamics_model_inputs)
        Y_train = np.array(dynamics_model_outputs)
        dynamics_model.train(X_train, Y_train, max_epochs=100, lr=0.002, batch_size=100)

        # Plan with CEM and the learned dynamics.
        action_traj, _, _ = planner.optimize(state, dynamics_model, reward_L1, goal=0.2)
        action = action_traj[0] # Take first action from the trajectory and apply it.

        t += dt

        if VISUALIZE is True:
            if v.is_running():
                v.sync()