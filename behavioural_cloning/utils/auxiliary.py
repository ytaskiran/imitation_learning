import time
import numpy as np
import torch


class Path:

    def __init__(self, observations, image_obs, actions,
                 rewards, next_obs, terminals):
    
        if image_obs != []:
            image_obs = np.stack(image_obs, axis=0)

        self.observations = np.array(observations, dtype=np.float32)
        self.image_obs = np.array(image_obs, dtype=np.uint8)
        self.actions = np.array(actions, dtype=np.float32)
        self.rewards = np.array(rewards, dtype=np.float32)
        self.next_obs = np.array(next_obs, dtype=np.float32)
        self.terminals = np.array(terminals, dtype=np.float32)

    def __len__(self):
        return len(self.rewards)



def sample_single_trajectory(env, policy, max_path_length, render=False, 
                             render_mode=("rgb_array")):
    
    observations = []
    actions = []
    rewards = []
    next_obs = []
    image_obs  = []
    terminals = []

    steps = 0


    while True:

        if render:
            if "rgb_array" in render_mode:
                if hasattr(env, "sim"):
                    image_obs.append(env.sim.render(camera_name="track", width=800, height=600)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))

            if "human" in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)

        observation = env.reset()
        observations.append(observation)
        action = policy.get_action(observation)[0] # TODO not yet implemented
        actions.append(action)

        observation, reward, done, _ = env.step(action)

        steps += 1
        next_obs.append(observation)
        rewards.append(reward)

        is_rollout_done = done or (steps >= max_path_length)
        terminals.append(is_rollout_done)

        if is_rollout_done:
            break
    
    return Path(observations, image_obs, actions, rewards, next_obs, terminals)
    

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length,
                        render=False, render_mode=("rgb_array")):

    timesteps_this_batch = 0
    paths = []

    while timesteps_this_batch < min_timesteps_per_batch:
                            
        paths.append(sample_single_trajectory(env, policy, max_path_length, 
                                              render, render_mode))

        timesteps_this_batch += len(paths[-1])

    return paths, timesteps_this_batch

def sample_trajectories_video(env, policy, n, max_path_length, render=True,
                              render_mode=("rgb_array")):
    
    paths = []

    for i in range(n):
        paths.append(sample_single_trajectory(env, policy, max_path_length,
                                              render, render_mode))
    
    return paths


def concatenate_rollouts(paths):

    observations = np.concatenate([path.observations for path in paths])
    actions = np.concatenate([path.actions for path in paths])
    rewards = np.concatenate([path.rewards for path in paths])
    next_obs = np.concatenate([path.next_obs for path in paths])
    terminals = np.concatenate([path.terminals for path in paths])

    return observations, actions, rewards, next_obs, terminals


device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu_id))
        print(f"GPU activated, ID: {gpu_id}")
    else:
        device = torch.device("cpu")
        print("GPU cannot be activated, using CPU instead")


def convert_to_tensor(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def convert_to_numpy(tensor):
    return tensor.to(device).detach().numpy()

