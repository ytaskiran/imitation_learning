import numpy as np
import torch
import gym
import time
import pickle

from imitation_learning.utils.auxiliary import sample_trajectories, sample_trajectories_video

NUM_VIDEO = 2

class Trainer:
    
    def __init__(self, params):

        self.params = params
        # self.logger = Logger(self.params["logdir"]) TODO Implement Logger class at utils

        seed = self.params["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        # pytorch utils gpu initalization TODO

        ##########
        ## ENV
        ##########

        # Make the gym environment
        self.env = gym.make(self.params["env_name"])
        self.env.seed(seed)

        # Maximum episode length
        self.episode_len = self.params["ep_len"] or self.env.spec.max_episode_steps

        # Number of videos to save at tensorboard
        self.video_num = NUM_VIDEO

        # Is the action space of the environment continuous or discrete?
        is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params["agent_params"]["discrete"] = is_discrete

        # Observation and Action Space sizes
        ob_space_size = self.env.observation_space.shape[0]
        act_space_size = self.env.action_space.n if is_discrete else self.env.action_space.shape[0]
        self.params["agent_params"]["observation_size"] = ob_space_size
        self.params["agent_params"]["action_size"] = act_space_size

        if "model" in dir(self.env):
            self.fps = 1 / self.env.model.opt.timestep
        else:
            self.fps = self.env.env.metadata["video.frames_per_second"]

        ##########
        ## AGENT
        ##########

        # agent = self.params["agent_class"] TODO not yet implemented in run.py, I think pass directly instead of parameter
        # self.agent = agent(self.env, self.params["agent_params"]) TODO implement agent class in agent folder

    
    def execute_training(self, n_iter, current_policy, eval_policy, 
                         initial_expert_data=None, do_dagger=True, 
                         start_dagger_at=1, expert_policy=None):

        self.env_total_steps = 0
        self.beginning = time.time()
        batch_size = self.params["batch_size"]
        log_video_freq = self.params["video_log_freq"]
        log_metrics_freq = self.params["scalar_log_freq"]

        for i in range(n_iter):
            print(f"\n\n*********** ITERATION {i} **********")

            if (i % log_video_freq == 0) and (log_video_freq != -1):
                self.log_video = True
            else:
                self.log_video = False

            if i % log_metrics_freq == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            paths, env_steps_this_batch, train_video_paths = self.collect_training_trajectories(
                                                                i, 
                                                                initial_expert_data,
                                                                current_policy, 
                                                                batch_size)

            self.env_total_steps += env_steps_this_batch

            if do_dagger and i >= start_dagger_at:
                paths = self.relabel_with_expert(expert_policy, paths)

            # self.agent.add_to_replay_buffer(paths) TODO examine the function and uncomment after initializing the agent class


            
    def collect_training_trajectories(self, itr, initial_expert_data,
                                      current_policy, batch_size):
        
        if itr == 0:
            with open(initial_expert_data, "rb") as f:
                print("Loading expert trajectories")
                loaded_expert_paths = pickle.load(f)

            return loaded_expert_paths, 0, None

        print("Running the current policy to collect data for training")

        paths, env_steps_this_batch = sample_trajectories(self.env, 
                                                          current_policy,
                                                          batch_size,
                                                          self.episode_len)

        video_paths = None

        if self.log_video:
            print("Collecting training rollouts for video saving...")
            video_paths = sample_trajectories_video(self.env,
                                                    current_policy,
                                                    self.video_num,
                                                    self.episode_len)

        return paths, env_steps_this_batch, video_paths


    def relabel_with_expert(self, expert_policy, paths):
        print("Relabelling observations with expert policy for DAgger")
        
        for path in paths:
            path.actions = expert_policy.get_action(path.observations)

        return paths

    def train_agent(self):
        pass

    def log(self, itr, paths, eval_policy, train_video_paths, training_logs):
        pass


 












