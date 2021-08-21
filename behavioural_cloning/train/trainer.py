import numpy as np
import torch
import gym
import time
import pickle

from behavioural_cloning.utils.auxiliary import *
from behavioural_cloning.agent.bc_agent import BCAgent

NUM_VIDEO = 2

class Trainer:
    
    def __init__(self, params):

        self.params = params
        # self.logger = Logger(self.params["logdir"]) TODO Implement Logger class at utils

        seed = self.params["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        init_gpu(use_gpu=not self.params["no_gpu"],
                 gpu_id=self.params["gpu_id"])

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

        self.agent = BCAgent(self.env, self.params["agent_params"])

    
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
                paths = self.relabel_with_expert(paths, expert_policy)

            self.agent.add_to_buffer(paths)

            training_logs = self.train_agent()

            if self.log_video or self.log_metrics:
                print("Performing logging...")
                #TODO define perform logging function


            
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


    def relabel_with_expert(self, paths, expert_policy):
        print("Relabelling observations with expert policy for DAgger")
        
        for path in paths:
            path.actions = expert_policy.get_action(path.observations)

        return paths


    def train_agent(self):
        print("Training the agent using sampled data from experience replay buffer")
        train_logs = []

        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, rew_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])

            train_log = self.agent.train(ob_batch, ac_batch)
            train_logs.append(train_log)
        
        return train_logs



    def log(self, itr, paths, eval_policy, train_video_paths, training_logs):
        pass


 












