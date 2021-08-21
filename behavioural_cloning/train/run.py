import os
import time
import argparse
from behavioural_cloning.train.trainer import Trainer

class MainTrainer:
    
    def __init__(self, params):

        agent_params = {
            "n_layers" : params["n_layers"],
            "size" : params["size"],
            "learning_rate" : params["learning_rate"],
            "max_replay_buffer_size" : params["max_replay_buffer_size"]
        }

        agent_params.items

        self.params = dict(filter(lambda key: key != agent_params.keys, params.items()))
        self.params["agent_params"] = agent_params

        self.trainer = Trainer(self.params) # TODO Check implementation is finished

    def run(self):
        self.trainer.execute_training(self.params["n_iter"],
                                      self.trainer.agent.policy,
                                      self.trainer.agent.policy,
                                      initial_expert_data=self.params["expert_data"],
                                      do_dagger=self.params["do_dagger"]) #TODO add loaded expert policy

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True) 
    parser.add_argument('--expert_data', '-ed', type=str, required=True)
    parser.add_argument('--env_name', '-env', type=str, help='choices: Ant-v2, Humanoid-v2, Walker-v2, HalfCheetah-v2, Hopper-v2', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, help='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true', default=True)
    parser.add_argument('--ep_len', type=int, default=40)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=1000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--max_n_video', type=int, default=2)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    params = vars(args)

    if params["do_dagger"]:
        logdir_prefix = "bc_dagger_"
        assert (params["n_iter"] > 1), ("DAgger needs more than 1 iteration.\nCheck the parameter --n_iter--")

    else:
        logdir_prefix = "bc_"
        assert (params["n_iter"] == 1), ("Vanilla BC collects expert data once.\nCheck the parameter --n_iter--")

    log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../log")

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logfile = logdir_prefix + params["exp_name"] + "_" + params["env_name"] + "_" + time.strftime("%d-%m-%Y_%H-%M-%S") 
    logfile = os.path.join(log_path, logfile)
    if not os.path.exists(logfile):
        os.makedirs(logfile)

    params["logdir"] = logfile

    trainer = MainTrainer(params)
    trainer.run()
 
if __name__ == "__main__":
    main()