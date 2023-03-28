import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
from shutil import copyfile
import time
from distutils.util import strtobool

import gym
import numpy as np
# import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import yaml

# ROS
import rospy
from ur3e_openai.common import load_environment, log_ros_params, clear_gym_params, load_ros_params
from ur3e_openai.initialize_logger import initialize_logger
from ur3e_openai.prepare_output_dir import prepare_output_dir


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument('--save-summary-interval', type=int, default=int(1e3),
        help='Interval to save summary')
    parser.add_argument("--save-model-interval", type=int, default=int(5e3),
        help="Interval to save model")    
    parser.add_argument("--wandb-project-name", type=str, default="trufus",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HopperBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--ros-params-file", type=str, default="",
        help="(Optional) for multiple test, specify the path to the unique parameters")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed) # depricated
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args = parse_args()

    rospy.init_node('ur3e_tf2rl',
                    anonymous=True,
                    log_level=rospy.INFO)

    clear_gym_params('ur3e_gym')
    clear_gym_params('ur3e_force_control')

    param_file = None

    print("Env ID", args.env_id)
    if int(args.env_id) == 0:
        args.dir_suffix = "pih_m24"
        param_file = "simulation/force_control/peg_in_hole_m24.yaml"
    else:
        raise Exception("invalid env_id")

    if args.ros_params_file:
        # Load common parameters
        common_ros_param_path = load_ros_params(rospackage_name="ur3e_rl",
                                                rel_path_from_package_to_file="config",
                                                yaml_file_name='simulation/force_control/test/common.yaml')
        # Then load specific parameters
        ros_param_path = load_ros_params(rospackage_name="ur3e_rl",
                                         rel_path_from_package_to_file="config",
                                         yaml_file_name=args.ros_params_file)
    else:
        ros_param_path = load_ros_params(rospackage_name="ur3e_rl",
                                         rel_path_from_package_to_file="config",
                                         yaml_file_name=param_file)

    episode_max_steps = rospy.get_param("ur3e_gym/steps_per_episode", 200)

    ros_env_id = rospy.get_param('ur3e_gym/env_id')
    load_environment(ros_env_id,
                     max_episode_steps=episode_max_steps,
                     register_only=True)

    ##### Start Training #####

    experiment_name = f"{ros_env_id}__{args.exp_name}__{int(time.time())}"
    outdir = f"runs/{experiment_name}"
    prepare_output_dir(args=args, user_specified_dir=outdir)

    if args.ros_params_file:
        copyfile(common_ros_param_path, outdir + "/common_ros_gym_env_params.yaml")
        copyfile(ros_param_path, outdir + "/ros_gym_env_params.yaml")
    else:
        copyfile(ros_param_path, outdir + "/ros_gym_env_params.yaml")

    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
        )

        wandb.save(f"runs/{experiment_name}/*", base_path=f"runs/{experiment_name}", policy="now")

    writer = SummaryWriter(outdir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    rospy.set_param('ur3e_gym/output_dir', outdir)
    log_ros_params(outdir)
    if args.ros_params_file:
        copyfile(common_ros_param_path, outdir + "/common_ros_gym_env_params.yaml")
        copyfile(ros_param_path, outdir + "/ros_gym_env_params.yaml")
    else:
        copyfile(ros_param_path, outdir + "/ros_gym_env_params.yaml")

    logger = initialize_logger(log_tag="cleanrl", filename=outdir + "/training_log.log")

    p_seed = rospy.get_param("ur3e_gym/seed", args.seed)
    p_batch_size = rospy.get_param("ur3e_gym/batch_size", args.batch_size)
    p_policy_lr = rospy.get_param("ur3e_gym/policy_lr", args.policy_lr)
    p_alpha = rospy.get_param("ur3e_gym/alpha", args.alpha)
    p_autotune = rospy.get_param("ur3e_gym/auto_alpha", args.autotune)
    p_warmup = rospy.get_param("ur3e_gym/warmup", args.learning_starts)

    # TRY NOT TO MODIFY: seeding
    random.seed(p_seed)
    np.random.seed(p_seed)
    torch.manual_seed(p_seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(ros_env_id, p_seed, 0, args.capture_video, experiment_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=p_policy_lr)

    # Automatic entropy tuning
    if p_autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = p_alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    if args.track and wandb.run.resumed:
        api = wandb.Api()
        run = api.run(f"{run.entity}/{run.project}/{run.id}")
        model = run.file("last_checkpoint.pt")
        model.download(f"runs/{experiment_name}/")
        checkpoint = torch.load(f"runs/{experiment_name}/last_checkpoint.pt", map_location=device)
        actor.load_state_dict(checkpoint["actor"])
        qf1.load_state_dict(checkpoint["qf1"])
        qf2.load_state_dict(checkpoint["qf2"])
        actor.eval()
        print(f"Model training resumed")

    st = rospy.get_time()
    
    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < p_warmup:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                # Compute episodes metrics
                time_per_step = (rospy.get_time() - st) / info["episode"]["l"]

                # Reset variables
                st = rospy.get_time()

                logger.info(f"Total steps:{global_step:>10}, # steps:{info['episode']['l']:>4}, episodic_return:{info['episode']['r']:>8.2f}, TPS:{time_per_step:>5.3f}")
                writer.add_scalar("Common/training_return", info["episode"]["r"], global_step)
                writer.add_scalar("Common/training_episode_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > p_warmup:
            data = rb.sample(p_batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % args.save_model_interval == 0:
                if not os.path.exists(f"runs/{experiment_name}"):
                    os.makedirs(f"runs/{experiment_name}")
                checkpoint_data = {
                    "actor": actor.state_dict(),
                    "qf1": qf1.state_dict(),
                    "qf2": qf2.state_dict(),
                }
                torch.save(checkpoint_data, f"runs/{experiment_name}/last_checkpoint.pt")
                torch.save(checkpoint_data, f"runs/{experiment_name}/checkpoint-step-{global_step}.pt")
                if args.track:
                    wandb.save(f"runs/{experiment_name}/last_checkpoint.pt", base_path=f"runs/{experiment_name}", policy="now")
                    
            if global_step % args.save_summary_interval == 0:
                writer.add_scalar("SAC/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("SAC/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("SAC/critic_loss", qf1_loss.item(), global_step)
                writer.add_scalar("SAC/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("SAC/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("SAC/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("SAC/alpha", alpha, global_step)
                if p_autotune:
                    writer.add_scalar("SAC/alpha_loss", alpha_loss, global_step)

                # writer.add_scalar("Common/SPS", int(global_step / (rospy.get_time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("SAC/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()

    if args.track:
        wandb.save(f"runs/{experiment_name}/*.log", base_path=f"runs/{experiment_name}", policy="now")
