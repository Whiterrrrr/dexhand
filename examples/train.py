import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from collections import OrderedDict
import torch.nn as nn
import argparse
from dexart.env.create_env import create_env
from dexart.env.task_setting import TRAIN_CONFIG, IMG_CONFIG, RANDOM_CONFIG
from stable_baselines3.common.torch_layers import PointNetImaginationExtractorGP
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO
from stable_baselines3.a2c import A2C
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3
from stable_baselines3.simple_callback import SimpleCallback
import torch
from OBAC.utilis.default_config import default_config
from OBAC.train import train_loop
import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_3d_policy_kwargs(extractor_name):
    feature_extractor_class = PointNetImaginationExtractorGP
    feature_extractor_kwargs = {"pc_key": "instance_1-point_cloud", "gt_key": "instance_1-seg_gt",
                                "extractor_name": extractor_name,
                                "imagination_keys": [f'imagination_{key}' for key in IMG_CONFIG['robot'].keys()],
                                "state_key": "state",
                                "origin_state":True}

    policy_kwargs = {
        "features_extractor_class": feature_extractor_class,
        "features_extractor_kwargs": feature_extractor_kwargs,
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        "activation_fn": nn.ReLU,
    }
    return policy_kwargs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='PPO')
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--freeze', dest='freeze', action='store_true', default=True)
    parser.add_argument('--task_name', type=str, default="laptop")
    parser.add_argument('--extractor_name', type=str, default="smallpn")
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--save_path', type=str, default=BASE_DIR)
    args = parser.parse_args()

    task_name = args.task_name
    extractor_name = args.extractor_name
    seed = args.seed if args.seed >= 0 else random.randint(0, 100000)
    pretrain_path = args.pretrain_path
    horizon = 200
    env_iter = args.iter * horizon * args.n
    print(f"freeze: {args.freeze}")

    rand_pos = RANDOM_CONFIG[task_name]['rand_pos']
    rand_degree = RANDOM_CONFIG[task_name]['rand_degree']


    def create_env_fn():
        seen_indeces = TRAIN_CONFIG[task_name]['seen']
        environment = create_env(task_name=task_name,
                                 use_visual_obs=True,
                                 use_gui=False,
                                 is_eval=False,
                                 pc_noise=True,
                                 index=seen_indeces,
                                 img_type='robot',
                                 rand_pos=rand_pos,
                                 rand_degree=rand_degree
                                 )
        return environment


    def create_eval_env_fn():
        unseen_indeces = TRAIN_CONFIG[task_name]['unseen']
        environment = create_env(task_name=task_name,
                                 use_visual_obs=True,
                                 use_gui=False,
                                 is_eval=True,
                                 pc_noise=True,
                                 index=unseen_indeces,
                                 img_type='robot',
                                 rand_pos=rand_pos,
                                 rand_degree=rand_degree)
        return environment


    env = SubprocVecEnv([create_env_fn] * args.workers, "spawn")  # train on a list of envs.
    eval_env = SubprocVecEnv([create_eval_env_fn] * args.workers, "spawn") 

    if args.alg == 'PPO':
        model = PPO("PointCloudPolicy", env, verbose=1,
                    n_epochs=args.ep,
                    n_steps=(args.n // args.workers) * horizon,
                    learning_rate=args.lr,
                    batch_size=args.bs,
                    seed=seed,
                    policy_kwargs=get_3d_policy_kwargs(extractor_name=extractor_name),
                    min_lr=args.lr,
                    max_lr=args.lr,
                    adaptive_kl=0.02,
                    target_kl=0.2,
                    tensorboard_log=f'/home/tsinghuaair/zhengkx/dexart-release/results/{task_name}/ppo/{extractor_name}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
                    )
        if pretrain_path is not None:
            state_dict: OrderedDict = torch.load(pretrain_path)
            model.policy.features_extractor.extractor.load_state_dict(state_dict, strict=False)
            print("load pretrained model: ", pretrain_path)

        rollout = int(model.num_timesteps / (horizon * args.n))

        # after loading or init the model, then freeze it if needed
        if args.freeze:
            model.policy.features_extractor.extractor.eval()
            for param in model.policy.features_extractor.extractor.parameters():
                param.requires_grad = False
            print("freeze model!")

        model.learn(
            total_timesteps=int(env_iter),
            reset_num_timesteps=False,
            iter_start=rollout,
            callback=SimpleCallback(model_save_freq=args.save_freq, model_save_path=args.save_path, rollout=0),
        )
    elif args.alg == 'A2C':
        model = A2C(
            "PointCloudPolicy", env, verbose=1,
            learning_rate=args.lr,
            n_steps=(args.n // args.workers) * horizon,
            seed=seed,
            policy_kwargs=get_3d_policy_kwargs(extractor_name=extractor_name),
            tensorboard_log=args.save_path
        )
        if pretrain_path is not None:
            state_dict: OrderedDict = torch.load(pretrain_path)
            model.policy.features_extractor.extractor.load_state_dict(state_dict, strict=False)
            print("load pretrained model: ", pretrain_path)
        
        if args.freeze:
            model.policy.features_extractor.extractor.eval()
            for param in model.policy.features_extractor.extractor.parameters():
                param.requires_grad = False
            print("freeze model!")

        model.learn(
            total_timesteps=int(env_iter),
            reset_num_timesteps=False,
            callback=SimpleCallback(model_save_freq=args.save_freq, model_save_path=args.save_path, rollout=0),
        )
    elif args.alg == 'SAC':
        model = SAC(
            "PointCloudPolicy", env, verbose=1,
            learning_rate=args.lr,
            seed=seed,
            policy_kwargs=get_3d_policy_kwargs(extractor_name=extractor_name),
            tensorboard_log=args.save_path,
            train_freq=(2, "episode")
        )
        if pretrain_path is not None:
            state_dict: OrderedDict = torch.load(pretrain_path)
            model.policy.actor.features_extractor.extractor.load_state_dict(state_dict, strict=False)
            model.policy.critic.features_extractor.extractor.load_state_dict(state_dict, strict=False)
            model.policy.critic_target.features_extractor.extractor.load_state_dict(state_dict, strict=False)
            print("load pretrained model: ", pretrain_path)
        
        if args.freeze:
            model.policy.actor.features_extractor.extractor.eval()
            model.policy.critic.features_extractor.extractor.eval()
            model.policy.critic_target.features_extractor.extractor.eval()
            for param in model.policy.actor.features_extractor.extractor.parameters():
                param.requires_grad = False
            for param in model.policy.critic.features_extractor.extractor.parameters():
                param.requires_grad = False
            for param in model.policy.critic_target.features_extractor.extractor.parameters():
                param.requires_grad = False
            print("freeze model!")

        model.learn(
            total_timesteps=int(env_iter),
            reset_num_timesteps=False,
            callback=SimpleCallback(model_save_freq=args.save_freq, model_save_path=args.save_path, rollout=0),
        )
    elif args.alg == 'TD3':
        model = TD3(
            "PointCloudPolicy", env, verbose=1,
            learning_rate=args.lr,
            seed=seed,
            policy_kwargs=get_3d_policy_kwargs(extractor_name=extractor_name),
            tensorboard_log=args.save_path,
            train_freq=(2, "episode")
        )
        if pretrain_path is not None:
            state_dict: OrderedDict = torch.load(pretrain_path)
            model.policy.actor.features_extractor.extractor.load_state_dict(state_dict, strict=False)
            model.policy.actor_target.features_extractor.extractor.load_state_dict(state_dict, strict=False)
            model.policy.critic.features_extractor.extractor.load_state_dict(state_dict, strict=False)
            model.policy.critic_target.features_extractor.extractor.load_state_dict(state_dict, strict=False)
            print("load pretrained model: ", pretrain_path)
        
        if args.freeze:
            model.policy.actor.features_extractor.extractor.eval()
            model.policy.actor_target.features_extractor.extractor.eval()
            model.policy.critic.features_extractor.extractor.eval()
            model.policy.critic_target.features_extractor.extractor.eval()
            for param in model.policy.actor.features_extractor.extractor.parameters():
                param.requires_grad = False
            for param in model.policy.actor_target.features_extractor.extractor.parameters():
                param.requires_grad = False
            for param in model.policy.critic.features_extractor.extractor.parameters():
                param.requires_grad = False
            for param in model.policy.critic_target.features_extractor.extractor.parameters():
                param.requires_grad = False
            print("freeze model!")

        model.learn(
            total_timesteps=int(env_iter),
            reset_num_timesteps=False,
            callback=SimpleCallback(model_save_freq=args.save_freq, model_save_path=args.save_path, rollout=0),
        )
    elif args.alg == 'OBAC':
        train_loop(env, get_3d_policy_kwargs(extractor_name), default_config, msg="default", task=args.task_name, pretrain_path=args.pretrain_path, freeze_encoder=True, eval_env=eval_env)
    else:
        raise NotImplementedError(f'alg {args.alg} has not been implemented.')

