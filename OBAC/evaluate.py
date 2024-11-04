import os
import sys
from OBAC.utilis.config import Config
from OBAC.utilis.default_config import default_config
from OBAC.model.algo import OBAC
from copy import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from OBAC.utilis.video import recorder
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import argparse
from dexart.env.task_setting import TRAIN_CONFIG, RANDOM_CONFIG
import numpy as np
from dexart.env.create_env import create_env
from examples.train import get_3d_policy_kwargs


def display_model(result_path, result_epoch, policy_kwargs, config_path=None, env=None, video_path=None, task_name='default'):
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Error, model file {result_path} not exists")
    if config_path is None:
        config_path = result_path + '/' + 'config.log'
    
    config = Config().load_saved(config_path)
    config.device = "cpu"
    eval_recoder = recorder(video_path)
    eval_recoder.init(f'obac_{task_name}.mp4', enabled=True)
    # env = gym.make(config.env_name)

    # Agent
    agent = OBAC(env, env.action_space, True, policy_kwargs, config)
    # agent = OBAC(env.observation_space.shape[0], env.action_space, config)
    agent.load_checkpoint(result_path, result_epoch)

    # test agent
    avg_reward = 0.
    eval_episodes=10
    for _  in range(eval_episodes):
        obs = env.reset()
        state = agent.extract_features(obs).detach().cpu().numpy()
        eval_recoder.record(env.render()[0][:,:,:3])
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)

            next_obs, reward, done, info = env.step(action) # Step
            done, reward, info = done[0], reward[0], info[0]
            # env.render()
            eval_recoder.record(env.render()[0][:,:,:3])
            episode_reward += reward
            next_state = agent.extract_features(next_obs).detach().cpu().numpy()
            state = next_state
        avg_reward += episode_reward
    avg_reward /= eval_episodes
    eval_recoder.release('obac_avg_reward%d.mp4'%(int(avg_reward)))

    print("----------------------------------------")
    print("Env: {}, Test Episodes: {}, Avg. Reward: {}".format(task_name, eval_episodes, round(avg_reward, 2)))
    print("----------------------------------------")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--use_test_set', dest='use_test_set', action='store_true', default=False)
    args = parser.parse_args()
    task_name = args.task_name
    use_test_set = args.use_test_set
    checkpoint_path = args.checkpoint_path
    config_path = args.config_path
    video_path = args.video_path

    if use_test_set:
        indeces = TRAIN_CONFIG[task_name]['unseen']
        print(f"using unseen instances {indeces}")
    else:
        indeces = TRAIN_CONFIG[task_name]['seen']
        print(f"using seen instances {indeces}")

    rand_pos = RANDOM_CONFIG[task_name]['rand_pos']
    rand_degree = RANDOM_CONFIG[task_name]['rand_degree']
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

    env = SubprocVecEnv([create_eval_env_fn], "spawn")

    # rand_pos = RANDOM_CONFIG[task_name]['rand_pos']
    # rand_degree = RANDOM_CONFIG[task_name]['rand_degree']
    # env = create_env(
    #     task_name=task_name,
    #     use_visual_obs=True,
    #     use_gui=False,
    #     is_eval=True,
    #     pc_noise=True,
    #     pc_seg=True,
    #     index=indeces,
    #     img_type='robot',
    #     rand_pos=rand_pos,
    #     rand_degree=rand_degree
    # )
    # eval_env = SubprocVecEnv([env] * 1, "spawn")
    policy_kwargs=get_3d_policy_kwargs(extractor_name='smallpn')
    display_model(checkpoint_path, 'best', policy_kwargs, config_path, env, video_path, task_name)