import os
import torch
# import gym
import numpy as np
from OBAC.utilis.config import ARGConfig
from OBAC.utilis.default_config import default_config
from OBAC.model.algo import OBAC
from OBAC.utilis.Replaybuffer import ReplayMemory
from OBAC.utilis.video import recorder
import datetime
import itertools
from torch.utils.tensorboard import SummaryWriter
# from dm_control import suite
# from envs.meta_world_env import make_env
# import robohive


def evaluation(agent, env, total_numsteps, writer, best_reward, video_path=None, config=None):
    avg_reward = 0.
    avg_success = 0.
    if video_path is not None:
        eval_recoder = recorder(video_path)
        eval_recoder.init(f'{total_numsteps}.mp4', enabled=True)
    else:
        eval_recoder = None
    for _  in range(config.eval_times):
        obs = env.reset()
        state = agent.extract_features(obs).detach().cpu().numpy()
        # state, _ = agent.get_latent(obs)
        # state = state.detach().cpu().numpy()

        if eval_recoder is not None:
            eval_recoder.record(env.render())
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)

            next_obs, reward, done, info = env.step(action) # Step
            done, reward, info = done[0], reward[0], info[0]
            if eval_recoder is not None:
                eval_recoder.record(env.render())
            episode_reward += reward
            if done and info.get("TimeLimit.truncated"):
                done = False
            next_state = agent.extract_features(next_obs).detach().cpu().numpy()
            # next_state, _ = agent.get_latent(next_obs)
            # next_state = next_state.detach().cpu().numpy()
            state = next_state
        avg_reward += episode_reward
        if 'solved' in info.keys():
            avg_success += float(info['solved'])
        elif 'success' in info.keys():
            avg_success += float(info['success'])
    avg_reward /= config.eval_times
    avg_success /= config.eval_times

    if eval_recoder is not None and avg_reward >= best_reward:
        eval_recoder.release('%d_%d.mp4'%(total_numsteps, int(avg_reward)))

    writer.add_scalar('test/avg_reward', avg_reward, total_numsteps)
    writer.add_scalar('test/avg_success', avg_success, total_numsteps)

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}, Avg. Success: {}".format(config.eval_times, round(avg_reward, 2), round(avg_success, 2)))
    print("----------------------------------------")
    
    return avg_reward

def train_loop(env, policy_kwargs, config, msg = "default", task='default', pretrain_path=None, freeze_encoder=False, eval_env=None):
    # set seed
    # env = gym.make(config.env_name)
    # env = make_env(config)
    # env.seed(config.seed)
    
    env.action_space.seed(config.seed)
    if eval_env == None:
        eval_env = env
    eval_env.action_space.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Agent
    # agent = OBAC(env.observation_space.shape[0], env.action_space, policy_kwargs, config)
    agent = OBAC(env, env.action_space, policy_kwargs, config)
    # import ipdb; ipdb.set_trace()
    if pretrain_path is not None:
        state_dict = torch.load(pretrain_path)
        agent.features_extractor.extractor.load_state_dict(state_dict, strict=False)
        print("load pretrained model: ", pretrain_path)
    if freeze_encoder:
        agent.features_extractor.extractor.eval()
        for param in agent.features_extractor.extractor.parameters():
            param.requires_grad = False
        print("freeze model!")

    result_path = './results/{}/{}/{}/{}_{}_{}_{}'.format(task, config.algo, msg, 
                                                      datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                                                      config.policy, config.seed, 
                                                      "autotune" if config.automatic_entropy_tuning else "")

    checkpoint_path = result_path + '/' + 'checkpoint'
    video_path = result_path + '/eval_video'
    
    # training logs
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    with open(os.path.join(result_path, "config.log"), 'w') as f:
        f.write(str(config))
    writer = SummaryWriter(result_path)

    memory = ReplayMemory(config.replay_size, config.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    best_reward = -1e6
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        obs = env.reset()
        state = agent.extract_features(obs).detach().cpu().numpy()
        # state_pi, _ = agent.get_latent(obs)
        # state_pi = state_pi.detach().cpu().numpy()

        while not done:
            if config.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
                action = np.array([action])
            else:
                action = agent.select_action(state)  # Sample action from policy

            if config.start_steps <= total_numsteps:
                # Number of updates per step in environment
                for i in range(config.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, value_loss, policy_loss, ent_loss, alpha, q_pi, q_behavior_pi = agent.update_parameters(memory, config.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/offline_value', value_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('parameter/alpha', alpha, updates)
                    writer.add_scalar('parameter/q_current_pi', q_pi, updates)
                    writer.add_scalar('parameter/q_behavior_pi', q_behavior_pi, updates)
                    writer.add_scalar('parameter/q_diff', q_pi - q_behavior_pi, updates)
                    updates += 1
            next_obs, reward, done, info = env.step(action) # Step
            done, info = done[0], info[0]
            if done and info.get("TimeLimit.truncated"):
                done = False
            next_state = agent.extract_features(next_obs).detach().cpu().numpy()
            # next_state_pi, _ = agent.get_latent(next_obs)
            # state_pi = next_state_pi.detach().cpu().numpy()
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward[0]

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == 250 else float(not done)
            memory.push(state, action[0], reward[0], next_state, mask) # Append transition to memory
            state = next_state
            # memory.push(obs, action[0], reward[0], next_obs, mask) # Append transition to memory

            # test agent
            if total_numsteps % config.eval_numsteps == 0 and config.eval is True:
                video_path = None
                avg_reward = evaluation(agent, eval_env, total_numsteps, writer, best_reward, video_path, config)
                if avg_reward >= best_reward and config.save is True:
                    best_reward = avg_reward
                    agent.save_checkpoint(checkpoint_path, 'best')

        if total_numsteps > config.num_steps:
            break
        
        if i_episode % 1 == 0:
            writer.add_scalar('train/reward', episode_reward, total_numsteps)
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
            print()
    env.close() 

    

if __name__ == "__main__":
    arg = ARGConfig()
    arg.add_arg("task", "humanoid_h1hand-walk-v0", "Environment name")
    arg.add_arg("device", 0, "Computing device")
    arg.add_arg("algo", "OBAC", "choose algo")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("seed", np.random.randint(1000), "experiment seed")
    arg.parser()

    config = default_config
    
    config.update(arg)

    print(f">>>> Training {config.algo} on {config.task} environment, on {config.device}")
    train_loop(config, msg=config.tag)