import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import copy
from .utils import soft_update, hard_update
from .model import GaussianPolicy, QNetwork, DeterministicPolicy, ValueNetwork
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import (
    MlpExtractor,
    NatureCNN,
)
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class OBAC(object):
    def __init__(self, env, action_space, freeze_encoder, policy_kwargs,args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.quantile = args.quantile
        self.bc_weight = args.bc_weight
        self.freeze_encoder = freeze_encoder
        self.device = torch.device("cuda:{}".format(str(args.device)) if args.cuda else "cpu")
        self.features_extractor = policy_kwargs['features_extractor_class'](env.observation_space, **policy_kwargs['features_extractor_kwargs']).to(self.device)
        self.features_dim = self.features_extractor.features_dim
        # self.features_dim = 320
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.env = env
        if policy_kwargs['features_extractor_class'] == NatureCNN:
            self.net_arch = []
            self.activation_fn = torch.nn.ReLU
        else:
            # self.net_arch = [dict(pi=[64, 64], vf=[64, 64])]
            self.net_arch = [128, 128]
            self.activation_fn = torch.nn.ReLU
        self._build_mlp_extractor()
        self.pi_input = self.features_dim
        self.v_input = self.features_dim
        
        self.critic = QNetwork(self.v_input, action_space.shape[0], args.hidden_size, preprocessing=False).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(self.v_input, action_space.shape[0], args.hidden_size, preprocessing=False).to(self.device)
        hard_update(self.critic_target, self.critic)
        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(self.pi_input, action_space.shape[0], args.hidden_size, action_space, preprocessing=False).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(self.v_input, action_space.shape[0], args.hidden_size, action_space, preprocessing=False).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        
        self.critic_buffer = QNetwork(self.v_input, action_space.shape[0], args.hidden_size, preprocessing=False).to(device=self.device)
        self.critic_buffer_optim = Adam(self.critic_buffer.parameters(), lr=args.lr)
        hard_update(self.critic_buffer, self.critic)

        self.critic_target_buffer = QNetwork(self.v_input, action_space.shape[0], args.hidden_size, preprocessing=False).to(self.device)
        hard_update(self.critic_target_buffer, self.critic_buffer)

        self.V_critic_buffer = ValueNetwork(self.v_input, args.hidden_size, preprocessing=False).to(device=self.device)
        self.V_critic_buffer_optim = Adam(self.V_critic_buffer.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        obs_batch, action_batch, reward_batch, next_obs_batch, mask_batch = memory.sample(batch_size=batch_size)
        # state_pi_batch, state_value_batch = self.get_latent_batch(obs_batch)
        # next_state_pi_batch, next_state_value_batch = self.get_latent_batch(next_obs_batch)
        # obs_batch = self.extract_features_batch(obs_batch).detach()
        # next_obs_batch = self.extract_features_batch(next_obs_batch).detach()
        # state_pi_batch = torch.FloatTensor(state_pi_batch).to(self.device)
        # state_value_batch = torch.FloatTensor(state_value_batch).to(self.device)
        if not self.freeze_encoder:
            obs_batch = self.batch2dict(obs_batch)
            next_obs_batch = self.batch2dict(next_obs_batch)
            obs_batch = self.extract_features(obs_batch)
            next_obs_batch = self.extract_features(next_obs_batch)
        else:
            obs_batch = torch.FloatTensor(obs_batch).to(self.device).squeeze()
            next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device).squeeze()
        action_batch = torch.FloatTensor(action_batch).to(self.device).squeeze()
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_obs_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_obs_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # Compute the target Q value for current policy
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # compute the Q loss for current policy
        qf1, qf2 = self.critic(obs_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value) 
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        # Compute the target Q value for behavior policy
        vf_pred = self.V_critic_buffer(obs_batch.detach())
        target_Vf_pred = self.V_critic_buffer(next_obs_batch.detach())
        next_q_value_buffer = reward_batch + mask_batch * self.gamma * target_Vf_pred
        
        # compute the Q loss for behavior policy
        qf1_buffer, qf2_buffer = self.critic_buffer(obs_batch.detach(), action_batch)
        qf_buffer = torch.min(qf1_buffer, qf2_buffer).mean()   # compute the Q value for (s,a) pair under the behavior policy
        qf1_buffer_loss = F.mse_loss(qf1_buffer, next_q_value_buffer)  
        qf2_buffer_loss = F.mse_loss(qf2_buffer, next_q_value_buffer)
        qf_buffer_loss = qf1_buffer_loss + qf2_buffer_loss
        
        # compute the V loss for behavior policy
        q_pred_1, q_pred_2 = self.critic_target_buffer(obs_batch.detach(), action_batch)
        q_pred = torch.min(q_pred_1, q_pred_2)
        vf_err = q_pred - vf_pred
        vf_sign = (vf_err < 0).float()
        vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
        vf_loss = (vf_weight * (vf_err ** 2)).mean()
        
        # compute action by current policy
        pi, log_pi, _ = self.policy.sample(obs_batch.detach())
        # estimate the Q value 
        qf1_pi, qf2_pi = self.critic(obs_batch.detach(), pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi) # compute the Q value for (s,a) pair under the current policy
        qf_pi = min_qf_pi.mean()
        
        if updates == 0:
            self.policy_loss = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_loss = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_tlogs = torch.zeros(1, requires_grad=True, device=self.device)
        
        # if self.freeze_encoder:  
            # update Q value of current policy
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        # update Q value of behavior policy
        self.critic_buffer_optim.zero_grad()
        qf_buffer_loss.backward()
        self.critic_buffer_optim.step()
        
        # update V value of behavior policy
        self.V_critic_buffer_optim.zero_grad()
        vf_loss.backward()
        self.V_critic_buffer_optim.step()
        
        if updates % self.target_update_interval == 0:
            if qf_pi >= qf_buffer:  # means current policy can surpass behavior policy; or current policy can get exploration bonus
                policy_loss = (self.alpha * log_pi - min_qf_pi).mean()
            else:
                log_density = self.policy.get_log_density(obs_batch, action_batch)
                log_density = torch.clamp(log_density, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
                policy_loss = (self.alpha * log_pi - self.bc_weight * log_density - min_qf_pi).mean()
            
            # if self.freeze_encoder:
                # update policy
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()
            # else:
            #     online_loss = policy_loss + qf_loss
            #     self.policy_optim.zero_grad()
            #     self.critic_optim.zero_grad()
            #     online_loss.backward()
            #     self.policy_optim.step()
            #     self.critic_optim.step()
            
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone() # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs
            
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.critic_target_buffer, self.critic_buffer, self.tau)
            self.policy_loss = copy.copy(policy_loss)
            self.alpha_loss = copy.copy(alpha_loss)
            self.alpha_tlogs = copy.copy(alpha_tlogs)
            
        return qf1_loss.item(), qf2_loss.item(), vf_loss.item(), self.policy_loss.item(), self.alpha_loss.item(), self.alpha_tlogs.item(), qf_pi.item(), qf_buffer.item()
    
    # Save model parameters
    def save_checkpoint(self, path, i_episode):
        ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict(),
                    'critic_buffer_state_dict': self.critic_buffer.state_dict(),
                    'critic_target_buffer_state_dict': self.critic_target_buffer.state_dict(),
                    'critic_buffer_optimizer_state_dict': self.critic_buffer_optim.state_dict(),
                    'V_critic_buffer_state_dict': self.V_critic_buffer.state_dict(),
                    'V_critic_buffer_optimizer_state_dict': self.V_critic_buffer_optim.state_dict()
                    },
                    ckpt_path)
    
    # Load model parameters
    def load_checkpoint(self, path, i_episode, evaluate=False):
        ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.critic_buffer.load_state_dict(checkpoint['critic_buffer_state_dict'])
            self.critic_target_buffer.load_state_dict(checkpoint['critic_target_buffer_state_dict'])
            self.critic_buffer_optim.load_state_dict(checkpoint['critic_buffer_optimizer_state_dict'])
            self.V_critic_buffer.load_state_dict(checkpoint['V_critic_buffer_state_dict'])
            self.V_critic_buffer_optim.load_state_dict(checkpoint['V_critic_buffer_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
                self.critic_buffer.eval()
                self.critic_target_buffer.eval()
                self.V_critic_buffer.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
                self.critic_buffer.train()
                self.critic_target_buffer.train()
                self.V_critic_buffer.train()
                
    def get_latent_batch(self, obs_batch):
        latenr_pi_batch, latenr_value_batch = [], []
        for obs in obs_batch:
            # obs = obs_as_tensor(obs, self.device)
            feature = self.extract_features(obs)
            latenr_pi, latent_value = self.mlp_extractor(feature)
            latenr_pi_batch.append(latenr_pi)
            latenr_value_batch.append(latent_value)
        return torch.stack(latenr_pi_batch).squeeze(), torch.stack(latenr_value_batch).squeeze()
        
    def get_latent(self, obs):
        # obs = obs_as_tensor(obs, self.device)
        feature = self.extract_features(obs)
        return self.mlp_extractor(feature)

    def extract_features_batch(self, obs_batch):
        features = []
        for obs in obs_batch:
            # obs = obs_as_tensor(obs, self.device)
            feature = self.extract_features(obs)
            features.append(feature)
        return torch.stack(features).squeeze()
    
    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        obs = obs_as_tensor(obs, self.device)
        preprocessed_obs = preprocess_obs(obs, self.env.observation_space, normalize_images=True)
        return self.features_extractor(preprocessed_obs)
    
    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
        
    def batch2dict(self, obs):
        result = {}
        keys = obs[0].keys()
        for key in keys:
            result[key] = np.concatenate([item[key] for item in obs], axis=0)
        return result