from time import time

import os

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer

from memory_profiler import profile

class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self.checkpoint_loaded = False
		print("self.cfg.checkpoint", self.cfg.checkpoint)
		if self.cfg.checkpoint is not None and os.path.exists(self.cfg.checkpoint):
		# if os.path.exists(self.cfg.checkpoint):
			print("loading checkpoint: ", self.cfg.checkpoint)
			self.agent.load(self.cfg.checkpoint)
			self.checkpoint_loaded = True
			print("Model loaded from checkpoint")
		else:
			print("Train from scratch")
	
	#@profile
	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)
	#@profile
	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			print("eval episode", i)
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)

	#@profile
	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			# print("online_trainer.py obs is dict")
			# print("online_trainer.py obs[image]", obs['image'].shape)
			# print("online_trainer.py obs[state]", obs['state'].shape)
			# obs['image'] = obs['image'].unsqueeze(0).cpu()
			obs['state'] = obs['state'].unsqueeze(0).cpu()

			# print("online_trainer.py after obs[image]", obs['image'].shape)
			# print("online_trainer.py after obs[state]", obs['state'].shape)

			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			# print("online_trainer.py obs is not dict")
			# print("online_trainer.py obs", obs.shape)
			obs = obs.unsqueeze(0).cpu()
			# print("online_trainer.py after obs", obs.shape)

		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		), batch_size=(1,))
		return td

	#@profile
	def train(self):
		"""Train a TD-MPC2 agent."""
		try:
			train_metrics, done, eval_next = {}, True, True
			while self._step <= self.cfg.steps:

				# Evaluate agent periodically
				if self._step % self.cfg.eval_freq == 0:
					eval_next = True

				# Reset environment
				if done:
					if eval_next:
						eval_metrics = self.eval()
						eval_metrics.update(self.common_metrics())
						self.logger.log(eval_metrics, 'eval')
						eval_next = False

					if self._step > 0:
						train_metrics.update(
							episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
							episode_success=info['success'],
						)
						train_metrics.update(self.common_metrics())
						self.logger.log(train_metrics, 'train')
						self._ep_idx = self.buffer.add(torch.cat(self._tds))

					obs = self.env.reset()
					self._tds = [self.to_td(obs)]

				# Collect experience
				if self._step > self.cfg.seed_steps:
					print("online_trainer.py agent act")
					action = self.agent.act(obs, t0=len(self._tds)==1)
				elif self.checkpoint_loaded:
					print("online_trainer.py collecting data agent act")
					action = self.agent.act(obs, t0=len(self._tds)==1)
				else:
					print("online_trainer.py rand act")
					action = self.env.rand_act()
				obs, reward, done, info = self.env.step(action)
				self._tds.append(self.to_td(obs, action, reward))

				# Update agent
				if self._step >= self.cfg.seed_steps:
					if self._step == self.cfg.seed_steps:
						num_updates = self.cfg.seed_steps
						print('Pretraining agent on seed data...')
					else:
						num_updates = 1
					for j in range(num_updates):
						print("onlinetrainer.py agent update ", j)
						_train_metrics = self.agent.update(self.buffer)
					train_metrics.update(_train_metrics)

				self._step += 1
					#save model after every 1000 steps
				if self._step % 5000 == 0:
					print("step", self._step)
					print("saving model")
					print("Max steps", self.cfg.steps)
					self.logger.save_agent(self.agent, self._step)
			self.logger.finish(self.agent)
		except Exception as e:
			self.logger.finish(self.agent)
			print("Exception in training", e)
			self.env.close()
			raise e
