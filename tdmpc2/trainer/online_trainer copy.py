from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer
import os

class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		# load model if checkpoint is provided
		# print("checkpoint", self.cfg.checkpoint)
		# if model exists, load it

		if os.path.exists(self.cfg.checkpoint):
			print("loading checkpoint: ", self.cfg.checkpoint)
			self.agent.load(self.cfg.checkpoint)
			print("Model loaded from checkpoint")
		else:
			print("Train from scratch")
		# self._tds = []

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		# for i in range(self.cfg.eval_episodes):
		# Correct later
		for i in range(10):
			print("eval episode", i)
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			# print("eval obs", obs)
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				print("eval step", t)
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

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		# print("onlinetrainer.py before td obs", obs['image'].shape, obs['state'].shape)
		if isinstance(obs, dict):
			# print("obs is dict")
			obs['image'] = obs['image'].unsqueeze(0)
			obs['state'] = obs['state'].unsqueeze(0)
			# print("onlinetrainer.py obs", obs)
			obs = TensorDict(obs, batch_size=(), device='cpu')
			# print("onlinetrainer.py obs tensordictified", obs)
		else:
			# print("onlinetrainer.py obs is not dict")
			obs = obs.unsqueeze(0).cpu()
			# print("onlinetrainer.py obs", obs.shape)
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		), batch_size=(1,))

		# print("onlinetrainer.py after td obs", td['obs'])
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, True
		while self._step <= self.cfg.steps:
			
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					print("Evaluating agent")
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					print("Logging training metrics")
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))
					print("onlinetrainer.py self.buffer", self.buffer)
				obs = self.env.reset()
				print("onlinetrainer.py reset image obs", obs['image'].shape)
				print("onlinetrainer.py reset state obs", obs['state'].shape)

				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				print("Agent acting")
				print("onlinetrainer.py before act image obs", obs['image'].shape)
				print("onlinetrainer.py before act state obs", obs['state'].shape)

				## NO IDEA WHERE THE /BUGG IS FROM
				if isinstance(obs, dict):
					if obs['image'].ndim == 4:
						obs['image'] = obs['image'].squeeze(0)
						print("onlinetrainer.py after squeeze image obs", obs['image'].shape)
						print("BUGG")
					if obs['state'].ndim == 2:
						obs['state'] = obs['state'].squeeze(0)
						print("onlinetrainer.py after squeeze state obs", obs['state'].shape)

				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				print("Agent random acting")
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			print("onlinetrainer.py step image obs", obs['image'].shape)
			print("onlinetrainer.py step state obs", obs['state'].shape)
			self._tds.append(self.to_td(obs, action, reward))
			print("onlinetrainer.py self._td", self._tds)

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
					# done = True
				else:
					# print("incremental update")
					num_updates = 1
				for j in range(num_updates):
					print("updating agent", j)
					_train_metrics = self.agent.update(self.buffer)
				# print("train metrics", _train_metrics)
				train_metrics.update(_train_metrics)
				# print("onlinetrainer.py done: ", done)

			self._step += 1

			#save model after every 1000 steps
			if self._step % 5000 == 0:
				print("step", self._step)
				print("saving model")
				print("Max steps", self.cfg.steps)
				self.logger.save_agent(self.agent, self._step)

	
		self.logger.finish(self.agent)
