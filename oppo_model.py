
import numpy as np
import random

import json
import sys

from two_player.pong import PongGame
from prey_predator.env import PreyPredatorEnv
import time
import argparse

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers
import util.tf_util as U

def lstm_model(input, num_outputs, scope, reuse = False, num_units = 4):

    with tf.variable_scope(scope, reuse = reuse):
        weight = tf.get_variable("W", shape=[num_units, num_outputs],
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("B", shape=[num_outputs],
                               initializer=tf.contrib.layers.xavier_initializer())
        # x = tf.reshape(input, (-1, TIMESTEPS, input.get_shape().as_list()[1] *input.get_shape().as_list()[-1]))
        x = tf.unstack(input, axis=1)
        rnn_cell = rnn.BasicLSTMCell(int(num_units))
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32, scope="trainging_agent")
        out = layers.fully_connected(outputs[-1], num_outputs=num_outputs, activation_fn=tf.nn.relu)
        # out =layers.fully_connected(outputs[-1], num_outputs=num_outputs)
        return out

class Oppo_Model():
	def __init__(self,num_agent, timesteps,  action_space, output_size, scope):
		self.timesteps = timesteps
		self.action_space = action_space
		self.output_size = output_size
		self._lr = 0.0001
		

		#define placeholders
		self.act_traj = tf.placeholder(tf.float32, shape = (None, timesteps, action_space))
		self.action_target = tf.placeholder(tf.float32, shape = (None, action_space))
		with tf.variable_scope(scope):
			self.tau = lstm_model(self.act_traj, action_space, scope = "tau_model")
			self.tau_loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.tau, labels=self.action_target))
			self.tau_opt_op  = tf.train.AdamOptimizer(self._lr).minimize(self.tau_loss)
			self.get_tau = U.function(inputs = [self.act_traj], outputs = [self.tau])
			self.train_tau = U.function(inputs = [self.act_traj] + [self.action_target], outputs = [self.tau_loss], updates = [self.tau_opt_op])
	# 	self.g = tf.Graph()
	# 	self.init =tf.global_variables_initializer()
	# def init_session(self):	
	# 	self.sess = tf.Session(grah = self.g)
	# 	self.sess.run(self.init)	
	def supervise_tau(self, a_next, act_traj):
		loss = self.train_tau(*([a_next] + [act_traj]))[0]
		return loss			

	def get_inference(self, act_traj):
		return self.get_tau(*([[act_traj]]))[0]

