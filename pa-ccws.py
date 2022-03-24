import copy
import random
import uuid
from collections import deque

import gc
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf

import parameters
from common.schedules import LinearSchedule
from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from utils import env_util, plot_util
from utils import http_util

tf.disable_v2_behavior()
GAMMA = 0.99  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 128  # experience replay buffer size
BATCH_SIZE = 128  # size of minibatch
REPLACE_TARGET_FREQ = 10  # frequency to update target Q network
LEARN_RATE = 0.0003
CLUSTER_SAMPLES_NUM = 1000

# Hyper Parameters
EPISODE = 5000  # Episode limitation
STEP = 10000000  # Step limitation in an episode
TEST = 5  # The number of experiment test every 100 episode


class CCWS():
    def __init__(self, id):
        self.id = id
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = 42  # env.observation_space.shape[0]
        self.action_dim = 20  # env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

        self.batch_size = 64
        self.lr = 0.001

    def create_Q_network(self):
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        with tf.variable_scope('current_net' + str(self.id)):
            W1 = self.weight_variable([self.state_dim, 128])
            b1 = self.bias_variable([128])
            W2 = self.weight_variable([128, self.action_dim])
            b2 = self.bias_variable([self.action_dim])
            h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
            self.Q_value = tf.matmul(h_layer, W2) + b2

        with tf.variable_scope('target_net' + str(self.id)):
            W1t = self.weight_variable([self.state_dim, 128])
            b1t = self.bias_variable([128])
            W2t = self.weight_variable([128, self.action_dim])
            b2t = self.bias_variable([self.action_dim])

            h_layer_t = tf.nn.relu(tf.matmul(self.state_input, W1t) + b1t)
            self.target_Q_value = tf.matmul(h_layer_t, W2t) + b2t

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net' + str(self.id))
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net' + str(self.id))

        with tf.variable_scope('soft_replacement' + str(self.id)):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])  # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(LEARN_RATE).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done, weights):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self, minibatch):
        td_errors = []
        self.time_step += 1
        state_batch = minibatch[0]
        action_batch = minibatch[1]
        reward_batch = minibatch[2]
        next_state_batch = minibatch[3]
        weights = minibatch[4]

        y_batch = []
        current_Q_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        max_action_next = np.argmax(current_Q_batch, axis=1)
        Q_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        target_Q_batch = self.target_Q_value.eval(feed_dict={self.state_input: next_state_batch})

        for i in range(0, BATCH_SIZE):
            done = minibatch[4][i]
            if done:
                y_batch.append(reward_batch[i])
            else:
                target_Q_value = target_Q_batch[i, max_action_next[i]]
                y_batch.append(reward_batch[i] + GAMMA * target_Q_value)

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

        for i in range(len(target_Q_batch)):
            td_error = np.dot(target_Q_batch[i], action_batch[i]) - np.dot(Q_batch[i], action_batch[i])
            td_errors.append(td_error)

        return td_errors

    def random_action(self, state):
        return random.randint(0, self.action_dim - 1)

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def update_target_q_network(self, episode):
        if episode % REPLACE_TARGET_FREQ == 0:
            self.session.run(self.target_replace_op)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def train(self,
              print_freq=10,
              checkpoint_freq=10000,
              checkpoint_path=None,
              train_freq=1,
              target_network_update_freq=500,
              learning_starts=1,
              prioritized_replay=True,
              prioritized_replay_alpha=0.6,
              prioritized_replay_beta0=0.4,
              prioritized_replay_beta_iters=None,
              buffer_size=50000,
              total_timesteps=1000000000,
              prioritized_replay_eps=1e-6):

        tasks_list = http_util.get_task_list()
        bath_path = "../../data/" + '_PE_DPTQ_' + str(uuid.uuid1()) + '_lmd_' + str(pa.lmd) + '_rur0.3rbd0.7_'
        if prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = total_timesteps
            beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                           initial_p=prioritized_replay_beta0,
                                           final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(buffer_size)
            beta_schedule = None
        timestep = 0
        cumulative_rewards, cumulative_rewards_test = [], []
        tasks, tasks_test = [], []
        episode_rur, episode_rid = [], []
        traj_rid = []
        episode_per_10_rewards, episode_per_10_tasks, episode_per_10_rur, episode_per_10_rid = [], [], [], []
        for episode in range(EPISODE):
            http_util.env_reset()
            ob = http_util.get_env_observe()
            total_reward = 0
            _rid = 0
            DPTQ = copy.deepcopy(tasks_list[0: pa.f])
            current_task = DPTQ[0]
            for step in range(STEP):
                if step % (pa.f - 1) == 0 and step > 0:
                    connection_point = tasks_list[step + 1]
                    current_task, next_task = DPTQ[0], connection_point
                elif step % pa.f == 0 and step > 0:
                    current_task, next_task = connection_point, DPTQ[0]
                else:
                    next_task = DPTQ[1]
                timestep += 1
                state = env_util.reshape_obv_to_state(ob, current_task, True)
                action = self.egreedy_action(state)
                ob, rew, done, info = http_util.step(current_task, action)
                rur, rid = env_util.get_resource_util_and_balance_from_obv(ob)
                new_state = env_util.reshape_obv_to_state(ob, next_task, True)
                replay_buffer.add(state, action, rew, new_state, float(done))
                total_reward += rew
                _rid += rid
                current_task = next_task
                if step == 0 or step % pa.num_single_launch_tasks != 0:
                    del DPTQ[0]
                if done:
                    break
                if step % (pa.f - 1) == 0 and step > 0:
                    DPTQ = copy.deepcopy(tasks_list[step + 2:step + 2 + pa.f])
                    if len(self.replay_buffer) > CLUSTER_SAMPLES_NUM:
                        cluster_samples = random.sample(self.replay_buffer, CLUSTER_SAMPLES_NUM)
                        env_util.get_priority_queue(cluster_samples, DPTQ, step, ob, pa)
                #
                # if step % (pa.num_single_launch_tasks - 1) == 0 and step > 0:
                #     DPTQ = copy.deepcopy(tasks_list[step + 2:step + 2 + pa.num_single_launch_tasks])
                #     if len(replay_buffer) > BATCH_SIZE:
                #         env_util.get_priority_queue(replay_buffer, DPTQ, step, ob, pa)

                if timestep > learning_starts and timestep % train_freq == 0:
                    if prioritized_replay:
                        experience = replay_buffer.sample(BATCH_SIZE, beta=beta_schedule.value(timestep))
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(BATCH_SIZE)
                        weights, batch_idxes = np.ones_like(rewards), None
                    td_errors = self.train_Q_network([obses_t, actions, rewards, obses_tp1, dones, weights])
                    td_errors = td_errors  # lmd * G(i) + (1-lmd)*|deta(i)|
                    if prioritized_replay:
                        new_priorities = pa.lmd * rewards + (1 - pa.lmd) * np.abs(td_errors) + prioritized_replay_eps
                        replay_buffer.update_priorities(batch_idxes, new_priorities)

            episode_per_10_tasks.append(step)
            episode_per_10_rewards.append(total_reward)
            episode_per_10_rur.append(rur)
            episode_per_10_rid.append(_rid / step)

            if episode % 10 == 0:
                _cumulative_reward, _task_num, _episode_rur, _episode_rid = np.mean(episode_per_10_rewards), np.mean(
                    episode_per_10_tasks), np.mean(episode_per_10_rur), np.mean(episode_per_10_rid)
                cumulative_rewards.append(_cumulative_reward)
                tasks.append(_task_num)
                episode_rur.append(_episode_rur)
                episode_rid.append(_episode_rid)

                with open(bath_path + "_data" + ".txt", 'a', encoding='utf-8') as f:
                    f.writelines(str(_cumulative_reward) + ',' + str(_task_num) + ',' + str(_episode_rur) + ',' + str(
                        _episode_rid) + '\n')
                episode_per_10_rewards = []
                episode_per_10_tasks = []
                episode_per_10_rur = []
                episode_per_10_rid = []
                gc.collect()

            if episode == pa.num_epochs - 1:
                data_file_name = bath_path + "_data" + "traj_rid" + ".txt"
                with open(data_file_name, 'a', encoding='utf-8') as f:
                    for rid in traj_rid:
                        f.writelines(str(rid) + '\n')

            if episode % 100 == 0:
                plot_util.plot_lr_curve(bath_path, cumulative_rewards, tasks, episode_rur, episode_rid, 'ddqn')

            self.update_target_q_network(episode)


if __name__ == '__main__':
    pa = parameters.Parameters()
    agent = CCWS()
    agent.train()
    agent.session.close()
