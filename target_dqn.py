import tensorflow as tf
import numpy as np
import os, sys
from dynamic_network import network


class Agent:

    def __init__(self, state_shape, no_actions, config):
        self.state_shape = state_shape
        self.no_actions = no_actions
        self.learning_rate = config.get("learning_rate", 0.00025)
        self.gamma = config.get("gamma", 0.9)
        self.name = config.get("name", "DQNetwork")
        tf.reset_default_graph()
        self.net = network(state_shape, no_actions, self.learning_rate, config, "DQNetwork")
        self.target_net = network(state_shape, no_actions, self.learning_rate, config, "TargetNetwork")

        # self.create_hardwired_network(name)
        self.sess = tf.Session()
        self.net.set_session(self.sess)
        self.target_net.set_session(self.sess)
        self.writer = tf.summary.FileWriter(os.getcwd() + "/Graphs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        self.batch_train = config.get("batch_train", "whole") == "whole"
        self.tau = 1
        self.max_tau = config.get("max_tau", 500)
        self.max_tau_done = False
        if self.max_tau == "on_terminal_state":
            self.max_tau_done = True
            self.max_tau = self.tau + 1
        print("agent initialized !")

    def update_target_graph(self):

        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

        op_holder = []

        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def evaluate(self, state):
        Qs = self.sess.run(self.net.output, feed_dict={self.net.states_: state.reshape([1, *state.shape])})
        return np.argmax(Qs)

    def train(self, states_mb,actions_mb,rewards_mb,next_states_mb,dones_mb, done=False):
        batch_size = len(states_mb)
        qs_next_target = self.sess.run(self.target_net.output, feed_dict={self.target_net.states_: next_states_mb})
        target_qs_mb = self.sess.run(self.net.output, feed_dict={self.net.states_: states_mb})

        for i in range(batch_size):
            if dones_mb[i]:
                target_qs_mb[i, actions_mb[i]] = rewards_mb[i]
            else:
                target_qs_mb[i, actions_mb[i]] = rewards_mb[i] + self.gamma * np.max(qs_next_target[i])
        loss = 0

        if not self.batch_train:
            for i in range(batch_size):
                ind_loss, _ = self.sess.run([self.net.loss, self.net.optimizer],
                                            feed_dict={self.net.states_: states_mb[i].reshape([1, *states_mb[i].shape]),
                                                       self.net.target_Qs: target_qs_mb[i].reshape(
                                                           [1, *target_qs_mb[i].shape])})
                loss += ind_loss
            loss /= batch_size
        else:
            loss, _ = self.sess.run([self.net.loss, self.net.optimizer],
                                    feed_dict={self.net.states_: states_mb,
                                               self.net.target_Qs: target_qs_mb})

        if self.max_tau_done:
            if not done:
                self.max_tau = self.tau + 1
            else:
                self.max_tau = self.tau - 1
        if self.tau > self.max_tau:
            self.tau = 0
            update_target = self.update_target_graph()
            self.sess.run(update_target)
            print("Model updated")
        else:
            self.tau += 1
        # print("tau: {}".format(self.tau))
        # self.writer.close()
        return loss

    def save(self, path):
        self.saver.save(self.sess, path)
        print("Model saved !")

    def restore(self, path):
        try:
            self.saver.restore(self.sess, path)
            print("restored the model from {}".format(path))
            return True
        except:
            print("exception -> {}".format(sys.exc_info()[0]))
            print("can't restore the model")
            return False

    def end(self):
        self.sess.close()
