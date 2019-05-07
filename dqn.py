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
        tf.reset_default_graph()
        self.net = network(state_shape, no_actions, self.learning_rate, config, "DQNetwork")
        # self.create_hardwired_network(name)
        self.sess = tf.Session()
        self.net.set_session(self.sess)
        #self.writer = tf.summary.FileWriter(os.getcwd() + "/Graphs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.batch_train = config.get("batch_train", "whole") == "whole"
        print("agent initialized !")

    def evaluate(self, state):
        Qs = self.sess.run(self.net.output, feed_dict={self.net.states_: state.reshape([1, *state.shape])})
        return np.argmax(Qs)

    def train(self, states_mb,actions_mb,rewards_mb,next_states_mb,dones_mb, done=False):
        batch_size = len(states_mb)
        qs_next_state = self.sess.run(self.net.output, feed_dict={self.net.states_: next_states_mb})
        target_qs_mb = []
        for i in range(batch_size):
            if dones_mb[i]:
                target_qs_mb.append(rewards_mb[i])
            else:
                target_qs_mb.append(rewards_mb[i] + self.gamma * np.max(qs_next_state[i]))

        # print(np.array(target_qs_mb).shape,np.array(rewards_mb).shape)
        loss = 0
        # self.batch_train = True
        if not self.batch_train:
            for i in range(batch_size):
                ind_loss, _ = self.sess.run([self.net.loss, self.net.optimizer], feed_dict=
                {self.net.states_: states_mb[i].reshape([1, *states_mb[i].shape]),
                 self.net.actions_: actions_mb[i].reshape([1,*actions_mb[i].shape]),
                 self.net.target_Qs: target_qs_mb[i].reshape([1,*target_qs_mb[i].shape])})
                loss += ind_loss
            loss /= batch_size
        else:
            loss, _ = self.sess.run([self.net.loss, self.net.optimizer],
                                    feed_dict={self.net.states_: states_mb,
                                               self.net.actions_: actions_mb,
                                               self.net.target_Qs: target_qs_mb})
        # self.writer.close()

        return loss

    def save(self, path):
        self.saver.save(self.sess, path)
        print("Model saved !")

    def restore(self, path):
        print(path)
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
