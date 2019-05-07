import tensorflow as tf
from keras.layers import Input, Conv2D, Dense, Flatten
import keras.backend as K


class network:

    def __init__(self,
                 state_shape,
                 no_actions,
                 learning_rate,
                 config,
                 name
                 ):
        self.state_shape = state_shape
        self.no_actions = no_actions
        self.learning_rate = learning_rate
        name_format = lambda name, digit: "{}{}".format(name, digit)

        self.conv_layers = config["conv_layers"]
        self.fc_layers = config["fc_layers"]
        self.total_layers = self.conv_layers + self.fc_layers
        weights = config["weights"]
        self.kernel_sizes = weights[:self.conv_layers]
        self.units = weights[self.conv_layers:]
        activations = config["activations"]
        self.conv_activations = activations[:self.conv_layers]
        self.fc_activations = activations[self.conv_layers:]
        self.strides = config["strides"]
        self.paddings = config["paddings"]
        self.filters = config["filters"]
        self.epsilon = config.get("epsilon", 1e-10)
        assert len(self.filters) >= self.conv_layers
        assert len(self.strides) >= self.conv_layers
        assert len(self.paddings) >= self.conv_layers
        assert len(self.kernel_sizes) >= self.conv_layers
        assert len(self.units) >= self.fc_layers
        assert len(self.conv_activations) >= self.conv_layers
        assert len(self.fc_activations) >= self.fc_layers

        with tf.variable_scope(name):

            self.states_ = tf.placeholder(tf.float32, [None, *self.state_shape], name="states")
            # self.actions_ = tf.placeholder(tf.float32, [None, self.no_actions], name="actions")

            # self.target_Qs = tf.placeholder(tf.float32,[None,self.no_actions])
            x = Input(tensor=self.states_)

            for i in range(0, self.conv_layers):
                x = Conv2D(filters=self.filters[i],
                           kernel_size=self.kernel_sizes[i],
                           strides=self.strides[i],
                           padding=self.paddings[i],
                           data_format="channels_first",
                           activation=self.conv_activations[i],
                           kernel_initializer=tf.variance_scaling_initializer(scale=2),
                           name=name_format("conv", i)
                           )(x)

            # if self.conv_layers > 0:
            x = Flatten()(x)
            for i in range(0, self.fc_layers):
                x = Dense(units=self.units[i],
                          activation=self.fc_activations[i],
                          kernel_initializer=tf.variance_scaling_initializer(scale=2),
                          name=name_format("fc", i)
                          )(x)

            self.output = Dense(
                units=self.no_actions,
                activation=None,
                name='output',
                kernel_initializer="random_uniform"
            )(x)
            # print(self.output.shape)
            self.target_Qs = tf.placeholder(tf.float32,[None], name="target")
            self.actions_ = tf.placeholder(tf.int32,[None], name="actions")
            self.Qs = tf.reduce_sum(
                tf.multiply(self.output, tf.one_hot(self.actions_, self.no_actions, dtype=tf.float32)), axis=1)
            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.target_Qs, predictions=self.Qs))
            opt_nm = config.get("optimizer", "rmsprop")
            if opt_nm == "adam":
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            else:
                self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, epsilon=self.epsilon).minimize(self.loss)
        print("network initialized !")

    def get_session(self):
        return K.get_session()

    def set_session(self, sess):
        K.set_session(sess)
