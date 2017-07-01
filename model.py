import numpy as np
import random
import tensorflow as tf

class Model:

    def __init__(self, n_actions, n_features, learning_rate, discount_rate, explore_rate):
        random.seed()
        # Only two action in this case: moving left or right
        self.n_actions = n_actions
        # Four features: which is the first output of gym.step(action), which is (x, dx, theta, dtheta), representing the current pole's status:
        # Position on x-axis, speed on x-axis, angle, and angle speed
        self.n_features = n_features
        # Define the learning rate
        self.learning_rate = learning_rate
        # Define the reward decay rate, it is the 'gamma' symbol in the Bellman equation.
        self.discount_rate = discount_rate
        # Define the explore rate, which means the probability to choose the highest Q against picking a random value
        # to avoid trapped in local optimisation
        self.explore_rate = explore_rate
        # Define the memory size to store the transations for experience replay
        self.memory_size = 2000
        # Every time pick certain amount of transaction records from the memory for experience replay
        self.batch_size = 32
        # the point to the memory area
        self.memory_total_records = 0

        # counting total learning step from beginning
        self.learn_step_counter = 0

        # initialize memory to all 0: each record stores [observation, action, reward, next_observation]
        # so the total memory size = memory_size * (sizeof(observation) + sizeof(next_observation) + sizeof(action) + sizeof(reward)
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        # build up two neural network: one for
        self.build_dqn()

        self.sess = tf.Session()

        # $ tensorboard --logdir=logs
        tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        #self.cost_his = []

    def build_dqn(self):

        # build up two neural networks as evaluation and target
        #placeholder of input: observation
        self.observation = tf.placeholder(tf.float32, [None, self.n_features], name='observation')
        #placeholder for target Q value
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        #Two hidden layers, each layer has 10 nodes
        level1 = 10
        level2 = 10

        #randomise the initial weights / bias using gaussian distribution
        w_initialiser = tf.random_normal_initializer(-0.25, 0.25)
        #b_initialiser = tf.constant_initializer(0.1)
        b_initialiser = tf.random_normal_initializer(-0.1, 0.1)

        # Evaluation Neural Network
        # first layer.
        self.ev_w1 = tf.get_variable('ev_w1', [self.n_features, level1], initializer=w_initialiser, collections=['eval_net_params', 'variables'])
        self.ev_b1 = tf.get_variable('ev_b1', [1, level1], initializer=b_initialiser, collections=['eval_net_params','variables'])
        self.ev_l1 = tf.nn.relu(tf.matmul(self.observation, self.ev_w1) + self.ev_b1)
        # second layer
        self.ev_w2 = tf.get_variable('ev_w2', [level1, level2], initializer=w_initialiser, collections=['eval_net_params','variables'])
        self.ev_b2 = tf.get_variable('ev_b2', [1, level2], initializer=b_initialiser, collections=['eval_net_params','variables'])
        self.ev_l2 = tf.nn.relu(tf.matmul(self.ev_l1, self.ev_w2) + self.ev_b2)

        # third layer.
        self.ev_w3 = tf.get_variable('ev_w3', [level2, self.n_actions], initializer=w_initialiser, collections=['eval_net_params','variables'])
        self.ev_b3 = tf.get_variable('ev_b3', [1, self.n_actions], initializer=b_initialiser, collections=['eval_net_params','variables'])
        self.q_eval = tf.matmul(self.ev_l2, self.ev_w3) + self.ev_b3



        self.new_observation = tf.placeholder(tf.float32, [None, self.n_features], name='new_observation')

        # Target Neural Network
        # first layer.
        self.tg_w1 = tf.get_variable('tg_w1', [self.n_features, level1], initializer=w_initialiser, collections=['target_net_params','variables'])
        self.tg_b1 = tf.get_variable('tg_b1', [1, level1], initializer=b_initialiser, collections=['target_net_params','variables'])
        self.tg_l1 = tf.nn.relu(tf.matmul(self.new_observation, self.tg_w1) + self.tg_b1)

        #second layer
        self.tg_w2 = tf.get_variable('tg_w2', [level1, level2], initializer=w_initialiser, collections=['target_net_params','variables'])
        self.tg_b2 = tf.get_variable('tg_b2', [1, level2], initializer=b_initialiser, collections=['target_net_params','variables'])
        self.tg_l2 = tf.nn.relu(tf.matmul(self.tg_l1, self.tg_w2) + self.tg_b2)

        # third layer
        self.tg_w3 = tf.get_variable('tg_w3', [level2, self.n_actions], initializer=w_initialiser, collections=['target_net_params','variables'])
        self.tg_b3 = tf.get_variable('tg_b3', [1, self.n_actions], initializer=b_initialiser, collections=['target_net_params','variables'])
        # The Q Nest is for being used to calculate the Target Q applied to Bellman Equation
        self.q_next = tf.matmul(self.tg_l2, self.tg_w3) + self.tg_b3

        # Define the cost be the difference between Q value of Evaluation Neural Network and the Target Neural Network
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        # Optimise to get the minimised cost, which is the difference between evaluation Q and target Q
        self.train = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def store_transition(self, observation, action, reward, new_observation):

        transition = np.append(observation, np.array([action, reward]))
        transition = np.append(transition, new_observation)

        # earliest records will be replaced by the newest one
        ptr = self.memory_total_records % self.memory_size
        self.memory[ptr, :] = transition

        self.memory_total_records += 1

    def select_action(self, observation):

        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        # Select a random action
        if np.random.uniform() < self.explore_rate:
            #action = round(random.random())
            action = np.random.randint(0, self.n_actions)
        # Select the action with the highest q
        else:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.observation: observation})
            action = np.argmax(actions_value)
        return action

    def replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
        #self.tg_b1 = self.ev_b1
        #self.tg_b2 = self.ev_b2
        #self.tg_b3 = self.ev_b3
        #self.tg_w1 = self.ev_w1
        #self.tg_w2 = self.ev_w2
        #self.tg_w3 = self.ev_w3

    def learn(self):
        # check to replace target parameters. Target network's parameter are only changed every 500 steps
        if self.learn_step_counter % 500 == 0:
            self.replace_target_params()
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_total_records > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_total_records, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],feed_dict={
                self.new_observation: batch_memory[:, -self.n_features:],  # the target network is using new observation and old parameters
                self.observation: batch_memory[:, :self.n_features],  # the evaluation network is using old observation and new parammeters
            })

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        # use Bellman equation to update the target Q value, the ones not being touched will be kept same as q_eval for calculation purpose
        q_target[batch_index, eval_act_index] = reward + self.discount_rate * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self.train, self.loss],feed_dict={self.observation: batch_memory[:, :self.n_features], self.q_target: q_target})
        #print("cost is " + str(self.cost) + "\n")
        self.learn_step_counter += 1

    # Not used
    def update(self, learning_rate, explorer_rate):
        self.learning_rate = learning_rate
        self.explore_rate = explorer_rate