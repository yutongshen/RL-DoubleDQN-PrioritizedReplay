import pandas as pd
import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential, Model, clone_model
from keras.layers.core import Dense, Activation, Lambda
from keras.layers import Input, LSTM, TimeDistributed, RepeatVector, Reshape, Dropout, Bidirectional, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop

class SumTree:
    def __init__(self, mem_size, data_len):
        self.tree = np.zeros(2 * mem_size - 1)
        self.data = np.zeros((mem_size, data_len), np.float32)
        self.size = mem_size
        self.ptr  = 0

    def update(self, idx, p):
        tree_idx = idx + self.size - 1
        diff = p - self.tree[tree_idx]
        self.tree[tree_idx] += diff
        while tree_idx:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += diff

    def store(self, p, data):
        self.data[self.ptr] = data
        self.update(self.ptr, p)

        self.ptr += 1
        if self.ptr == self.size:
            self.ptr = 0

    def sample(self, value):
        ptr = 0
        while ptr < self.size - 1:
            left = 2 * ptr + 1
            if value < self.tree[left]:
                ptr = left
            else:
                value -= self.tree[left]
                ptr = left + 1

        return ptr - (self.size - 1), self.tree[ptr], self.data[ptr - (self.size - 1)]

    @property
    def total_p(self):
        return self.tree[0]

    @property
    def max_p(self):
        return np.max(self.tree[-self.size:])

    @property
    def min_p(self):
        return np.min(self.tree[-self.size:])

class Memory:
    p_upper = 1.
    epsilon = .01
    alpha   = .7
    beta    = .5
    def __init__(self, mem_size, feature_size, prior=True):
        self.prior    = prior
        self.data_len = 2 * feature_size + 2
        if prior:
            self.tree  = SumTree(mem_size, self.data_len)
        else:
            self.mem_size = mem_size
            self.mem      = np.zeros((mem_size, self.data_len), np.float32)
            self.mem_ptr  = 0

    def store(self, transition):
        if self.prior:
            p = self.tree.max_p
            if not p:
                p = self.p_upper
            self.tree.store(p, transition)
        else:
            self.mem[self.mem_ptr] = transition
            self.mem_ptr += 1
            if self.mem_ptr == self.mem_size:
                self.mem_ptr = 0
            
    def sample(self, n):
        if self.prior:
            min_p = self.tree.min_p
            seg   = self.tree.total_p / n
            batch = np.zeros((n, self.data_len), np.float32)
            w     = np.zeros((n, 1), np.float32)
            idx   = np.zeros(n, np.int32)
            a = 0
            for i in range(n):
                b = a + seg
                v = np.random.uniform(a, b)
                idx[i], p, batch[i] = self.tree.sample(v)
                w[i] = (p / min_p) ** (-self.beta)
                a += seg
            self.beta = min(1., self.alpha + .01)
            return idx, w, batch
        else:
            mask = np.random.choice(range(self.mem_size), n)
            return self.mem[mask]

    def update(self, idx, tderr):
        if self.prior:
            tderr += self.epsilon
            tderr = np.minimum(tderr, self.p_upper)
            for i in range(len(idx)):
                self.tree.update(idx[i], tderr[i] ** self.alpha)

class RL:
    def __init__(self, feature_size, actions, epsilon=.5, epsilon_min=.1, epsilon_decrease=.9, gamma=.9, lr=.0005, memory_size=10000, batch_size=500, replace_target_iter=500, prior=True, verbose=True):
        self.epsilon             = epsilon
        self.epsilon_min         = epsilon_min
        self.epsilon_decrease    = epsilon_decrease
        self.gamma               = gamma
        self.lr                  = lr
        self.actions             = actions
        self.feature_size        = feature_size
        self.memory_size         = memory_size
        self.replace_target_iter = replace_target_iter
        self.prior               = prior
        self.memory              = Memory(memory_size, feature_size, prior)
        self.q_eval_model        = None
        self.q_target_model      = None
        self.batch_size          = batch_size
        self.step_cnt            = 0
        self.learning_cnt        = 0
        self.history             = []
        self.verbose             = verbose
        
        self._build_model()

    def _build_model(self):
        # Q-Evaluation Model
        HIDDEN_SIZE = 20

        def weight_loss_wrapper(input_tensor):
            def weight_loss(y_true, y_pred):
                return K.mean(K.square(y_true - y_pred) * input_tensor)
            return weight_loss

        def mse_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred))

        inputs = Input(shape=(self.feature_size,))
        if self.prior:
            weights = Input(shape=(1,))
        fc_1   = Dense(HIDDEN_SIZE, activation='relu')(inputs)
        fc_2   = Dense(len(self.actions))(fc_1)
        rmsp = RMSprop(lr=self.lr)
        if self.prior:
            self.q_eval_model = Model([inputs, weights], fc_2)
            self.q_eval_model.compile(loss=weight_loss_wrapper(weights), optimizer=rmsp)
        else:
            self.q_eval_model = Model(inputs, fc_2)
            self.q_eval_model.compile(loss=mse_loss, optimizer=rmsp)

        self.q_eval_model.summary()

    def _copy_model(self):
        self.q_target_model = keras.models.clone_model(self.q_eval_model)
        self.q_target_model.set_weights(self.q_eval_model.get_weights())

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((list(s), [self.actions.index(a), r], list(s_)))
        self.memory.store(transition)
        self.step_cnt += 1

    def draw_policy(self):
        arr = np.zeros((100, 100, 3), np.int32)
        for i in range(100):
            for j in range(100):
                value = self.q_eval_model.predict([np.array([[i * 0.018 - 1.2, -j * 0.0014 + 0.07]]), np.ones((1, 1))])
                value[0] = value[0] - value[0].max()
                value[0] = np.exp(value[0]) + .000001
                value[0] = value[0] * 768 / np.sum(value[0])
                arr[j, i] = value[0]
        return arr

    def actor(self, observation):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            observation = np.array(observation)
            observation = observation[np.newaxis, :]
            if self.prior:
                q_value = self.q_eval_model.predict([observation, np.ones((1, 1))])
            else:
                q_value = self.q_eval_model.predict(observation)
            action = self.actions[q_value.argmax()]

        return action

    def learn(self):
        if self.learning_cnt % self.replace_target_iter == 0:
            self._copy_model()
            if self.verbose: print('Copy model')

        if self.prior:
            idx, w, transition = self.memory.sample(self.batch_size)
        else:
            transition = self.memory.sample(self.batch_size)

        s  = transition[:, :self.feature_size]
        s_ = transition[:, -self.feature_size:]
        r  = transition[:, self.feature_size + 1]

        if self.prior:
            index = self.q_eval_model.predict([s_, np.ones((self.batch_size, 1))]).argmax(axis=1)
            max_q = self.q_target_model.predict([s_, np.ones((self.batch_size, 1))])[range(self.batch_size), index]
            q_predict = self.q_eval_model.predict([s, np.ones((self.batch_size, 1))])
        else:
            index = self.q_eval_model.predict(s_).argmax(axis=1)
            max_q = self.q_target_model.predict(s_)[range(self.batch_size), index]
            q_predict = self.q_eval_model.predict(s)

        q_target = np.copy(q_predict)
        q_target[range(self.batch_size), transition[:, self.feature_size].astype(np.int32)] = r + self.gamma * max_q
        
        if self.prior:
            q_pred = self.q_eval_model.predict([s, np.ones((self.batch_size, 1))])
            p = np.sum(np.abs(q_pred - q_target), axis=1)
            self.memory.update(idx, p)
            report = self.q_eval_model.fit([s, w], q_target, verbose=0)
        else:
            report = self.q_eval_model.fit(s, q_target, verbose=0)

        self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * self.epsilon_decrease

        self.history.append(report.history['loss'])

        if self.verbose and not self.learning_cnt % 100:
            print('training', self.learning_cnt, ': loss', report.history['loss'])

        self.learning_cnt += 1
