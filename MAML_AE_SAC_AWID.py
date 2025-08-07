import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import json
from sklearn.utils import shuffle
import os
import time
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import itertools
from sklearn.metrics import  confusion_matrix
#from ConfusionMatrix import ConfusionMatrix
#from keras_flops import get_flops
import pickle as pickle
import random
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
train_path = "C:/Users/duyda/Documents/MAML/data/AWID/DATASET/AWID-CLS-R-Trn/1"
test_path = "C:/Users/duyda/Documents/MAML/data/AWID/DATASET/AWID-CLS-R-Tst/1"
ATT_NUM = [0,0,0,0]

class data_cls:
    def __init__(self, train_test, **kwargs):
        # ... (existing initialization code remains unchanged) ...
        self.index = 0
        self.loaded = False
        self.train_test = train_test
        self.train_path = train_path
        self.test_path = test_path
        self.formated_train_path = kwargs.get('formated_train_path', "formated_train_awid2.data")
        self.formated_test_path = kwargs.get('formated_test_path', "formated_test_awid2.data")
        
        features = ['frame.interface_id',
    'frame.dlt',
    'frame.offset_shift',
    'frame.time_epoch',
    'frame.time_delta',
    'frame.time_delta_displayed',
    'frame.time_relative',
    'frame.len',
    'frame.cap_len',
    'frame.marked',
    'frame.ignored',
    'radiotap.version',
    'radiotap.pad',
    'radiotap.length',
    'radiotap.present.tsft',
    'radiotap.present.flags',
    'radiotap.present.rate',
    'radiotap.present.channel',
    'radiotap.present.fhss',
    'radiotap.present.dbm_antsignal',
    'radiotap.present.dbm_antnoise',
    'radiotap.present.lock_quality',
    'radiotap.present.tx_attenuation',
    'radiotap.present.db_tx_attenuation',
    'radiotap.present.dbm_tx_power',
    'radiotap.present.antenna',
    'radiotap.present.db_antsignal',
    'radiotap.present.db_antnoise',
    'radiotap.present.rxflags',
    'radiotap.present.xchannel',
    'radiotap.present.mcs',
    'radiotap.present.ampdu',
    'radiotap.present.vht',
    'radiotap.present.reserved',
    'radiotap.present.rtap_ns',
    'radiotap.present.vendor_ns',
    'radiotap.present.ext',
    'radiotap.mactime',
    'radiotap.flags.cfp',
    'radiotap.flags.preamble',
    'radiotap.flags.wep',
    'radiotap.flags.frag',
    'radiotap.flags.fcs',
    'radiotap.flags.datapad',
    'radiotap.flags.badfcs',
    'radiotap.flags.shortgi',
    'radiotap.datarate',
    'radiotap.channel.freq',
    'radiotap.channel.type.turbo',
    'radiotap.channel.type.cck',
    'radiotap.channel.type.ofdm',
    'radiotap.channel.type.2ghz',
    'radiotap.channel.type.5ghz',
    'radiotap.channel.type.passive',
    'radiotap.channel.type.dynamic',
    'radiotap.channel.type.gfsk',
    'radiotap.channel.type.gsm',
    'radiotap.channel.type.sturbo',
    'radiotap.channel.type.half',
    'radiotap.channel.type.quarter',
    'radiotap.dbm_antsignal',
    'radiotap.antenna',
    'radiotap.rxflags.badplcp',
    'wlan.fc.type_subtype',
    'wlan.fc.version',
    'wlan.fc.type',
    'wlan.fc.subtype',
    'wlan.fc.ds',
    'wlan.fc.frag',
    'wlan.fc.retry',
    'wlan.fc.pwrmgt',
    'wlan.fc.moredata',
    'wlan.fc.protected',
    'wlan.fc.order',
    'wlan.duration',
    'wlan.ra',
    'wlan.da',
    'wlan.ta',
    'wlan.sa',
    'wlan.bssid',
    'wlan.frag',
    'wlan.seq',
    'wlan.bar.type',
    'wlan.ba.control.ackpolicy',
    'wlan.ba.control.multitid',
    'wlan.ba.control.cbitmap',
    'wlan.bar.compressed.tidinfo',
    'wlan.ba.bm',
    'wlan.fcs_good',
    'wlan_mgt.fixed.capabilities.ess',
    'wlan_mgt.fixed.capabilities.ibss',
    'wlan_mgt.fixed.capabilities.cfpoll.ap',
    'wlan_mgt.fixed.capabilities.privacy',
    'wlan_mgt.fixed.capabilities.preamble',
    'wlan_mgt.fixed.capabilities.pbcc',
    'wlan_mgt.fixed.capabilities.agility',
    'wlan_mgt.fixed.capabilities.spec_man',
    'wlan_mgt.fixed.capabilities.short_slot_time',
    'wlan_mgt.fixed.capabilities.apsd',
    'wlan_mgt.fixed.capabilities.radio_measurement',
    'wlan_mgt.fixed.capabilities.dsss_ofdm',
    'wlan_mgt.fixed.capabilities.del_blk_ack',
    'wlan_mgt.fixed.capabilities.imm_blk_ack',
    'wlan_mgt.fixed.listen_ival',
    'wlan_mgt.fixed.current_ap',
    'wlan_mgt.fixed.status_code',
    'wlan_mgt.fixed.timestamp',
    'wlan_mgt.fixed.beacon',
    'wlan_mgt.fixed.aid',
    'wlan_mgt.fixed.reason_code',
    'wlan_mgt.fixed.auth.alg',
    'wlan_mgt.fixed.auth_seq',
    'wlan_mgt.fixed.category_code',
    'wlan_mgt.fixed.htact',
    'wlan_mgt.fixed.chanwidth',
    'wlan_mgt.fixed.fragment',
    'wlan_mgt.fixed.sequence',
    'wlan_mgt.tagged.all',
    'wlan_mgt.ssid',
    'wlan_mgt.ds.current_channel',
    'wlan_mgt.tim.dtim_count',
    'wlan_mgt.tim.dtim_period',
    'wlan_mgt.tim.bmapctl.multicast',
    'wlan_mgt.tim.bmapctl.offset',
    'wlan_mgt.country_info.environment',
    'wlan_mgt.rsn.version',
    'wlan_mgt.rsn.gcs.type',
    'wlan_mgt.rsn.pcs.count',
    'wlan_mgt.rsn.akms.count',
    'wlan_mgt.rsn.akms.type',
    'wlan_mgt.rsn.capabilities.preauth',
    'wlan_mgt.rsn.capabilities.no_pairwise',
    'wlan_mgt.rsn.capabilities.ptksa_replay_counter',
    'wlan_mgt.rsn.capabilities.gtksa_replay_counter',
    'wlan_mgt.rsn.capabilities.mfpr',
    'wlan_mgt.rsn.capabilities.mfpc',
    'wlan_mgt.rsn.capabilities.peerkey',
    'wlan_mgt.tcprep.trsmt_pow',
    'wlan_mgt.tcprep.link_mrg',
    'wlan.wep.iv',
    'wlan.wep.key',
    'wlan.wep.icv',
    'wlan.tkip.extiv',
    'wlan.ccmp.extiv',
    'wlan.qos.tid',
    'wlan.qos.priority',
    'wlan.qos.eosp',
    'wlan.qos.ack',
    'wlan.qos.amsdupresent',
    'wlan.qos.buf_state_indicated1',
    'wlan.qos.bit4',
    'wlan.qos.txop_dur_req',
    'wlan.qos.buf_state_indicated2',
    'data.len',
    'labels']

        main_features = [
        'frame.len',
        'radiotap.length',
        'radiotap.dbm_antsignal',
        'wlan.duration',
        'radiotap.present.tsft',
        'radiotap.channel.freq',
        'radiotap.channel.type.cck',
        'radiotap.channel.type.ofdm',
        'wlan.fc.type',
        'wlan.fc.subtype',
        'wlan.fc.ds',
        'wlan.fc.frag',
        'wlan.fc.retry',
        'wlan.fc.pwrmgt',
        'wlan.fc.moredata',
        'wlan.fc.protected',
        'labels'
    ]

        self.attack_types = ['normal', 'flooding', 'injection', 'impersonation']
        self.all_attack_types = ['normal', 'flooding', 'injection', 'impersonation']
        self.attack_names = []
        self.attack_map = {'normal': 'normal', 'flooding': 'flooding', 'injection': 'injection',
                        'impersonation': 'impersonation'}
        self.all_attack_names = list(self.attack_map.keys())

        formated = False

        if os.path.exists(self.formated_train_path) and os.path.exists(self.formated_test_path):
            formated = True
        self.formated_dir = "../datasets/formated/"
        if not os.path.exists(self.formated_dir):
            os.makedirs(self.formated_dir)
        
        if not formated:
            ''' Formatting the dataset for ready-to-use data '''
            # Training set
            self.df = pd.read_csv(self.train_path, sep=',', names=features, encoding='latin-1', low_memory=False)
            self.df= self.df[main_features].copy()
            self.df.replace("?", np.nan, inplace=True)
            self.df.dropna(inplace=True)

            data2 = pd.read_csv(self.test_path, sep=',',names=features, encoding='latin-1',low_memory=False)
            data2 = data2[main_features].copy()
            data2.replace("?", np.nan, inplace=True)
            data2.dropna(inplace=True)
            
            test_index = data2.shape[0]
            frames = [data2, self.df]
            self.df = pd.concat(frames)
            print(f"Original training shape: {self.df.shape}")

            numeric_features = ['frame.len', 'radiotap.length', 'radiotap.dbm_antsignal', 'wlan.duration']
            categorical_features = [col for col in main_features if col not in numeric_features]
            
            for col in numeric_features:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    if self.df[col].max() == self.df[col].min():
                        self.df[col] = 0
                    else:
                        self.df[col] = (self.df[col] - self.df[col].min()) / (
                                    self.df[col].max() - self.df[col].min())
            self.df[numeric_features] = self.df[numeric_features].astype('float64')
            
            for col in categorical_features:
                if col in self.df.columns:
                    self.df = pd.concat([self.df.drop(col, axis=1), pd.get_dummies(self.df[col])], axis=1)
            #self.df.drop(columns=cols_to_drop, inplace=True)
            

            test_df = self.df.iloc[:test_index]
            self.df = self.df[test_index:self.df.shape[0]]
            # Shuffle and save
            test_df = shuffle(test_df, random_state=np.random.randint(0, 100))
            self.df = shuffle(self.df, random_state=np.random.randint(0, 100))
            test_df.to_csv(self.formated_test_path, sep=',', index=False)
            self.df.to_csv(self.formated_train_path, sep=',', index=False)
    def get_shape(self):
        if self.loaded is False:
            self._load_df()

        self.data_shape = self.df.shape
        # stata + labels
        return self.data_shape

    ''' Get n-rows from loaded data 
        The dataset must be loaded in RAM
    '''

    def get_batch(self, batch_size=100):
        
        # Read the df rows
        indexes = list(range(self.index, self.index + batch_size))
        if max(indexes) > self.data_shape[0] - 1:
            dif = max(indexes) - self.data_shape[0]
            indexes[len(indexes) - dif - 1:len(indexes)] = list(range(dif + 1))
            self.index = batch_size - dif
            batch = self.df.iloc[indexes]
        else:
            batch = self.df.iloc[indexes]
            self.index += batch_size

        labels = batch[self.attack_names]

        batch = batch.drop(self.all_attack_names, axis=1)

        return batch, labels

    def get_full(self):

        labels = self.df[self.attack_names]

        batch = self.df.drop(self.all_attack_names, axis=1)

        return batch, labels
    
    def _load_df(self):
        if self.train_test == 'train':
            self.df = pd.read_csv(self.formated_train_path, sep=',')  # Read again the csv
        else:
            self.df = pd.read_csv(self.formated_test_path, sep=',')
        self.index = np.random.randint(0, self.df.shape[0] - 1, dtype=np.int32)
        self.loaded = True
        # Create a list with the existent attacks in the df
        for att in self.attack_map:
            if att in self.df.columns:
                # Add only if there is exist at least 1
                if np.sum(self.df[att].values) > 1:
                    self.attack_names.append(att)
        # self.headers = list(self.df)

class QNetwork():
    """
    Q-Network Estimator
    Represents the global model for the table
    """

    def __init__(self, obs_size, num_actions, hidden_size=100,
                 hidden_layers=1, learning_rate=.2, net_name='qv'):
        """
        Initialize the network with the provided shape
        """
        self.net_name = net_name
        self.obs_size = obs_size
        self.num_actions = num_actions

        # Network arquitecture
        self.model = Sequential()
        # Add imput layer
        self.model.add(Dense(hidden_size, input_shape=(obs_size,),
                             activation='relu'))
        # Add hidden layers
        for layers in range(hidden_layers):
            self.model.add(Dense(hidden_size, activation='relu'))
        # Add output layer
        if self.net_name == 'actor':
            self.model.add(Dense(num_actions, activation='softmax'))
        else:
            self.model.add(Dense(num_actions))

        optimizer = optimizers.Adam(0.00025)

        # Compilation of the model with optimizer and loss
        if self.net_name == 'actor':
            self.model.compile(loss=sac_loss, optimizer=optimizer)
        else:
            self.model.compile(loss=tf.keras.losses.mse, optimizer=optimizer)

    def predict(self, state, batch_size=1):
        """
        Predicts action values.
        """
        return self.model.predict(state, batch_size=batch_size, verbose=0)

    def update(self, states, q):
        """
        Updates the estimator with the targets.

        Args:
          states: Target states
          q: Estimated values

        Returns:
          The calculated loss on the batch.
        """
        loss = self.model.train_on_batch(states, q)
        return loss

    def copy_model(model):
        """Returns a copy of a keras model."""
        model.save('tmp_model')
        return tf.keras.models.load_model('tmp_model')
@tf.keras.utils.register_keras_serializable()
def sac_loss(y_true, y_pred):
    """ y_true 是 Q(*, action_n), y_pred 是 pi(*, action_n) """
    qs = 0.2 * tf.math.xlogy(y_pred, y_pred) - y_pred * y_true
    return tf.reduce_sum(qs, axis=-1)



class ReplayMemory(object):
    """Implements basic replay memory"""

    def __init__(self, observation_size, max_size):
        self.observation_size = observation_size
        self.num_observed = 0
        self.max_size = max_size
        self.samples = {
                 'obs'      : np.zeros(self.max_size * 1 * self.observation_size,
                                       dtype=np.float32).reshape(self.max_size,self.observation_size),
                 'action'   : np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
                 'reward'   : np.zeros(self.max_size * 1).reshape(self.max_size, 1),
                 'terminal' : np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
               }

    def observe(self, state, action, reward, done):
        index = self.num_observed % self.max_size
        self.samples['obs'][index, :] = state
        self.samples['action'][index, :] = action
        self.samples['reward'][index, :] = reward
        self.samples['terminal'][index, :] = done

        self.num_observed += 1

    def observe_batch(self, states, actions, rewards, dones):
        batch_size = len(states)
        indices = np.arange(self.num_observed, self.num_observed + batch_size) % self.max_size
        
        self.samples['obs'][indices] = states
        self.samples['action'][indices] = actions.reshape(-1, 1)
        self.samples['reward'][indices] = rewards.reshape(-1, 1)
        self.samples['terminal'][indices] = dones.reshape(-1, 1)
        
        self.num_observed += batch_size
        
    def sample_minibatch(self, minibatch_size):
        max_index = min(self.num_observed, self.max_size) - 1
        sampled_indices = np.random.randint(max_index, size=minibatch_size)

        s      = np.asarray(self.samples['obs'][sampled_indices, :], dtype=np.float32)
        s_next = np.asarray(self.samples['obs'][sampled_indices+1, :], dtype=np.float32)

        a      = self.samples['action'][sampled_indices].reshape(minibatch_size)
        r      = self.samples['reward'][sampled_indices].reshape((minibatch_size, 1))
        done   = self.samples['terminal'][sampled_indices].reshape((minibatch_size, 1))

        return (s, a, r, s_next, done)


'''
Reinforcement learning Agent definition
'''


class Agent(object):
    def __init__(self, actions, obs_size, **kwargs):
        self.actions = actions
        self.ddqn_time = 100
        self.ddqn_update = self.ddqn_time
        self.num_actions = len(actions)
        self.obs_size = obs_size
        self.alpha = 0.2
        self.net_learning_rate = 0.1
        self.gamma = kwargs.get('gamma', .001)
        self.minibatch_size = kwargs.get('minibatch_size', 2)
        self.epoch_length = kwargs.get('epoch_length', 100)
        self.ExpRep = kwargs.get('ExpRep', True)
        self.Type= kwargs.get("Type","Agent")
        self.decay_rate = kwargs.get("decay_rate", 0.99)
        if self.ExpRep:
            self.memory = ReplayMemory(self.obs_size, kwargs.get('mem_size', 10))

        self.actor_network = QNetwork(self.obs_size, self.num_actions,
                                      kwargs.get('hidden_size', 100),
                                      kwargs.get('hidden_layers', 1),
                                      kwargs.get('learning_rate', .2),
                                      net_name='actor')
        self.q0_network = QNetwork(self.obs_size, self.num_actions,
                                      kwargs.get('hidden_size', 100),
                                      kwargs.get('hidden_layers', 1),
                                      kwargs.get('learning_rate', .2))
        self.q1_network = QNetwork(self.obs_size, self.num_actions,
                                             kwargs.get('hidden_size', 100),
                                             kwargs.get('hidden_layers', 1),
                                             kwargs.get('learning_rate', .2))
        self.v_network = QNetwork(self.obs_size, 1,
                                             kwargs.get('hidden_size', 100),
                                             kwargs.get('hidden_layers', 1),
                                             kwargs.get('learning_rate', .2))
        self.target_v_network = QNetwork(self.obs_size, 1,
                                  kwargs.get('hidden_size', 100),
                                  kwargs.get('hidden_layers', 1),
                                  kwargs.get('learning_rate', .2))

        self.update_target_net(self.target_v_network.model,
                               self.v_network.model, self.net_learning_rate)

    def learn(self, states, actions, next_states, rewards, done):
        if self.ExpRep:
            self.memory.observe(states, actions, rewards, done)
        else:
            self.states = states
            self.actions = actions
            self.next_states = next_states
            self.rewards = rewards
            self.done = done


        
    def update_model_meta(self, K, H):
        if self.ExpRep:
            (states, actions, rewards, next_states, done) = self.memory.sample_minibatch(self.minibatch_size)
        else:
            states = self.states
            rewards = self.rewards
            next_states = self.next_states
            actions = self.actions
            done = self.done

        groups = {}
        for idx, a in enumerate(actions):
            groups.setdefault(a, []).append(idx)
        tasks = sorted(groups, key=lambda a: len(groups[a]), reverse=True)[:H]
        loss_meta = 0
        for task in tasks:
            indices = np.array(groups[task])
            if len(indices) >= K*2:
                chosen = np.random.choice(indices, K*2, replace=False)
            else:
                chosen = np.random.choice(indices, K*2, replace=True)

            states_meta = states[chosen]
            actions_meta = actions[chosen]
            rewards_meta = rewards[chosen]
            next_states_meta = next_states[chosen]
            done_meta = done[chosen]

            states_support, states_query = np.array_split(states_meta, 2)
            actions_support, actions_query = np.array_split(actions_meta, 2)
            rewards_support, rewards_query = np.array_split(rewards_meta, 2)
            next_states_support, next_states_query = np.array_split(next_states_meta, 2)
            done_support, done_query = np.array_split(done_meta, 2)
            
            loss_meta += self.meta_train(states_support, actions_support, next_states_support, rewards_support, done_support, K)
            self.meta_memory.observe_batch(states_query, actions_query, rewards_query, done_query)
            
        (states, actions, rewards, next_states, done) = self.meta_memory.sample_minibatch(K*H)

        pis = self.target_actor_network.predict(states, verbose =0)
        q0s = self.target_q0_network.predict(states, verbose =0)
        q1s = self.target_q1_network.predict(states, verbose =0) 
        
        loss = self.actor_network.model.train_on_batch(states, q0s)


        q01s = np.minimum(q0s, q1s)
        
        entropic_q01s = pis * q01s - self.alpha * tf.math.xlogy(pis, pis)

        v_network_target = tf.math.reduce_sum(entropic_q01s, axis=-1)
        self.v_network.model.fit(states, v_network_target, verbose=0)

        next_vs = self.target_v_network_target.predict(next_states, verbose =0)
        q_targets = rewards.reshape(q0s[range(K*H), actions].shape) + self.gamma * (1. - done.reshape(q0s[range(K*H), actions].shape)) * \
                    next_vs[:, 0]

        q0s[range(K*H), actions] = q_targets
        q1s[range(K*H), actions] = q_targets

        self.q0_network.model.fit(states, q0s, verbose=0)
        self.q1_network.model.fit(states, q1s, verbose=0)

        self.update_target_net(self.target_v_network.model,
                            self.v_network.model, self.net_learning_rate)
        return loss
    
    def update_model(self):
        if self.ExpRep:
            (states, actions, rewards, next_states, done) = self.memory.sample_minibatch(self.minibatch_size)
        else:
            states = self.states
            rewards = self.rewards
            next_states = self.next_states
            actions = self.actions
            done = self.done
        #print(self.q0_network.get_params(regularizable=True))

        pis = self.actor_network.predict(states)
        q0s = self.q0_network.predict(states)
        q1s = self.q1_network.predict(states) 

        loss = self.actor_network.model.train_on_batch(states, q0s)


        q01s = np.minimum(q0s, q1s)
        
        entropic_q01s = pis * q01s - self.alpha * tf.math.xlogy(pis, pis)
        v_targets = tf.math.reduce_sum(entropic_q01s, axis=-1)
        self.v_network.model.fit(states, v_targets, verbose=0)

        next_vs = self.target_v_network.predict(next_states)
        q_targets = rewards.reshape(q0s[range(self.minibatch_size), actions].shape) + self.gamma * (1. - done.reshape(q0s[range(self.minibatch_size), actions].shape)) * \
                    next_vs[:, 0]

        q0s[range(self.minibatch_size), actions] = q_targets
        q1s[range(self.minibatch_size), actions] = q_targets
        self.q0_network.model.fit(states, q0s, verbose=0)
        self.q1_network.model.fit(states, q1s, verbose=0)


        self.update_target_net(self.target_v_network.model,
                            self.v_network.model, self.net_learning_rate)
        return loss
    def init_target_network(self, K,H):  

        self.target_actor_network = tf.keras.models.clone_model(self.actor_network.model)
        self.target_actor_network.set_weights(self.actor_network.model.get_weights())
        self.target_actor_network.compile(optimizer=optimizers.Adam(0.00025),
                               loss=sac_loss)  
        self.target_q0_network = tf.keras.models.clone_model(self.q0_network.model)
        self.target_q0_network.set_weights(self.q0_network.model.get_weights())
        self.target_q0_network.compile(optimizer=optimizers.Adam(0.00025),
                               loss=tf.keras.losses.mse)  
        self.target_q1_network = tf.keras.models.clone_model(self.q1_network.model)
        self.target_q1_network.set_weights(self.q1_network.model.get_weights())
        self.target_q1_network.compile(optimizer=optimizers.Adam(0.00025),
                               loss=tf.keras.losses.mse)  
        self.v_network_target = tf.keras.models.clone_model(self.v_network.model)
        self.v_network_target.set_weights(self.v_network.model.get_weights())
        self.v_network_target.compile(optimizer=optimizers.Adam(0.00025),
                               loss=tf.keras.losses.mse) 
        self.target_v_network_target = tf.keras.models.clone_model(self.target_v_network.model)
        self.target_v_network_target.set_weights(self.target_v_network.model.get_weights())
        self.target_v_network_target.compile(optimizer=optimizers.Adam(0.00025),
                               loss=tf.keras.losses.mse)  
        self.meta_memory = ReplayMemory(self.obs_size, K*H)
    
    def meta_train(self, states, actions, next_states, def_reward, dones, K):

        self.target_actor_network.set_weights(self.actor_network.model.get_weights())
        self.target_q0_network.set_weights(self.q0_network.model.get_weights())
        self.target_q1_network.set_weights(self.q1_network.model.get_weights())
        self.v_network_target.set_weights(self.v_network.model.get_weights())
        self.target_v_network_target.set_weights(self.target_v_network.model.get_weights())

        pis = self.target_actor_network.predict(states, verbose =0)
        q0s = self.target_q0_network.predict(states, verbose =0)
        q1s = self.target_q1_network.predict(states, verbose =0) 

        loss_actor = self.target_actor_network.train_on_batch(states, q0s)
        q01s = np.minimum(q0s, q1s)
        
        entropic_q01s = pis * q01s - self.alpha * tf.math.xlogy(pis, pis)
        v_targets = tf.math.reduce_sum(entropic_q01s, axis=-1)
        self.v_network_target.fit(states, v_targets, verbose=0)

        next_vs = self.target_v_network_target.predict(next_states, verbose =0)
        q_targets = def_reward.reshape(q0s[range(K), actions].shape) + self.gamma * (1. - dones.reshape(q0s[range(K), actions].shape)) * \
                    next_vs[:, 0]

        q0s[range(K), actions] = q_targets
        q1s[range(K), actions] = q_targets
        self.target_q0_network.fit(states, q0s, verbose=0)
        self.target_q1_network.fit(states, q1s, verbose=0)

        self.update_target_net(self.target_v_network_target,
                           self.v_network_target, 0.000125)
        return loss_actor

    def act(self, state, policy):
        raise NotImplementedError

    def update_target_net(self, target_net, evaluate_net, learning_rate=1.):
        target_weights = target_net.get_weights()
        evaluate_weights = evaluate_net.get_weights()
        average_weights = [(1. - learning_rate) * t + learning_rate * e
                for t, e in zip(target_weights, evaluate_weights)]
        target_net.set_weights(average_weights)

class DefenderAgent(Agent):
    def __init__(self, actions, obs_size, **kwargs):
        super().__init__(actions, obs_size, **kwargs)

    def act(self, states):
        # Get actions under the policy
        probs = self.actor_network.model.predict(states,verbose=0)[0]
        actions = np.random.choice(self.actions, size=1, p=probs)
        return actions


class AttackAgent(Agent):
    def __init__(self, actions, obs_size, **kwargs):
        super().__init__(actions, obs_size, **kwargs)

    def act(self, states):
        # Get actions under the policy
        probs = self.actor_network.model.predict(states,verbose=0)[0]
        actions = np.random.choice(self.actions, size=1, p=probs)
        return actions


'''
Reinforcement learning Enviroment Definition
'''


class RLenv(data_cls):
    def __init__(self, train_test, **kwargs):
        data_cls.__init__(self, train_test)
        data_cls._load_df(self)
        self.data_shape = data_cls.get_shape(self)
        self.batch_size = kwargs.get('batch_size', 1)  # experience replay -> batch = 1
        self.iterations_episode = kwargs.get('iterations_episode', 10)

    '''
    _update_state: function to update the current state
    Returns:
        None
    Modifies the self parameters involved in the state:
        self.state and self.labels
    Also modifies the true labels to get learning knowledge
    '''

    def _update_state(self):
        self.states, self.labels = data_cls.get_batch(self)

        # Update statistics
        self.true_labels += np.sum(self.labels).values

    '''
    Returns:
        + Observation of the enviroment
    '''

    def reset(self):
        # Statistics
        self.def_true_labels = np.zeros(len(self.attack_types), dtype=int)
        self.def_estimated_labels = np.zeros(len(self.attack_types), dtype=int)
        self.att_true_labels = np.zeros(len(self.attack_names), dtype=int)

        self.state_numb = 0

        self.states, self.labels = data_cls.get_batch(self, self.batch_size)

        self.total_reward = 0
        self.steps_in_episode = 0
        return self.states.values

    '''
    Returns:
        State: Next state for the game
        Reward: Actual reward
        done: If the game ends (no end in this case)

    In the adversarial enviroment, it's only needed to return the actual reward
    '''

    def act(self, defender_actions, attack_actions):
        # Clear previous rewards
        self.att_reward = np.zeros(len(attack_actions))
        self.def_reward = np.zeros(len(defender_actions))

        attack = [self.attack_types.index(self.attack_map[self.attack_names[att]]) for att in attack_actions]
        ATT_NUM[attack[0]] += 1
        if attack[0] in [1,2]:
            self.def_reward = (np.asarray(defender_actions) == np.asarray(attack)) * 2
            self.att_reward = (np.asarray(defender_actions) != np.asarray(attack)) * 2
        elif attack[0] in [3]:
            self.def_reward = (np.asarray(defender_actions) == np.asarray(attack)) * 2
            self.att_reward = (np.asarray(defender_actions) != np.asarray(attack)) * 2
        else:
            self.def_reward = (np.asarray(defender_actions) == np.asarray(attack)) * 1
            self.att_reward = (np.asarray(defender_actions) != np.asarray(attack)) * 1

        self.def_estimated_labels += np.bincount(defender_actions, minlength=len(self.attack_types))
        # TODO
        # list comprehension

        for act in attack_actions:
            self.def_true_labels[self.attack_types.index(self.attack_map[self.attack_names[act]])] += 1

        # Get new state and new true values
        attack_actions = attacker_agent.act(self.states)
        self.states = env.get_states(attack_actions)

        # Done allways false in this continuous task
        self.done = np.zeros(len(attack_actions), dtype=bool)

        return self.states, self.def_reward, self.att_reward, attack_actions, self.done

    '''
    Provide the actual states for the selected attacker actions
    Parameters:
        self:
        attacker_actions: optimum attacks selected by the attacker
            it can be one of attack_names list and select random of this
    Returns:
        State: Actual state for the selected attacks
    '''

    def get_states(self, attacker_actions):
        first = True
        for attack in attacker_actions:
            if first:
                minibatch = (self.df[self.df[self.attack_names[attack]] == 1].sample(1))
                first = False
            else:
                minibatch = minibatch.append(self.df[self.df[self.attack_names[attack]] == 1].sample(1))

        self.labels = minibatch[self.attack_names]
        minibatch.drop(self.all_attack_names, axis=1, inplace=True)
        self.states = minibatch

        return self.states

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":
    # Train batch
    batch_size = 1
    # batch of memory ExpRep
    minibatch_size = 180
    ExpRep = True
    formated_train_path = "formated_train_awid2.data"
    formated_test_path = "formated_test_awid2.data"
    iterations_episode = 180

    # Initialization of the enviroment
    env = RLenv('train',train_path=train_path, test_path=test_path,
                formated_train_path=formated_train_path,
                formated_test_path=formated_test_path, batch_size=batch_size,
                iterations_episode=iterations_episode)
    # obs_size = size of the state
    obs_size = env.data_shape[1] - len(env.all_attack_names)
    env_val = RLenv('test'
                    )
    
    num_episodes = 400

    '''
    Definition for the defensor agent.
    '''
    defender_valid_actions = list(range(len(env.attack_types)))  # only detect type of attack
    defender_num_actions = len(defender_valid_actions)

    def_gamma = 0.001
    def_decay_rate = 0.99

    K=100
    H=3

    def_hidden_size = 100
    def_hidden_layers = 2

    def_learning_rate = .2

    defender_agent = DefenderAgent(defender_valid_actions, obs_size,
                                   epoch_length=iterations_episode,
                                   decay_rate=def_decay_rate,
                                   gamma=def_gamma,
                                   hidden_size=def_hidden_size,
                                   hidden_layers=def_hidden_layers,
                                   minibatch_size=minibatch_size,
                                   mem_size=1000,
                                   learning_rate=def_learning_rate,
                                   ExpRep=ExpRep, Type= "Defender")
    # Pretrained defender
    # defender_agent.model_network.model.load_weights("models/type_model.h5")
    defender_agent.init_target_network(K,H)
    '''
    Definition for the attacker agent.
    In this case the exploration is better to be greater
    The correlation sould be greater too so gamma bigger
    '''
    attack_valid_actions = list(range(len(env.attack_names)))
    attack_num_actions = len(attack_valid_actions)

    att_gamma = 0.001
    att_decay_rate = 0.99

    att_hidden_layers = 2
    att_hidden_size = 100

    att_learning_rate = 0.2

    attacker_agent = AttackAgent(attack_valid_actions, obs_size,
                                 epoch_length=iterations_episode,
                                 decay_rate=att_decay_rate,
                                 gamma=att_gamma,
                                 hidden_size=att_hidden_size,
                                 hidden_layers=att_hidden_layers,
                                 minibatch_size=minibatch_size,
                                 mem_size=1000,
                                 learning_rate=att_learning_rate,
                                 ExpRep=ExpRep,Type= "Attacker")

    # Statistics
    att_reward_chain = []
    def_reward_chain = []
    att_loss_chain = []
    def_loss_chain = []
    def_total_reward_chain = []
    att_total_reward_chain = []

    # Print parameters
    print("-------------------------------------------------------------------------------")
    print("Total epoch: {} | Iterations in epoch: {}"
          "| Minibatch from mem size: {} | Total Samples: {}|".format(num_episodes,
                                                                      iterations_episode, minibatch_size,
                                                                      num_episodes * iterations_episode))
    print("-------------------------------------------------------------------------------")
    print("Dataset shape: {}".format(env.data_shape))
    print("-------------------------------------------------------------------------------")
    print("Attacker parameters: Num_actions={} | gamma={} |"
          " ANN hidden size={} | "
          "ANN hidden layers={}|".format(attack_num_actions,
                                         att_gamma, att_hidden_size,
                                         att_hidden_layers))
    print("-------------------------------------------------------------------------------")
    print("Defense parameters: Num_actions={} | gamma={} | "
          " ANN hidden size={} |"
          " ANN hidden layers={}|".format(defender_num_actions,
                                          def_gamma, def_hidden_size,
                                          def_hidden_layers))
    print("-------------------------------------------------------------------------------")

    # Main loop
    attacks_by_epoch = []
    attack_labels_list = []
    for epoch in range(num_episodes):
         start_time = time.time()
         att_loss = 0.
         def_loss = 0.
         def_total_reward_by_episode = 0
         att_total_reward_by_episode = 0
         states = env.reset()
         states = np.array(states, dtype=np.float32)
         attack_actions = attacker_agent.act(states)
         states = env.get_states(attack_actions)
         done = False
         attacks_list = []

         for i_iteration in range(iterations_episode):
             attacks_list.append(attack_actions[0])
             act_time = time.time()
             defender_actions = defender_agent.act(states)
             next_states, def_reward, att_reward, next_attack_actions, done = env.act(defender_actions, attack_actions)
             attacker_agent.learn(states, attack_actions, next_states, att_reward, done)
             defender_agent.learn(states, defender_actions, next_states, def_reward, done)
             act_end_time = time.time()

             if ExpRep and epoch * iterations_episode + i_iteration >= minibatch_size:
                 if i_iteration %5 ==0:
                    def_loss += defender_agent.update_model_meta(K,H)
                    att_loss += attacker_agent.update_model()
                 else:
                    def_loss += defender_agent.update_model() 
                    att_loss += attacker_agent.update_model()
             elif not ExpRep:
                 if i_iteration %5 ==0:
                        def_loss += defender_agent.update_model_meta(K,H)
                        att_loss += attacker_agent.update_model()
                 else:
                        def_loss += defender_agent.update_model() 
                        att_loss += attacker_agent.update_model()
             update_end_time = time.time()
     
             states = next_states
             attack_actions = next_attack_actions   
             def_total_reward_by_episode += np.sum(def_reward, dtype=np.int32)
             att_total_reward_by_episode += np.sum(att_reward, dtype=np.int32)
    
         attacks_by_epoch.append(attacks_list)
         def_reward_chain.append(def_total_reward_by_episode)
         att_reward_chain.append(att_total_reward_by_episode)
         def_loss_chain.append(def_loss)
         att_loss_chain.append(att_loss)
    
         end_time = time.time()

         print("\r\n|Epoch {:03d}/{:03d}| time: {:2.2f}|\r\n"
               "|Def Loss {:4.4f} | Def Reward in ep {:03d}|\r\n"
               "|Att Loss {:4.4f} | Att Reward in ep {:03d}|"
               .format(epoch, num_episodes, (end_time - start_time),
                       def_loss, def_total_reward_by_episode,   
                       att_loss, att_total_reward_by_episode))
    
         print("|Def Estimated: {}| Att Labels: {}".format(env.def_estimated_labels,
                                                           env.def_true_labels))
         attack_labels_list.append(env.def_true_labels)
        
    if not os.path.exists('models'):
         os.makedirs('models')

    defender_agent.actor_network.model.save_weights("models/defender_agent_awid_model2.weights.h5", overwrite=True)
    with open("models/defender_agent_awid_model2.json", "w") as outfile:
         json.dump(defender_agent.actor_network.model.to_json(), outfile)

    with open("models/defender_agent_awid_model2.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("models/defender_agent_awid_model2.weights.h5")

    model.compile(loss=sac_loss, optimizer="adam")

    env_test = RLenv('test')

    total_reward = 0

    true_labels = np.zeros(len(env_test.attack_types), dtype=int)
    estimated_labels = np.zeros(len(env_test.attack_types), dtype=int)
    estimated_correct_labels = np.zeros(len(env_test.attack_types), dtype=int)
    states, labels = env_test.get_full()

    start_time = time.time()
    q = model.predict(states,verbose=0)
    actions = np.argmax(q, axis=1)
    end_time = time.time()
    print("predit time:", end_time -start_time)
    maped = []
    for indx, label in labels.iterrows():
        maped.append(env_test.attack_types.index(env_test.attack_map[label.idxmax()]))

    labels, counts = np.unique(maped, return_counts=True)
    true_labels[labels] += counts

    for indx, a in enumerate(actions):
        estimated_labels[a] += 1
        if a == maped[indx]:
            total_reward += 1
            estimated_correct_labels[a] += 1

    action_dummies = pd.get_dummies(actions)
    posible_actions = np.arange(len(env_test.attack_types))
    for non_existing_action in posible_actions:
        if non_existing_action not in action_dummies.columns:
            action_dummies[non_existing_action] = np.uint8(0)
    labels_dummies = pd.get_dummies(maped)

    normal_f1_score = f1_score(labels_dummies[0].values, action_dummies[0].values)
    flooding_f1_score = f1_score(labels_dummies[1].values, action_dummies[1].values)
    injection_f1_score = f1_score(labels_dummies[2].values, action_dummies[2].values)
    impersonation_f1_score = f1_score(labels_dummies[3].values, action_dummies[3].values)

    Accuracy = [normal_f1_score, flooding_f1_score, injection_f1_score, impersonation_f1_score]
    Mismatch = estimated_labels - true_labels

    acc = float(100 * total_reward / len(states))
    print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {:.2f}%'.format(total_reward,
                                                                                     len(states), acc))
    outputs_df = pd.DataFrame(index=env_test.attack_types, columns=["Estimated", "Correct", "Total", "F1_score"])
    for indx, att in enumerate(env_test.attack_types):
        outputs_df.iloc[indx].Estimated = estimated_labels[indx]
        outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
        outputs_df.iloc[indx].Total = true_labels[indx]
        outputs_df.iloc[indx].F1_score = Accuracy[indx] * 100
        outputs_df.iloc[indx].Mismatch = abs(Mismatch[indx])

    print(outputs_df)

    aggregated_data_test = np.array(maped)

    print('Performance measures on Test data')
    print('Accuracy =  {:.4f}'.format(accuracy_score(aggregated_data_test, actions)))
    print('F1 =  {:.4f}'.format(f1_score(aggregated_data_test, actions, average='weighted')))
    print('Precision_score =  {:.4f}'.format(precision_score(aggregated_data_test, actions, average='weighted')))
    print('recall_score =  {:.4f}'.format(recall_score(aggregated_data_test, actions, average='weighted')))

    cnf_matrix = confusion_matrix(aggregated_data_test, actions)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=env.attack_types, normalize=True,
                           title='Normalized confusion matrix')
    plt.savefig('confusion_matrix_adversarial.svg', format='svg', dpi=1000)
