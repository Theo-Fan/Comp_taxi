from HeterTaxAI.entities.base import BaseEntity
import numpy as np
import copy
import math
import pandas as pd
import random
# import quantecon as qe
import matplotlib.pyplot as plt
import os,sys

from gym.spaces import Box

class Household(BaseEntity):
    name='Household'

    def __init__(self, entity_args):
        super().__init__()
        self.n_households = entity_args['n']
        # fixed hyperparameter
        self.CRRA = entity_args['CRRA']                                    #theta
        self.IFE = entity_args['IFE']                                      #inverse Frisch Elasticity
        self.eta = entity_args['eta']                                       # if eta=0, the transitory shocks are additive, if eta = 1, they are multiplicative
        self.e_p = entity_args['e_p']
        self.e_q = entity_args['e_q']
        self.rho_e = entity_args['rho_e']
        self.sigma_e = entity_args['sigma_e']
        self.super_e = entity_args['super_e']

        self.action_dim = entity_args['action_shape']

        self.weight, self.real_asset, self.real_e, self.real_income = self.get_real_data()
        self.households_init()
        self.reset()
        self.action_space = Box(
            low=-1, high=1, shape=(self.n_households, self.action_dim), dtype=np.float32
        )


    def e_initial(self, n):
        self.e_array = np.zeros((n, 2))  # super-star and normal
        # initialize as normal state
        random_set = np.random.rand(n)
        self.e_array[:, 0] = (random_set > self.e_p).astype(int) * self.e_init.flatten()
        self.e_array[:, 1] = (random_set < self.e_p).astype(int)
        self.e = np.sum(self.e_array, axis=1, keepdims=True)
        
        self.e_0 = copy.copy(self.e)
        self.e_array_0 = copy.copy(self.e_array)

    def generate_e_ability(self):
        """
        Generates n current ability levels for a given time step t.
        """
        self.e_past = copy.copy(self.e_array)
        e_past_mean = sum(self.e_past[:,0])/np.count_nonzero(self.e_past[:,0])
        for i in range(self.n_households):
            is_superstar = (self.e_array[i,1]>0).astype(int)
            if is_superstar == 0:
                # normal state
                if np.random.rand() < self.e_p:
                    # transit from normal to super-star
                    self.e_array[i, 0] = 0
                    self.e_array[i, 1] = self.super_e * e_past_mean
                else:
                    # remain in normal
                    self.e_array[i, 1] = 0
                    self.e_array[i, 0] = np.exp(self.rho_e * np.log(self.e_past[i, 0]) + self.sigma_e * np.random.randn())
            else:
                # super state
                if np.random.rand() < self.e_q:
                    # remain in super-star
                    self.e_array[i, 0] = 0
                    self.e_array[i, 1] = self.super_e * e_past_mean
                else:
                    # transit to normal
                    self.e_array[i, 1] = 0
                    self.e_array[i, 0] = random.uniform(self.e_array[:,0].min(), self.e_array[:,0].max())  # 随机来一个
        self.e = np.sum(self.e_array, axis=1, keepdims=True)


    def reset(self, **custom_cfg):
        # self.households_init()
        self.e = copy.copy(self.e_0)
        self.e_array = copy.copy(self.e_array_0)
        self.generate_e_ability()
        self.at = copy.copy(self.at_init)
        self.at_next = copy.copy(self.at)
        self.income = copy.copy(self.it_init)


    def households_init(self):
        self.at_init,self.e_init,self.it_init = self.sample_real_data()
        self.e_initial(self.n_households)
        

    def get_real_data(self):
        df = pd.read_csv(os.path.abspath('.')+ "/HeterTaxAI/agents/data/advanced_scfp2022.csv")
        weight = df['WGT'].values
        asset = df['ASSET'].values
        educ = df['EDUC'].values
        income = df['INCOME'].values
        return weight / np.sum(weight) ,asset,educ,income
    def sample_real_data(self):
        index = np.random.choice(range(len(self.real_asset)), self.n_households, p=self.weight)
        batch_asset = self.real_asset[index]
        batch_e = self.real_e[index]
        batch_income = self.real_income[index]
        return batch_asset.reshape(self.n_households, 1), batch_e.reshape(self.n_households, 1), batch_income.reshape(self.n_households, 1)
    
    def close(self):
        pass




