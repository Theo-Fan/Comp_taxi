# -*- coding:utf-8  -*-
# Time  : 2021/5/31 下午4:14
# Author: Yahui Cui

"""
# =================================== Important =========================================
Notes:
1. this agent is random agent , which can fit any env in Jidi platform.
2. if you want to load .pth file, please follow the instruction here:
https://github.com/jidiai/ai_lib/blob/master/examples/demo
"""


def my_controller(observation, action_space, is_act_continuous=False):
    """ 
    Parameters:
        observation: {
            'obs': {'agent_id': 'government', 'raw_obs': array([-1., -1., -1., -1., -1., -1.])}, 
            'controlled_player_index': 1
        }
        action_space: [Box(-1.0, 1.0, (5,), float32)]
        len(action_space): 1
    """

    agent_action = []
    # range(len(action_space)):  因为每个agent可能存在多个动作, 因此需要遍历动作空间(可以合并)
    # action_space = [
    #     Box(low=-1, high=1, shape=(2,)),  # 第一个动作空间（2维）
    #     Box(low=-1, high=1, shape=(3,)),  # 第二个动作空间（3维）
    #     Discrete(4)                        # 第三个动作空间（离散）
    # ])
    for i in range(len(action_space)): 
        action_ = sample_single_dim(action_space[i], is_act_continuous) # 由于使用的是 random policy, 因此没用到 observation
        agent_action.append(action_)
    return agent_action


def sample_single_dim(action_space_list_each, is_act_continuous):
    each = []
    if is_act_continuous:
        each = action_space_list_each.sample()
    else:
        if action_space_list_each.__class__.__name__ == "Discrete":
            each = [0] * action_space_list_each.n
            idx = action_space_list_each.sample()
            each[idx] = 1
        elif action_space_list_each.__class__.__name__ == "MultiDiscreteParticle":
            each = []
            nvec = action_space_list_each.high - action_space_list_each.low + 1
            sample_indexes = action_space_list_each.sample()

            for i in range(len(nvec)):
                dim = nvec[i]
                new_action = [0] * dim
                index = sample_indexes[i]
                new_action[index] = 1
                each.extend(new_action)
    return each


def sample(action_space_list_each, is_act_continuous):
    player = []
    if is_act_continuous:
        for j in range(len(action_space_list_each)):
            each = action_space_list_each[j].sample()
            player.append(each)
    else:
        player = []
        for j in range(len(action_space_list_each)):
            # each = [0] * action_space_list_each[j]
            # idx = np.random.randint(action_space_list_each[j])
            if action_space_list_each[j].__class__.__name__ == "Discrete":
                each = [0] * action_space_list_each[j].n
                idx = action_space_list_each[j].sample()
                each[idx] = 1
                player.append(each)
            elif action_space_list_each[j].__class__.__name__ == "MultiDiscreteParticle":
                each = []
                nvec = action_space_list_each[j].high
                sample_indexes = action_space_list_each[j].sample()

                for i in range(len(nvec)):
                    dim = nvec[i] + 1
                    new_action = [0] * dim
                    index = sample_indexes[i]
                    new_action[index] = 1
                    each.extend(new_action)
                player.append(each)
    return player