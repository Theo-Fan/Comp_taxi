# -*- coding:utf-8  -*-
import os
import time
import json
import numpy as np
import argparse
import sys
import torch

sys.path.append("./olympics_engine")

from pprint import pprint
from env.chooseenv import make
from utils.get_logger import get_logger
from env.obs_interfaces.observation import obs_type


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_players_and_action_space_list(g):
    if sum(g.agent_nums) != g.n_player:
        raise Exception("agent number = %d 不正确, 与n_player = %d 不匹配" % (sum(g.agent_nums), g.n_player))

    # 将智能体数量列表转化为累加形式
    n_agent_num = list(g.agent_nums)
    for i in range(1, len(n_agent_num)):
        n_agent_num[i] += n_agent_num[i - 1]

    # 根据 agent number 分配 player id
    players_id = []
    actions_space = []
    for policy_i in range(len(g.obs_type)):
        if policy_i == 0:
            players_id_list = range(n_agent_num[policy_i])
        else:
            players_id_list = range(n_agent_num[policy_i - 1], n_agent_num[policy_i])
        players_id.append(players_id_list)

        action_space_list = [g.get_single_action_space(player_id) for player_id in players_id_list]
        actions_space.append(action_space_list)

    return players_id, actions_space


def get_joint_action_eval(game, multi_part_agent_ids, policy_list, actions_spaces, all_observes):
    """
        Parameters:
            1.  multi_part_agent_ids[i]: 策略 i 控制的 agent id 列表
            2.  policy_list: 策略列表
            3.  actions_spaces[i]: 策略 i 控制的 agent 的动作空间, actions_spaces[i][j]表示第 i 个策略控制的第 j 个 agent 的动作空间
                    [
                        [Box(-1.0, 1.0, (5,), float32)],
                        ...,
                    ]
            4.  all_observes: 所有 agent 的观察值 == 每个 multi_part_agent_ids[i] 中包含的 agent 数量和
                    [
                        [{
                            'obs': {
                                'agent_id': 'government',
                                'raw_obs': array([-1., -1., -1., -1., -1., -1.])
                            },
                            'controlled_player_index': 1 # 当前控制的 agent idx
                        }],
                        ...,
                    ]
    """
    if len(policy_list) != len(game.agent_nums):
        error = "模型个数%d与玩家个数%d维度不正确! " % (len(policy_list), len(game.agent_nums))
        raise Exception(error)
    # [[[0, 0, 0, 1]], [[0, 1, 0, 0]]]
    joint_action = []
    for policy_i in range(len(policy_list)):
        if game.obs_type[policy_i] not in obs_type:
            raise Exception("可选obs类型: %s" % str(obs_type))

        agents_id_list = multi_part_agent_ids[policy_i] # 策略 i 控制的 agent id 列表
        action_space_list = actions_spaces[policy_i] # 策略 i 控制的 agent 的动作空间
        function_name = 'm%d' % policy_i

        ##### +++++ modify
        # print(f"function_name: {function_name}")

        for i in range(len(agents_id_list)):
            agent_id = agents_id_list[i] # 策略 i 控制的第 i 个 agent id
            a_obs = all_observes[agent_id] # 策略 i 控制的第 i 个 agent 的观察值

            # print(a_obs)
            # print(f"action_space_list[{i}]: {action_space_list[i]}")
            # print(f"game.is_act_continuous: {game.is_act_continuous}")

            """
                inputs Shape(6, ): 'raw_obs': array([-1., -1., -1., -1., -1., -1.])} 
                Outputs Shape(5, ): [Box(-1.0, 1.0, (5,), float32)]
            """

            each = eval(function_name)(a_obs, action_space_list[i], game.is_act_continuous)
            # print(f"each: {each}")
            joint_action.append(each)


        # print(f"joint_action: {joint_action}")
        # sys.exit()
        ##### +++++ modify

    return joint_action


def set_seed(g, env_name):
    if env_name.split("-")[0] in ['magent']:
        g.reset()
        seed = g.create_seed()
        g.set_seed(seed)


def run_game(g, env_name, multi_part_agent_ids, actions_spaces, policy_list, render_mode):
    """
    This function is used to generate log for Vue rendering. Saves .json file
    """
    log_path = os.getcwd() + '/logs/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger = get_logger(log_path, g.game_name, json_file=render_mode)
    set_seed(g, env_name)

    ##### +++++ modify(customize policy)
    for i in range(len(policy_list)):
        if policy_list[i] not in get_valid_agents():
            raise Exception("agent {} not valid!".format(policy_list[i]))

        file_path = os.path.dirname(os.path.abspath(__file__)) + "/agents/" + policy_list[i] + "/submission.py"
        if not os.path.exists(file_path):
            raise Exception("file {} not exist!".format(file_path))

        import_path = '.'.join(file_path.split('/')[-3:])[:-3]
        function_name = 'm%d' % i
        import_name = "my_controller"
        import_s = "from %s import %s as %s" % (import_path, import_name, function_name)
        print(import_s)
        exec(import_s, globals())
    ##### +++++ modify(customize policy)

    st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info = {
        "game_name": env_name,
        "n_player": g.n_player,
        "board_height": g.board_height if hasattr(g, "board_height") else None,
        "board_width": g.board_width if hasattr(g, "board_width") else None,
        "init_info": g.init_info,
        "start_time": st,
        "mode": "terminal",
        "seed": g.seed if hasattr(g, "seed") else None,
        "map_size": g.map_size if hasattr(g, "map_size") else None
    }

    steps = []
    all_observes = g.all_observes
    done = False
    ##### modify ===?
    max_episodes = 200  # 设置最大训练次数
    
    # 外层for循环控制训练回合数
    for episode in range(1, max_episodes + 1):
        # 重置环境
        g.reset()
        all_observes = g.all_observes
        done = False
        step_cnt = 1
        
        
        # 内层while循环控制每个回合的步数
        while not g.is_terminal():
            # step = "step%d" % step_cnt
            # if step_cnt % 100 == 0:
            #     print(f"Episode {episode}/{max_episodes}, {step}")

            if render_mode and hasattr(g, "env_core"):
                if hasattr(g.env_core, "render"):
                    g.env_core.render()
            elif render_mode and hasattr(g, 'render'):
                g.render()

            info_dict = {"time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}
            joint_act = get_joint_action_eval(g, multi_part_agent_ids, policy_list, actions_spaces, all_observes)
            all_observes, reward, done, info_before, info_after = g.step(joint_act)

            if env_name.split("-")[0] in ["magent"]:
                info_dict["joint_action"] = g.decode(joint_act)
            if info_before:
                info_dict["info_before"] = info_before
            info_dict["reward"] = reward
            if info_after:
                info_dict["info_after"] = info_after
            steps.append(info_dict)
            
            step_cnt += 1
        print(f"Episode {episode}/{max_episodes} finished")
    ##### modify ===

    game_info["steps"] = steps
    game_info["winner"] = g.check_win()
    game_info["winner_information"] = g.won
    game_info["n_return"] = g.n_return
    ed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info["end_time"] = ed
    logs = json.dumps(game_info, ensure_ascii=False, cls=NpEncoder)
    logger.info(logs)
    
    # 保存最终模型
    ##### modify ===
    if 'ppo' in policy_list:
        try:
            from agents.ppo.submission import ppo_agent
            if ppo_agent is not None:
                ppo_agent.save_models("final")
                print("Final models saved successfully!")
        except Exception as e:
            print(f"Error saving final models: {e}")
    ##### modify ===

def get_valid_agents():
    dir_path = os.path.join(os.path.dirname(__file__), 'agents')
    return [f for f in os.listdir(dir_path) if f != "__pycache__"]


if __name__ == "__main__":
    env_type = "taxing_gov_heter"
    # env_type = "taxing_households_heter"
    game = make(env_type, seed=None)
    render_mode = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default="random", help="random")

    args = parser.parse_args()
    _policy_list = [args.my_ai]
    policy_list = _policy_list[:len(game.agent_nums)]

    multi_part_agent_ids, actions_space = get_players_and_action_space_list(game)
    # policy_list = ['ppo', 'random']

    # multi_part_agent_ids: 每个策略控制的智能体id列表。
    # 例如: [range(0, 1)]表示策略 0 控制智能体 0
    # multi_part_agent_ids = [
    #     [0, 1],  # 策略 0 控制 agent 0 和 1
    #     [2]      # 策略 1 控制 agent 2
    # ]

    # actions_spaces = [
    #     [Box(...), Box(...)],   # 策略 0 控制的两个 agent 的动作空间
    #     [Discrete(...)]         # 策略 1 控制的一个 agent 的动作空间
    # ]

    print("policy_list:", policy_list)
    print("multi_part_agent_ids:", multi_part_agent_ids)
    print("actions_space:", actions_space)

    run_game(game, env_type, multi_part_agent_ids, actions_space, policy_list, render_mode)
