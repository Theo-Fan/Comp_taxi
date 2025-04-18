
from env.env_core import economic_society
from omegaconf import OmegaConf

yaml_cfg = OmegaConf.load(f'./cfg/default.yaml')  # get environment parameters

# create TaxAI env
env = economic_society(yaml_cfg.Environment)

# get the action max
gov_action_max = env.government.action_space.high[0]

house_action_max = env.households.action_space.high[0]

# env initialization
# global obs is observed by gov & households;
# Private obs are observed separately by each household.
global_obs, private_obs = env.reset()

for _ in range(100):
    # randomly sample actions for the government and household agents
    gov_action = env.government.action_space.sample()
    house_action = env.households.action_space.sample()

    action = {env.government.name: gov_action * gov_action_max,
              env.households.name: house_action * house_action_max}
    # env step
    next_global_obs, next_private_obs, gov_reward, house_reward, done = env.step(action)
    print("gov reward:", gov_reward, "\nhouseholds reward:", house_reward)

    if done:
        global_obs, private_obs = env.reset()
        
env.close()












