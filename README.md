<img src="imgs/Jidi%20logo.png" width='300px'> 

# AAMAS 2025 Computational Economics Competition

This repo provide the source code for the [AAMAS 2025 Computational Economics Competition ](http://jidiai.cn/aamas_tax_2025/)



## Multi-Agent Game Evaluation Platform --- Jidi (及第)
Jidi supports online evaluation service for various games/simulators/environments/testbeds. Website: [www.jidiai.cn](www.jidiai.cn).

A tutorial on Jidi: [Tutorial](https://github.com/jidiai/ai_lib/blob/master/assets/Jidi%20tutorial.pdf)


## Environment
The competition adopts a Taxing simulator [TaxAI](https://github.com/jidiai/TaxAI/). A brief description can be found on [JIDI](http://www.jidiai.cn/env_detail?envid=99).
A complementary document is also presented in [docs](./docs/). 

The game contains four roles, two of which are controllable. They are:
- A Government  (controllable)
- 100 Households  (controllable)
- Firm
- Bank

Government try to adjust tax parameters and the ratio of government spending to GDP, in order to optimize GDP and Gini index. The evaluation metric is the sum of sigmoid reward at each time step.

Each household try to balance saving ratio and working time, in order to optimize personal utility. The evaluation metric is the sum of individual at each time step.

Paper: 
- [TaxAI: A Dynamic Economic Simulator and Benchmark for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2309.16307)
- [Learning Macroeconomic Policies based on Microfoundations: A Stackelberg Mean Field Game Approach](https://arxiv.org/abs/2403.12093)



## Quick Start

You can use any tool to manage your python environment. Here, we use conda as an example.

```bash
conda create -n taxingai-venv python==3.9  #3.10
conda activate taxingai-venv
```

Next, clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/jidiai/Competition_TaxingAI.git
cd Competition_TaxingAI
pip install -r requirements.txt
conda install numba==0.58.0
```

Finally, run the game by executing:
```bash
python run_log.py
```

## Navigation

```
|-- Competition_OvercookedAI               
	|-- agents                              // Agents that act in the environment
	|	|-- random                      // A random agent demo
	|	|	|-- submission.py       // A ready-to-submit random agent file
	|-- env		                        // scripts for the environment
	|	|-- config.py                   // environment configuration file
	|	|-- taxing_gov.py               // The environment wrapper for taxing_gov env	
	|   |-- taxing_household.py             // The environment wrapper for taxing_household env	      
	|-- utils               
	|-- run_log.py		                // run the game with provided agents (same way we evaluate your submission in the backend server)
```



## How to test submission

- You can train your own agents using any framework you like as long as using the provided environment wrapper. 

- For your ready-to-submit agent, make sure you check it using the ``run_log.py`` scrips, which is exactly how we 
evaluate your submission.

- ``run_log.py`` takes agents from path `agents/` and run a game. For example:

>python run_log.py --my_ai "random" 

set both agents as a random policy and run a game.

- You can put your agents in the `agent/` folder and create a `submission.py` with a `my_controller` function 
in it. Then run the `run_log.py` to test:

>python run_log.py --my_ai your_agent_name 

- If you pass the test, then you can submit it to the Jidi platform. You can make multiple submission and the previous submission will
be overwritten.


