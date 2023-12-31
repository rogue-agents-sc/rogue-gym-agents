# rogue-gym-agents

Install Python3.8

```
python3.8 -m venv 38env
source 38env/bin/activate
```

Install Rainy

```
git clone git@github.com:rogue-agents-sc/Rainy.git
cd Rainy/
git checkout create-trajectories
pip install numpy=1.20.3 click torch
```

Install RogueGym

```
git clone git@github.com:rogue-agents-sc/rogue-gym.git
cd rogue-gym/
Follow README.md
```

Generate RogueGym episodes for Model Training:

```
git clone git@github.com:rogue-agents-sc/rogue-gym-agents.git
cd rogue-gym-agents/datasets/
python episode_generator.py episodes --model pretrained/rainy-agent.pth --num-episodes 20 --reward-thresh 50 episodes/

```

Train and Run DecisionTransformer

```
cd ..
python agents/decision_transformer.py
```

#Running Google Colab Files
If you want to run the colab files, you will have to allow colab to run git commands
You can use the python whl file located here https://drive.google.com/file/d/1eOHxv2MmOsFVLkaJQ-Oe1h4r6Ao2H56F/view?usp=drive_link
