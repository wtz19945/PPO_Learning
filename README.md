# PPO for Beginners
A PPO tutorial by Ericyang https://github.com/ericyangyu/PPO-for-Beginners

## Usage
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To train from scratch:
```
python main.py
```

To test model:
```
python main.py --mode test --actor_model ppo_actor.pth
```

To train with existing actor/critic models:
```
python main.py --actor_model ppo_actor.pth --critic_model ppo_critic.pth
```


## Note
The original installation does not work for me. I used conda instead.
```
conda create -n ppoexample python=3.7
conda activate ppoexample
pip install -r requirements.txt
```

