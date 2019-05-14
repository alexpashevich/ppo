# PPO adapted for MiME environments

The repo will use MiME and BC repos, make sure they are accessible. The only script to be executed is `ppo.scripts.train`. Please use `RLAgent` from BC repo to evaluate and render trained agents. Right now, 3 modes are supported: full state input + scripts, vision input + scripts and vision input + skills (HRLBC).

## UR5-Bowl
### Train RL on UR5-Bowl with full state and scripts
```bash
python3 -m ppo.scripts.train --env-name UR5-Paris-Aug-BowlEnv-v0 \
--max-length=600 --num-processes=8 --num-skills=4 --num-master-steps-per-update=12 \
--mime-action-space=tool_lin \
--logdir=$LOGDIR
```
*Useful flags*: `--eval-offline` if you do not want the evaluation to be executed (it slows down the training). `--render` to render what RL is doing. `--dask-batch-size` is the dask batch size.

### Train RL on UR5-Bowl with vision and scripts
```bash
python3 -m ppo.scripts.train --env-name UR5-Paris-AugEGL-BowlCamEnv-v0 \
--max-length=600 --num-processes=8 --num-skills=4 --num-master-steps-per-update=12 \
--checkpoint-path=$CHECKPOINT_PATH --logdir=$LOGDIR
```

### Train RL on UR5-Bowl with HRLBC
```bash
python3 -m ppo.scripts.train --env-name UR5-Paris-AugEGL-BowlCamEnv-v0 \
--max-length=600 --num-processes=8 --num-skills=4 --num-master-steps-per-update=12 \
--hrlbc-setup --timescale=50 \
--checkpoint-path=$CHECKPOINT_PATH --logdir=$LOGDIR
```
*Useful flags*: `--augmentation=$AUGMENTATION` to specify the frames augmentation to be used (from BC repo).

## UR5-SimplePour

### Train RL on UR5-SimplePour with full state and scripts
```bash
python3 -m ppo.scripts.train --env-name UR5-SimplePourNoDropsEnv-v0 \
--max-length=500 --num-processes=8 --num-skills=5 --num-master-steps-per-update=12 \
--mime-action-space=tool_lin_ori \
--logdir=$LOGDIR
```

### Train RL on UR5-SimplePour with full state and scripts and memory
```bash
python3 -m ppo.scripts.train --env-name UR5-SimplePourNoDropsEnv-v0 \
--max-length=500 --num-processes=8 --num-skills=5 --num-master-steps-per-update=12 \
--mime-action-space=tool_lin_ori \
--action-memory=5 \
--logdir=$LOGDIR
```

## UR5-Salad

### Train RL on UR5-Salad with full state and scripts
```bash
python3 -m ppo.scripts.train --env-name UR5-Paris-Aug-SaladEnv-v0 \
--max-length=2000 --num-processes=16 --num-skills=7 --num-mini-batch=8 --num-master-steps-per-update=30 \
--mime-action-space=tool_lin_ori \
--logdir=$LOGDIR
```

### Train RL on UR5-Salad with vision and scripts
```bash
python3 -m ppo.scripts.train --env-name UR5-Paris-AugEGL-SaladCamEnv-v0 \
--max-length=2000 --num-processes=16 --num-skills=7 --num-mini-batch=8 --num-master-steps-per-update=30 \
--checkpoint-path=$CHECKPOINT_PATH --logdir=$LOGDIR
```

### Train RL on UR5-Salad with HRLBC
```bash
python3 -m ppo.scripts.train --env-name UR5-Paris-AugEGL-SaladCamEnv-v0 \
--max-length=2000 --num-processes=16 --num-skills=7 --num-mini-batch=8 --num-master-steps-per-update=20 \
--hrlbc-setup --timescale=80 \
--checkpoint-path=$CHECKPOINT_PATH --logdir=$LOGDIR
```
*Useful flags*: you can use `--check-skills-silency` to make RL to use `skill_should_be_silent` function of a scene (will check whether the skill should be executed).

### Render HRLBC on UR5-Salad
```bash
python3 -m ppo.scripts.train  --env-name=UR5-Paris-AugEGL-SaladCamEnv-v0 \
--max-length=2000 --num-processes=1 --num-skills=7 --num-mini-batch=8 --num-master-steps-per-update=20 \
--hrlbc-setup --timescale=80 \
--render \
--checkpoint-path=$CHECKPOINT_PATH --logdir=$LOGDIR 
```
