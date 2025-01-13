import os
import sys
sys.path.append("../../")
sys.path.append("../../RATE")
from RATE_GTrXL import mem_transformer_v2_GTrXL
import torch
import numpy as np
from tqdm import tqdm
import faulthandler
faulthandler.enable()
with open("crash.log", "w") as f:
    faulthandler.enable(file=f)


from VizDoom.VizDoom_src.utils import env_vizdoom2

import matplotlib.pyplot as plt
import random
import seaborn as sns

import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# display(device)

import torch.nn as nn

sys.path.append("../3dcdrl/")

import torch
import numpy as np
# os.chdir("VizDoom/VizDoom_notebooks")
pwd=os.getcwd()
print(pwd)

from models2 import CNNPolicy
from doom_environment2 import DoomEnvironment

import pickle
from tqdm import tqdm

import env_vizdoom2

import matplotlib.pyplot as plt
from itertools import count
import time

print("Entering env_args")
env_args = {
    'simulator':'doom',
    'scenario':'custom_scenario_no_pil{:003}.cfg', #custom_scenario_no_pil{:003}.cfg
    # 'scenario':'custom_scenario_no_pil{:003}.cfg', #custom_scenario_no_pil{:003}.cfg
    'test_scenario':'',
    'screen_size':'320X180',
    'screen_height':64,
    'screen_width':112,
    'num_environments':16,# 16
    'limit_actions':True,
    'scenario_dir':'../../VizDoom/VizDoom_src/env/',
    'test_scenario_dir':'',
    'show_window':False,
    'resize':True,
    'multimaze':True,
    'num_mazes_train':16,
    'num_mazes_test':1, # 64
    'disable_head_bob':False,
    'use_shaping':False,
    'fixed_scenario':False,
    'use_pipes':False,
    'num_actions':0,
    'hidden_size':128,
    'reload_model':'',
    'model_checkpoint':'../../VizDoom/VizDoom_notebooks/two_col_p1_checkpoint_0198658048.pth.tar',   # two_col_p0_checkpoint_0049154048.pth.tar',  #two_col_p0_checkpoint_0198658048.pth.tar',
    'conv1_size':16,
    'conv2_size':32,
    'conv3_size':16,
    'learning_rate':0.0007,
    'momentum':0.0,
    'gamma':0.99,
    'frame_skip':4,
    'train_freq':4,
    'train_report_freq':100,
    'max_iters':5000000,
    'eval_freq':1000,
    'eval_games':50,
    'model_save_rate':1000,
    'eps':1e-05,
    'alpha':0.99,
    'use_gae':False,
    'tau':0.95,
    'entropy_coef':0.001,
    'value_loss_coef':0.5,
    'max_grad_norm':0.5,
    'num_steps':128,
    'num_stack':1,
    'num_frames':200000000,
    'use_em_loss':False,
    'skip_eval':False,
    'stoc_evals':False,
    'model_dir':'',
    'out_dir':'./',
    'log_interval':100,
    'job_id':12345,
    'test_name':'test_000',
    'use_visdom':False,
    'visdom_port':8097,
    'visdom_ip':'http://10.0.0.1'
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("inializing the DoomEnvironment")
env = DoomEnvironment(env_args, idx=0, is_train=True, get_extra_info=False)
print("Number of env actions:", env.num_actions)
obs_shape = (3, env_args['screen_height'], env_args['screen_width'])
print(obs_shape)

scene = 0
scenario = env_args['scenario_dir'] + env_args['scenario'].format(scene) # 0 % 63
config = scenario

env = env_vizdoom2.DoomEnvironmentDisappear(
    scenario=config,
    show_window=False,
    use_info=True,
    use_shaping=False, #if False bonus reward if #shaping reward is always: +1,-1 in two_towers
    frame_skip=2,
    no_backward_movement=True)

policy = CNNPolicy((3, 64, 112), env_args).to(device)
checkpoint = torch.load(env_args['model_checkpoint'], map_location=lambda storage, loc: storage)
policy.load_state_dict(checkpoint['model'])
policy.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = DoomEnvironment(env_args, idx=0, is_train=True, get_extra_info=False)
obs_shape = (3, env_args['screen_height'], env_args['screen_width'])

scene = 0
scenario = env_args['scenario_dir'] + env_args['scenario'].format(scene) # 0 % 63
config = scenario

env = env_vizdoom2.DoomEnvironmentDisappear(
    scenario=config,
    show_window=False,
    use_info=True,
    use_shaping=False, #if False bonus reward if #shaping reward is always: +1,-1 in two_towers
    frame_skip=2,
    no_backward_movement=True)

policy = CNNPolicy((3, 64, 112), env_args).to(device)
checkpoint = torch.load(env_args['model_checkpoint'], map_location=lambda storage, loc: storage)
policy.load_state_dict(checkpoint['model'])
policy.eval()

NUMBER_OF_TRAIN_DATA = 5000 # 5000
EPISODE_TIMEOUT = 90 # 90

returns_red, returns_green = [], []

for i in tqdm(range(NUMBER_OF_TRAIN_DATA)):
    obsList, actList, rewList, doneList, isRedList = [], [], [], [], []
    times = []
    obs = env.reset()
    state = torch.zeros(1, env_args['hidden_size']).to(device)
    mask = torch.ones(1,1).to(device)
    done = False

    for t in count():
        times.append(t)
        obsList.append(obs['image'])
        result = policy(torch.from_numpy(obs['image']).unsqueeze(0).to(device), state, mask)
        action, state = result['actions'], result['states']


        obs, reward, done, info = env.step(action.item())


        is_red = info['is_red']
        rewList.append(reward)
        actList.append(action.item())
        doneList.append(int(done))
        isRedList.append(is_red)

        if done or t == EPISODE_TIMEOUT-1:

            if is_red == 1.0:
                returns_red.append(np.sum(rewList))
            else:
                returns_green.append(np.sum(rewList))

            break

    DATA = {'obs': np.array(obsList), # (1152, 3, 64, 112)
            'action': np.array(actList),
            'reward': np.array(rewList),
            'done': np.array(doneList),
            'is_red': np.array(isRedList)}

    file_path = f'../VizDoom_data/iterative_data/train_data_{i}.npz'
    np.savez(file_path, **DATA)


env.close()
episode = np.load(f'../VizDoom_data/iterative_data/train_data_{4999}.npz')
episode = {key: episode[key] for key in episode.keys()}
episode.keys()
from moviepy.editor import ImageSequenceClip, VideoFileClip
import numpy as np
import cv2

desired_resolution = (945, 540)
original_aspect_ratio = 112 / 64
width = int(desired_resolution[0] * original_aspect_ratio)
height = desired_resolution[1]

# Assuming 'states1' is a list of numpy arrays representing images
observations = [np.squeeze(o).transpose(1, 2, 0) for o in episode['obs']]

# Create ImageSequenceClip
clip = ImageSequenceClip(observations, fps=24)
clip = clip.resize(width=width, height=height)

# Add text to each frame
# text = 'text'
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
thickness = 2

# Create a list to store the modified frames
modified_frames = []

for idx, frame in enumerate(clip.iter_frames()):
    modified_frame = cv2.putText(frame, str(times[idx]), (20, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)

    modified_frames.append(modified_frame)

# Create a new ImageSequenceClip with the modified frames
modified_clip = ImageSequenceClip(modified_frames, fps=clip.fps)

# Display the modified clip
modified_clip.ipython_display(maxduration=120)
# scene = 0
# scenario = env_args['scenario_dir'] + env_args['scenario'].format(scene)
# config_env = scenario
# #np.random.seed(seed)

# env = env_vizdoom2.DoomEnvironmentDisappear(
#     scenario=config_env,
#     show_window=False,
#     use_info=True,
#     use_shaping=False, # if False rew only +1 if True rew +1 or -1
#     frame_skip=1,
#     no_backward_movement=True,
#     seed=0)

# obs1 = env.reset()
# state, reward, done, info = env.step(0)
# plt.imshow(state['image'].transpose(1,2,0))