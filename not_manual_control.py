#!/usr/bin/env python
# not_manual

from PIL import Image
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
args = parser.parse_args()
last_tile = [0,6]
tiles = [[0, 7],  #pathfinder.py outputs
    [1, 7],
    [2, 7],
    [2, 8],
    [3, 8],
    [4, 8],
    [4, 7],
    [5, 7],
    [6, 7],
    [6, 6],
    [6, 5],
    [7, 5],
    [7, 4],
    [8, 4],
    [9, 4],
    [9, 3],
    [9, 2],
    [8, 2],
    [8, 1],
    [7, 1],
    [7, 0],
    [6, 0],
    [5, 0],
    [5, 1],
    [4, 1],
    [3, 1],
    [2, 1],
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [0, 4],
    [0, 5],
    [0, 6],
    [1, 6],
    [1, 7],
    [2, 7],
    [2, 8],
    [3, 8],
    [4, 8],
    [4, 7],
    [5, 7],
    [6, 7],
    [6, 6],
    [6, 5],
    [7, 5],
    [7, 4],
    [8, 4],
    [9, 4],
    [9, 3],
    [9, 2],
    [8, 2],
    [8, 1],
    [7, 1],
    [7, 0],
    [6, 0],
    [5, 0],
    [5, 1],
    [4, 1],
    [3, 1],
    [2, 1],
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [0, 4],
    [0, 5],
    [0, 6],
    [1, 6],
    [1, 7],
    [2, 7],
    [2, 8],
    [3, 8],
    [4, 8],
    [4, 7],
    [5, 7],
    [6, 7],
    [6, 6],
    [6, 5],
    [7, 5],
    [7, 4],
    [8, 4]]


if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

def update(dt):
    wheel_distance = 0.102
    min_rad = 0.08
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    with open('/home/platyprof/gym-duckietown/last_tile.txt', 'r') as input_file: #előző tile és előző irány kiolvasása (tartalmuk last_tile.txt, úgy kell megadni, mintha az előző tile-ról indultak volna!)
        row = input_file.readline()
        last_tile = [int(row[row.find('[')+1 : row.find(',')]), int(row[row.find(',')+1 : row.find(']')])]
        row = input_file.readline()
        last_direction = [int(row[row.find('[')+1 : row.find(',')]), int(row[row.find(',')+1 : row.find(']')])]
   
    pos = [env.cur_pos[0]//0.585, env.cur_pos[2]//0.585] #jelenlegi tile
    direction = [tiles[1][0] - tiles[0][0], tiles[1][1] - tiles[0][1]] #jelenlegi irány
    print(pos, tiles[0], last_tile, last_direction)
    if (pos != tiles[0]): #ha új tile jön, frissítés
        last_tile = tiles.pop(0)
        with open('/home/platyprof/gym-duckietown/last_tile.txt', 'w+') as output_file: #változás kiírása
            output_file.write(str(last_tile)+'\n')
            output_file.write(str(direction))

    
    
    print(env.cur_angle)
    
    if (direction == [0,-1]): #irányfüggően a sebességek megadása, ez még nincs kész! vigyázni kell a nagy ívre kis ívre és még nem kanyarodik szépen :'(
        if (env.cur_angle < np.pi/2-0.02): 
            action = np.array([0.01, 0.2])
        elif (env.cur_angle > np.pi/2+0.02):
            action = np.array([0.01, -0.2])
        else:
    	    action = np.array([0.01,0])
    elif (direction == [0,1]):
        if (env.cur_angle < -0.02):
        	action = np.array([0.01, 0.2])
        elif (env.cur_angle > 0.02):
            action = np.array([0.01, -0.2])
        else:
    	    action = np.array([0.1,0])
    elif (direction == [1,0]):
        if (env.cur_angle < np.pi/4-0.02):
            action = np.array([0.01, 2])
        elif (env.cur_angle > np.pi/4+0.02):
            action = np.array([0.01, -2])
        else:
    	    action = np.array([0.1,0])
    else:
        if (env.cur_angle < 3*np.pi/4-0.02):
            action = np.array([0.01, 0.2])
        elif (env.cur_angle > 3*np.pi/4+0.02):
            action = np.array([0.01, -0.2])
        else:
    	    action = np.array([0.1,0])
    v1 = action[0]
    v2 = action[1]
    # Limit radius of curvature
    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
        # adjust velocities evenly such that condition is fulfilled
        delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
        v1 += delta_v
        v2 -= delta_v

    action[0] = v1
    action[1] = v2


    obs, reward, done, info = env.step(action)
    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))

    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()
    


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
