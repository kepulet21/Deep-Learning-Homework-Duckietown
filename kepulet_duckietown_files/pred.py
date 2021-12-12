#!/usr/bin/env python

# Imports
from PIL import Image
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

import cv2 as cv2
from mss import mss
from PIL import Image, ImageEnhance, ImageOps
import keyboard
import time
import tqdm as tqdm
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, MaxPooling3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf                                                               
import random
from tqdm import tqdm
from tensorflow.keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau


#from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="Duckietown-udem1-v0")
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

def session(session_map):
    if args.env_name and args.env_name.find("Duckietown") != -1:
        env = DuckietownEnv(
            seed=args.seed,
            map_name=session_map,
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

    return env

env = session("kmap1")
env.reset()
env.render()
@env.unwrapped.window.event

def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0

    elif symbol == key.RETURN:
         print('saving screenshot')
         img = env.render('rgb_array')
         save_img('screenshot.png', img)
         
    elif symbol == key.ESCAPE:
         agent.model.save_weights ("DT.h5")
         print( "Saved model to disk, and exit")
         env.close()
         sys.exit(0)

key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

#checkpointer=ModelCheckpoint(filepath='weights.hdf5', save_best_only=True, verbose=1)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, min_lr=10e-5)

class Agent:
    def __init__(self):
        model = Sequential([ 
            Conv2D(32, kernel_size=(3,3), input_shape=(480,640,15,), activation='relu'), 
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(32, kernel_size=(3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, kernel_size=(3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(3, activation="softmax")
        ])
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=[tf.keras.metrics.Accuracy()])
        try:
            model.load_weights("DT.h5")
        except:
            pass
        self.model = model
        self.memory = []
        self.xTrain = []
        self.yTrain = []
        self.loss = []
        self.location = 0


    def predict_(self, state):
        stateConv = state
        stateConv = np.reshape(state, (-1,480,640,15))
        qval = self.model.predict(stateConv)
        return qval

    def act(self, state):
        qval = self.predict_(state)
        z = np.random.random()
        epsilon = 0.004
        if self.location > 1000:
            epsilon = 0.05
        epsilon = 0
        if z > epsilon:
            return np.argmax(qval.flatten())
        else:
            return np.random.choice(range(3))
        return action

n = 0
while n < 200:
    n+=1
    agent = Agent()
    
    env.close()
    # Changing maps
    map_base = "kmap"
    index = random.randint(1,5)
    env=session(map_base + str(index))
    env.reset()
    env.render()
    
    for i in tqdm(range(41)):
        env.reset()
        action = np.random.randint(0,3)
        state, reward, done, info = env.step(action)
        epReward = 0
        done = False
        episodeTime = time.time()
        stepCounter = 0
        acts = ""
        while not done:
            action = agent.act(state)
            acts += str(action) + ","
            nextState, reward, done, info = env.step(action)            
            if done == True: #game ended
                print("breaking")
                break
            state = nextState
            stepCounter += 1
            epReward += reward
            env.render()    
        if stepCounter != 0:
            print("Avg Frame-Rate: ", 1/((time.time()-episodeTime)/stepCounter))
        print(epReward) 
       	env.render()
         

def update(dt):
    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
