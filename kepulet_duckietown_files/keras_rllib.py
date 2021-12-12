#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
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
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, MaxPooling3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf                                                               
import random
from tqdm import tqdm
from tensorflow.keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau



from rl.agents import DDPQAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


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

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)



checkpointer=ModelCheckpoint(filepath='weights.hdf5', save_best_only=True, verbose=1)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, min_lr=10e-5)



class Agent:
    def __init__(self):
        #This is the actual Neural net
        
        model = Sequential([ 
            Conv2D(32, kernel_size=(3,3), input_shape=(60,80,3), activation='relu'), 
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(32, kernel_size=(3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, kernel_size=(3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(3, activation='linear')
            #Dense(3)
        ])
        #pick your learning rate here
        model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.Accuracy()])#
        #model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001)) 
        #This is where you import your pretrained weights
        try:
            model.load_weights("DT.h5")
        except:
            pass
        self.model = model
        self.memory = []
        # Print the model summary if you want to see what it looks like
        # print(self.model.summary()) 
        self.xTrain = []
        self.yTrain = []
        self.loss = []
        self.location = 0


    def predict_(self, state):
        #sprint(np.shape(state))
        stateConv = state
        stateConv = np.reshape(state, (-1,60,80,3))
        qval = self.model.predict(stateConv)
        return qval

    def act(self, state): #TODO: TANULÁS SGÍTŐ CUCC+RETURN ACTION
        #print(np.shape(state))
        qval = self.predict_(state)
        action = np.argmax(qval)
        #you can either pick softmax or epislon greedy actions.
        #To pick Softmax, un comment the bottom 2 lines and delete everything below that 
        #prob = tf.nn.softmax(tf.math.divide((qval.flatten()), 1)) 
        #action = np.random.choice(range(3), p=np.array(prob))
        '''
        #Epsilon-Greedy actions->  TODO:itt mi a fasz na mind1 komment megnézem mi lesz
        z = np.random.random()
        epsilon = 0.004
        if self.location > 1000:
            epsilon = 0.05
        #epsilon = 0
        if z > epsilon:
            return np.argmax(qval.flatten())
        else:
            return np.random.choice(range(3))'''
        return action

    # This function stores experiences in the experience replay
    def remember(self, state, nextState, action, reward, done, location):
        self.location = location
        self.memory.append(np.array([state, nextState, action, reward, done]))

    #This is where the AI learns
    def learn(self):
        #Feel free to tweak this. This number is the number of experiences the AI learns from every round
        self.batchSize = 256 

        #If you don't trim the memory, your GPU might run out of memory during training. 
        #I found 35000 works well
        if len(self.memory) > 35000:
            self.memory = []
            print("trimming memory")
        if len(self.memory) < self.batchSize:
            print("too little info")
            return  
        batch = random.sample(self.memory, self.batchSize)

        self.learnBatch(batch)

    #The alpha value determines how future oriented the AI is.
    #bigger number (up to 1) -> more future oriented
    def learnBatch(self, batch, alpha=0.9):
        batch = np.array(batch)
        actions = batch[:, 2].reshape(self.batchSize).tolist()
        rewards = batch[:, 3].reshape(self.batchSize).tolist()

        stateToPredict = batch[:, 0].reshape(self.batchSize).tolist()
        nextStateToPredict = batch[:, 1].reshape(self.batchSize).tolist()

        statePrediction = self.model.predict(np.reshape(
            stateToPredict, (self.batchSize, 60, 80,3)))
        nextStatePrediction = self.model.predict(np.reshape(
            nextStateToPredict, (self.batchSize, 60, 80,3)))
        statePrediction = np.array(statePrediction)
        nextStatePrediction = np.array(nextStatePrediction)

        for i in range(self.batchSize):
            action = actions[i]
            reward = rewards[i]
            nextState = nextStatePrediction[i]
            qval = statePrediction[i, action]
            if reward < -5: 
                statePrediction[i, action] = reward
            else:
                #this is the q learning update rule
                statePrediction[i, action] += alpha * (reward + 0.95 * np.max(nextState) - qval)

        self.xTrain.append(np.reshape(
           stateToPredict, (self.batchSize,60, 80,3)))
        self.yTrain.append(statePrediction)
        #print(type(self.yTrain))
        history = self.model.fit(
            self.xTrain, self.yTrain, batch_size=5, epochs=1, verbose=0, callbacks=[checkpointer])
        loss = history.history.get("loss")[0]
        print("LOSS: ", loss)
        self.loss.append(loss)
        self.xTrain = []
        self.yTrain = []


model = Sequential([ 
        Conv2D(32, kernel_size=(3,3), input_shape=(60,80,3), activation='relu'), 
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(32, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='linear')])
#pick your learning rate here
model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.Accuracy()])#


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DDGPAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


dqn = build_agent(model, 3)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)


'''
plotX = []
n = 0

while n < 200:
    n+=1
    agent = Agent() #currently agent is configured with only 2 actions  
    #3500 refers to the number of episodes/iterations of the game to play
    
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
        #cv2.imwrite('Screenshot2.png',state[:,:,12])
        epReward = 0
        done = False
        episodeTime = time.time()
        stepCounter = 0
        acts = ""
        while not done:
            action = agent.act(state)
            #print('action',np.shape(action))
            acts += str(action) + ","
            nextState, reward, done, info = env.step(action)
            ########
            #This next section is storing more memory of later parts of the game since 
            #if you don't do this, most of the experience replay fills up with the 
            #starting parts of the game since its played more often. A more elegant 
            #approach to this is "Prioritized experience replay" but this is an effective
            #alternative too
            if stepCounter> 700:
                for _ in range(5):
                    agent.remember(state, nextState, action, reward, done, stepCounter)
            elif stepCounter> 40:
                agent.remember(state, nextState, action, reward, done, stepCounter)              
            if done == True: #game ended
                for _ in range(10):
                    agent.remember(state, nextState, action, reward, done, stepCounter)
                print("breaking")
                break
            ########
            state = nextState
            stepCounter += 1
            epReward += reward
            env.render()    
        #print(acts)
        #post episode
        if stepCounter != 0:
            print("Avg Frame-Rate: ", 1/((time.time()-episodeTime)/stepCounter))
        plotX.append(epReward)
        print(epReward)
        agent.learn()


       
        if i % 20 == 0:
            agent.model.save_weights ("DT.h5")
            print( "Saved model to disk")
            
        
       	env.render()
         
plt.plot(range(len(plotX)),plotX) 
plt.show()
plt.plot(range(len(agent.loss)), agent.loss) 
plt.show() 


'''
def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing"""
    """
    
    
    wheel_distance = 0.102
    min_rad = 0.08
    if key_handler[key.UP]:
        action += np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action -= np.array([0.44, 0])
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
    if key_handler[key.RIGHT]:
        action -= np.array([0, 1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])
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
    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5
    print(action)
    obs, reward, done, info = env.step(action)
    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    #if key_handler[key.RETURN]:
    im = Image.fromarray(obs)
    im.save("screen.png")
    if done:
        print("done!")
        env.reset()
        env.render()
"""
    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
