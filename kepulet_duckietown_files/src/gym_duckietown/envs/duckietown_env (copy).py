# coding=utf-8
import numpy as np
from gym import spaces
from mss import mss

#from ..simulator import Simulator
from ..simulator import Simulator #ezt kell megváltoztatni TODO: 2021.12.11
from .. import logger


import cv2 as cv2
from PIL import Image, ImageEnhance, ImageOps
import keyboard
import time
import tqdm as tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf                                                               
import random
from tqdm import tqdm
from tensorflow.keras.models import model_from_json


class DuckietownEnv(Simulator):
    """
    Wrapper to control the simulator using velocity and steering angle
    instead of differential drive motor velocities
    """

    def __init__(self, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, **kwargs):
        Simulator.__init__(self, **kwargs)
        logger.info("using DuckietownEnv")
        
        self.mon = {'top': 243, 'left': 0, 'width': 640, 'height': 480}
        self.imageBank = []
        self.imageBankLength = 4 
        self.sct = mss()
        self.ones = np.ones((60,80,15))
        self.zeros = np.zeros((60,80,15))  
        self.zeros1 =np.zeros((60,80,15))   
        self.zeros2 = np.zeros((60,80,15))   
        self.zeros3 = np.zeros((60,80,15))   
        self.zeros1[:,:,0] = 1
        self.zeros2[:,:,1] = 1
        self.zeros3[:,:,2] = 1
        self.images_ = np.empty((60,80,3))
        self.first = True
        
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

    def step(self, action):
        acts = [[0.33,0],[0.3,1],[0.3,-1]]
        vel, angle = acts[action]

        # Distance between the wheels
        baseline = self.unwrapped.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        
        
        #get the obsrevation from an image
        obs, reward, done, info = Simulator.step(self, vels)
        img = obs
        #make preprocessing
        self.images_ = self.prep_image(img)
        #give it back to the observation and give to the outer code
        obs = self.images_

        mine = {}
        mine["k"] = self.k
        mine["gain"] = self.gain
        mine["train"] = self.trim
        mine["radius"] = self.radius
        mine["omega_r"] = omega_r
        mine["omega_l"] = omega_l
        info["DuckietownEnv"] = mine

        return obs, reward, done, info
        
    def _done(self,img):
        img = np.array(img)
        img  = img[30:60, 80:203]


        val = np.sum(img)
        #Sum of the reset pixels when the game ends in the night mode
        expectedVal = 331.9352517985612 
        #Sum of the reset pixels when the game ends in the day mode
        expectedVal2 = 243.53

        # This method checks if the game is done by reading the pixel values
        # of the area of the screen at the reset button. Then it compares it to
        # a pre determined sum. You might need to fine tune these values since each
        # person's viewport will be different. use the following print statements to 
        # help you find the appropirate values for your use case 

        # print("val: ", val)
        # print("Difference1: ", np.absolute(val-expectedVal2))
        # print("Difference2: ", np.absolute(val-expectedVal))
        if np.absolute(val-expectedVal) > 15 and np.absolute(val-expectedVal2) > 15: #seems to work well
            return False
        return True
        
        
       
    
    
    def prep_image(self,image_):#preprocess one image
        img_height=60#target height
        img_width=80#target width
        image_ = cv2.resize(image_, None,fx=(img_width/image_.shape[1]), fy=(img_height/image_.shape[0]), interpolation = cv2.INTER_CUBIC)#resize to target shape 
        image_ = image_[ 20:60,0:80]#crop image (get processable data)
        image_ = cv2.resize(image_, None,fx=(1), fy=(30/20), interpolation = cv2.INTER_CUBIC)#resize the cropped image to target size
        #image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)#toRGB
        cv2.imwrite('resized_raw.png',image_)#TODO: delete this line on release
        img_red = image_[:,:,0]#separate red channel
        img_green = image_[:,:,1]#separate green channel
        img_blue = image_[:,:,2]#separate blue channel
        
        lower_white = (0,0,50)#lower filter
        upper_white = (400,40 ,400 )#upper filter
        hsv_ = cv2.cvtColor(image_, cv2.COLOR_RGB2HSV)#make hsv picture
        mask_ = cv2.inRange(hsv_, lower_white, upper_white)#get red mask
        red = cv2.bitwise_and(img_red, img_red, mask = mask_)#mask red cahnnel

        lower_orange = (20,70,60)#lower filter
        upper_orange = (30,400 ,400 )#upper filter
        hsv = cv2.cvtColor(image_, cv2.COLOR_RGB2HSV)#make hsv picture
        mask = cv2.inRange(hsv, lower_orange, upper_orange)#get blue mask	
        blue = cv2.bitwise_and(img_blue, img_blue, mask = mask)#mask blue channel

        lower_def = (0,0,0)#lower filter
        upper_def = (0,0 ,0 )#upper filter
        hsv__ = cv2.cvtColor(image_, cv2.COLOR_RGB2HSV)#mask hsv picture
        mask__ = cv2.inRange(hsv__, lower_def, upper_def)#get green mask
        green = cv2.bitwise_and(img_green, img_green, mask = mask__)#mask green channel
        final = cv2.merge((red, green, blue))#merge the cahnnels to rgb image
        
        cv2.imwrite('processed.png',final)#TODO: delete this line on release
        
        final = final.astype('float32')#cast pixel values to float
        final /= 255.0#normalize pixel values between 0 and 1
       
        return final#return preprocessed image
        

        
        
        
    def prep_images(self, images_, image_):
    #make time series data
        if np.shape(images_)[2] < 15:#if first image
            x=np.empty((60,80,15))
            for i in range(5):
                for j in range(3):
                    x[:,:,i+j] = self.prep_image(image_)[:,:,j]
            return x
        else:#if not first image
            tmp = images_
            for i in range(15):
                images_[:,:,i] = tmp[:,:,i]
            for j in range(3):
                images_[:,:,-3+j] = self.prep_image(image_)[:,:,j]
            return images_

    
        

class DuckietownLF(DuckietownEnv):
    """
    Environment for the Duckietown lane following task with
    and without obstacles (LF and LFV tasks)
    """

    def __init__(self, **kwargs):
        DuckietownEnv.__init__(self, **kwargs)

    def step(self, action):
        obs, reward, done, info = DuckietownEnv.step(self, action)
        return obs, reward, done, info


class DuckietownNav(DuckietownEnv):
    """
    Environment for the Duckietown navigation task (NAV)
    """

    def __init__(self, **kwargs):
        self.goal_tile = None
        DuckietownEnv.__init__(self, **kwargs)

    def reset(self, segment=False):
        DuckietownNav.reset(self)

        # Find the tile the agent starts on
        start_tile_pos = self.get_grid_coords(self.cur_pos)
        start_tile = self._get_tile(*start_tile_pos)

        # Select a random goal tile to navigate to
        assert len(self.drivable_tiles) > 1
        while True:
            tile_idx = self.np_random.randint(0, len(self.drivable_tiles))
            self.goal_tile = self.drivable_tiles[tile_idx]
            if self.goal_tile is not start_tile:
                break

    def step(self, action):
        obs, reward, done, info, state = DuckietownNav.step(self, action)

        info["goal_tile"] = self.goal_tile

        # TODO: add term to reward based on distance to goal?

        cur_tile_coords = self.get_grid_coords(self.cur_pos)
        cur_tile = self._get_tile(*cur_tile_coords)

        if cur_tile is self.goal_tile:
            done = True
            reward = 1000

        return obs, reward, done, info
