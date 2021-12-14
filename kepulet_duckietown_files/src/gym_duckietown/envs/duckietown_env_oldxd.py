# coding=utf-8
import numpy as np
from gym import spaces
from mss import mss

from ..simulator import Simulator
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
        #acts = [[0.33,0],[0.3,1],[0.3,-1],[-0.3,1],[-0.3,-1], [-0.33,0],[0,0]]
        #acts = [[0.33,0],[0.3,1],[0.3,-1]]
        acts = [[0.4,0.04],[0.4,0.04],[0.3,0.3]]#TODO: módosítottam az actionoket dec 4. felette lévő komment az előző
        vel, angle = acts[action]
        #SZija balázs
        #Már megy a kacsaaaaa!
        #Mozog a simulatorban! na megvan a hiba most oldom megOkok!

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
        
        #state = self._imageBankHandler(images_)
        
        obs, reward, done, info = Simulator.step(self, vels)
        img = obs
        
        self.images_ = self.prep_image(img)
        obs = self.images_
        
       
        
                
        #cv2.imwrite('Screenshot.png',obs[:,:,14])
        #obs = self.prep_image(img)
        mine = {}
        mine["k"] = self.k
        mine["gain"] = self.gain
        mine["train"] = self.trim
        mine["radius"] = self.radius
        mine["omega_r"] = omega_r
        mine["omega_l"] = omega_l
        info["DuckietownEnv"] = mine
        #print(np.shape(obs))
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
        
        test = image_
        
        img_height=60#target height
        img_width=80#target width
        #cv2.imwrite('raw.png',image_)
        image_ = cv2.resize(image_, None,fx=(img_width/image_.shape[1]), fy=(img_height/image_.shape[0]), interpolation = cv2.INTER_CUBIC)#resize to target shape 
        image_ = image_[ 20:60,0:80]#crop image (get processable data)
        image_ = cv2.resize(image_, None,fx=(1), fy=(30/20), interpolation = cv2.INTER_CUBIC)
        #image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)#toRGB
        #cv2.imwrite('resized_raw.png',image_)
        img_red = image_[:,:,0]
        img_green = image_[:,:,1]
        img_blue = image_[:,:,2]
        
        lower_white = (0,0,50)#lower filter
        upper_white = (400,40 ,400 )#upper filter
        
        #lower_white = (0,0,150)#lower filter
        #upper_white = (170,20 ,300 )#upper filter
        hsv_ = cv2.cvtColor(image_, cv2.COLOR_RGB2HSV)#hsv
        mask_ = cv2.inRange(hsv_, lower_white, upper_white)#mask (one channel)
        #plt.imshow(mask_)
        #plt.show()
        red = cv2.bitwise_and(img_red, img_red, mask = mask_)
        #cv2.imwrite('red.png',red[:,:])
        #print("Red channel:")
        #plt.imshow(red)
        #plt.show()
        red = self.pos_glob_stand(red)



        lower_orange = (20,70,60)#lower filter
        upper_orange = (30,400 ,400 )#upper filter
        hsv = cv2.cvtColor(image_, cv2.COLOR_RGB2HSV)#hsv
        mask = cv2.inRange(hsv, lower_orange, upper_orange)#mask (one channel)
        #plt.imshow(mask)
        #plt.show()
        blue = cv2.bitwise_and(img_blue, img_blue, mask = mask)
        #cv2.imwrite('blue.png',blue[:,:])
        blue = self.pos_glob_stand(blue)
        #print("Blue cahnnel:")
   	
        


        lower_def = (0,0,0)#lower filter
        upper_def = (0,0 ,0 )#upper filter
        hsv__ = cv2.cvtColor(image_, cv2.COLOR_RGB2HSV)#hsv
        mask__ = cv2.inRange(hsv__, lower_def, upper_def)#mask (one channel)
        #plt.imshow(mask__)
        plt.show()
        green = cv2.bitwise_and(img_green, img_green, mask = mask__)
        #cv2.imwrite('green.png',green[:,:])
        #print("Green channel:")
        #plt.show()
        #plt.imshow(green)
        green = self.pos_glob_stand(green)

        final = cv2.merge((red, green, blue))
        #for i in range(3):
            #final[:,:,i] = final[:,:,i]/255.0 #normalize image pixels

        #cv2.imwrite('Screenshot3.png',final)
        #plt.imshow(final)
        return final#return preprocessed image
        

    def pos_glob_stand(self, img): #this function applies a positive global normalization to the image for better learning speed, and simulation performance
        img_ = img
        img = img.astype('float32')
        img /= 255.0
        #mean, std = img.mean(), img.std() #mean and std of the imput image channel
        #img = (img - mean) / std #standardize pixel values
        #img = np.clip(img, -1.0,1.0) #clip the pixel values to [-1,1]
        #img = (img + 1.0) /2 #shift from [-1,1] to [0,1] with mean = 0.5
        return img_
        
        
    ''' 
    def prep_images(self, images_, image_):
        if np.shape(images_)[2] < 15:
            x=np.empty((60,80,15))
            for i in range(5):
                for j in range(3):
                    x[:,:,i+j] = self.prep_image(image_)[:,:,j]
            return x
        else:
            #print(np.shape(images_))
            tmp = images_
            for i in range(15):
                images_[:,:,i] = tmp[:,:,i]
            for j in range(3):
                images_[:,:,-3+j] = self.prep_image(image_)[:,:,j]
            return images_
            '''
            
            
    def prep_images(self, images_, image_):
        if np.shape(images_)[2] < 6:
            x=np.empty((60,80,3))
            for i in range(2):
                for j in range(3):
                    x[:,:,3*i+j] = self.prep_image(image_)[:,:,j]
            return x
        else:
            #print(np.shape(images_))
            tmp = images_
            for i in range(3):
                images_[:,:,i] = tmp[:,:,i+3]
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