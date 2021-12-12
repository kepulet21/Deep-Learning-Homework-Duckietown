import itertools
import os
from collections import namedtuple
from ctypes import POINTER
from dataclasses import dataclass

import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from typing import Any, cast, Dict, List, NewType, Optional, Sequence, Tuple, Union

import geometry
import geometry as g
import gym
import math
import numpy as np
import pyglet
import yaml
from geometry import SE2value
from gym import spaces
from gym.utils import seeding
from numpy.random.mtrand import RandomState
from pyglet import gl, image, window

from duckietown_world import (
    get_DB18_nominal,
    get_DB18_uncalibrated,
    get_texture_file,
    MapFormat1,
    MapFormat1Constants,
    MapFormat1Constants as MF1C,
    MapFormat1Object,
    SE2Transform,
)
from duckietown_world.gltf.export import get_duckiebot_color_from_colorname
from duckietown_world.resources import get_resource_path
from duckietown_world.world_duckietown.map_loading import get_transform

# Rendering window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Camera image size
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480

# Blue sky horizon color
BLUE_SKY = np.array([0.45, 0.82, 1])

# Color meant to approximate interior walls
WALL_COLOR = np.array([0.64, 0.71, 0.28])

# np.array([0.15, 0.15, 0.15])
GREEN = (0.0, 1.0, 0.0)
# Ground/floor color


# Angle at which the camera is pitched downwards
CAMERA_ANGLE = 19.15

# Camera field of view angle in the Y direction
# Note: robot uses Raspberri Pi camera module V1.3
# https://www.raspberrypi.org/documentation/hardware/camera/README.md
CAMERA_FOV_Y = 75

# Distance from camera to floor (10.8cm)
CAMERA_FLOOR_DIST = 0.108

# Forward distance between the camera (at the front)
# and the center of rotation (6.6cm)
CAMERA_FORWARD_DIST = 0.066

# Distance (diameter) between the center of the robot wheels (10.2cm)
WHEEL_DIST = 0.102

# Total robot width at wheel base, used for collision detection
# Note: the actual robot width is 13cm, but we add a litte bit of buffer
#       to faciliate sim-to-real transfer.
ROBOT_WIDTH = 0.13 + 0.02

# Total robot length
# Note: the center of rotation (between the wheels) is not at the
#       geometric center see CAMERA_FORWARD_DIST
ROBOT_LENGTH = 0.18

# Height of the robot, used for scaling
ROBOT_HEIGHT = 0.12

# Safety radius multiplier
SAFETY_RAD_MULT = 1.8

# Robot safety circle radius
AGENT_SAFETY_RAD = (max(ROBOT_LENGTH, ROBOT_WIDTH) / 2) * SAFETY_RAD_MULT

# Minimum distance spawn position needs to be from all objects
MIN_SPAWN_OBJ_DIST = 0.25

# Road tile dimensions (2ft x 2ft, 61cm wide)
# self.road_tile_size = 0.61

# Maximum forward robot speed in meters/second
DEFAULT_ROBOT_SPEED = 1.0 #TODO módosítottam a sebességet 1-re dec 4.
# approx 2 tiles/second

DEFAULT_FRAMERATE = 30

DEFAULT_MAX_STEPS = 1500

DEFAULT_MAP_NAME = "udem1"

DEFAULT_FRAME_SKIP = 1

DEFAULT_ACCEPT_START_ANGLE_DEG = 60

REWARD_INVALID_POSE = -1000 #TODO módosítottam -1000-ről -40-re dec 4.

MAX_SPAWN_ATTEMPTS = 5000


class OBS_Processing(gym.Env):

    def __init__(
        self):
        
        # We observe an RGB image with pixels in [0, 255]
        # Note: the pixels are in uint8 format because this is more compact
        # than float32 if sent over the network or stored in a dataset
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.camera_height, self.camera_width, 3), dtype=np.uint8
        )

        # Distortion params, if so, load the library, only if not bbox mode
        self.distortion = distortion and not draw_bbox
        self.camera_rand = False
        if not draw_bbox and distortion:
            if distortion:
                self.camera_rand = camera_rand

                self.camera_model = Distortion(camera_rand=self.camera_rand)

        # Initialize the state
        self.reset()

        self.last_action = np.array([0, 0])
        self.wheelVels = np.array([0, 0])
        self.images_ = np.empty((60,80,3))
        
        
    def render_obs(self, segment: bool = False) -> np.ndarray:
        """
        Render an observation from the point of view of the agent
        """

        observation = self._render_img(
            self.camera_width,
            self.camera_height,
            self.multi_fbo,
            self.final_fbo,
            self.img_array,
            top_down=False,
            segment=segment,
        )
        # self.undistort - for UndistortWrapper
        if self.distortion and not self.undistort:
            observation = self.camera_model.distort(observation)
        
        img = observation
        self.images_ = self.prep_image(img)
        #print(self.images_.shape)
        observation = self.images_        
        return observation
        
        
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
        

