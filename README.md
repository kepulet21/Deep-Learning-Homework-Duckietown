# Deep Learning Homework - Duckietown
This is the official repo for our group: "K épület" in the course: "Deep Learning a gyakorlatban Python és LUA alapon" - (BMEVITMAV45). The members of our team:
- Endrész Balázs (F10RLU)
- Monori János Bence (PVUZ1Z)
- Wenesz Dominik (NBMU7U)

# First milestone:
As in this project we are utilizing a simulation environment, the following tasks have been done for the first milestone:
- *Initializing the docker environment*: This functions as the base of the headless running of the Duckietown simulation. The task was done by pulling the pre-built docker image after installing nvidia-docker.
- *Setting up the Duckietown-gym*. 
- *Creating a unique map*: The output.yaml file contains the data for our uniquely generated map. This was achieved by the map utilities (https://github.com/duckietown/map-utils/blob/master/README.md) provided by Duckietown.
- *Finding the shortest path between two positions* The previously mentioned link contains a pathfinder.py as well in which we implemented an A* searching algorithm to find the shortest path between two different points of the map. The path is later used in a lane following agent.
- *Writing a code for path-following* We have changed the code in *manual_control.py* as it follows a straight line the Duckiebot will maintain a certain speed and when it approaches a curve, the speed will decrease and a rotate in a specific angle.
- *Recording a run in the simulation* From the output of the path finder a run was recorded. 

These tasks mentioned above have been done and have been submitted by October 31st of 2021.

# Second milestone:
For this milestone we assembled the following method: *Taking screenshots from the environment* > *Preprocessing the image* & *Calculating a reward function* > *CNN model* > *Predicting the best action*. The following task have been done for the second milestone:
- *Preprocessing*: During the simulation a screenshot is taken every step. After that an image processing algorithm filters the lines on the road: the white continous line is filtered to the Blue chanel while the yellow dashed line in the middle is filtered to the Red chanel.
- *CNN model*: We have implemented a 2D convolutional neural network as our main task was image processing. 
- *Predicting the best action*: A Q-learning algorithm is attached after the CNN model, fundamentally speaking it predicts the best action (going straight, turning left orturning right) frame by frame. As the run ends (by hitting an object, leaving the map or reaching a step count of 1500) a reward function is calculated corresponding to the run

For both learning and predicting the following file shall be used: *q.py*.

 These tasks mentioned above have been done and have been submitted by December 2nd of 2021.
