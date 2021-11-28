# Deep Learning Homework - Duckietown
This is the official repo for our group: "K épület" in the course: "Deep Learning a gyakorlatban Python és LUA alapon" - (BMEVITMAV45). The members of our team:
- Endrész Balázs (F10RLU)
- Monori János Bence (PVUZ1Z)
- Wenesz Dominik (NBMU7U)

# First milestone:
As in this project we are utilizing a simulation environment, the following tasks have been done for the first milestone:
- *Initializing the docker environment*. This functions as the base of the headless running of the Duckietown simulation. The task was done by pulling the pre-built docker image after installing nvidia-docker.
- *Setting up the Duckietown-gym*. 
- *Creating a unique map*. The output.yaml file contains the data for our uniquely generated map. This was achieved by the map utilities (https://github.com/duckietown/map-utils/blob/master/README.md) provided by Duckietown.
- *Finding the shortest path between two positions* The previously mentioned link contains a pathfinder.py as well in which we implemented an A* searching algorithm to find the shortest path between two different points of the map. The path is later used in a lane following agent.
- *Writing a code for path-following* We have changed the code in *manual_control.py* as it follows a straight line the Duckiebot will maintain a certain speed and when it approaches a curve, the speed will decrease and a rotate in a specific angle.
- *Recording a run in the simulation* From the output of the path finder a run was recorded. 

These tasks mentioned above have been done and have been submitted by October 31st of 2021.
