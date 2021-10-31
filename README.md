# Deep Learning Homework - Duckietown
This is the official repo for our group: "K épület" in the course: "Deep Learning a gyakorlatban Python és LUA alapon" - (BMEVITMAV45). The members of our team:
- Endrész Balázs
- Monori János Bence
- Wenesz Dominik

# First milestone:
As in this project we are utilizing a simulation environment, the following tasks have been done for the first milestone:
- *Initializing the docker environment*. This functions as the base of the headless running of the Duckietown simulation. The task was done by pulling the pre-built docker image after installing nvidia-docker.
- *Setting up the Duckietown-gym*. 
- *Creating a unique map*. The output.yaml file contains the data for our uniquely generated map. This was achieved by the map utilities (https://github.com/duckietown/map-utils/blob/master/README.md) provided by Duckietown.
- *Creating a path-finder* The previously mentioned link contains a pathfinder.py as well in which we implemented an A* searching algorithm to find the shortest path between two different points of the map. The path is later 
