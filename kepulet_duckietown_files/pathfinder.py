#!/home/kepulet/anaconda3/bin/python
import sys
import numpy as np
import argparse
import random as rand


MAP  = []
ONES = []
WEIGHTED_GRAPH = {}
HEIGHT = 0
WIDTH  = 0


def main(file, rand_number, start):
    global HEIGHT
    global WIDTH
    data = np.load(file)
    # adj is a lower triangular matrix
    adj  = data['out']
    dims = data['dims']
    HEIGHT = dims[0]
    WIDTH = dims[1]
    to_graph(adj)
    
    
    sys.stdout.write("\n")
    maze = ascii_map()
    sys.stdout.write("\n")
    last = [4,1] #EEEEEEEEEEEEEEEEEEEEZ függ attól, hogy honnan indul az autó, ez az azzal ellentétes irányt jelöli! 
    #set_weighted(st, end)
    print(maze)
    for k in range(rand_number):
        #print('------------------------------')
        forward = True
        n = 0
        while (forward):
            n += 1
            if (n == 1):
                if(k == 0):
                    st = start
                else:
                    st = end
            not_good = True
            while (not_good):
                not_good = False
                end = [rand.randint(0, HEIGHT-1), rand.randint(0, WIDTH-1)]
                for m in maze:
                    if (m == end):
                        not_good = True
            a_star = AStar(dims[1], dims[0], [st[0], st[1]], [end[0], end[1]], False, maze)
            #print(dims[1], dims[0], [st[0], st[1]], [end[0], end[1]])
            final_path = a_star.main()
            forward = False
            
            if (len(final_path) > 4):
                if([final_path[-2].x, final_path[-2].y] != last):
                    if len(final_path) > 0:
                        #print("The way found!!!")
                        for i in range(len(final_path)-1, 0, -1):
                            if (i == len(final_path)-1 and k == 0):
                                print("[["+str(final_path[i].x) +  ", " + str(final_path[i].y)+"],")
                            else:
                                print("    ["+str(final_path[i].x) +  ", " + str(final_path[i].y)+"],")
                        print("    ["+str(final_path[0].x) +  ", " + str(final_path[0].y)+"],")
                        last = [final_path[0].x, final_path[0].y]
                else:
                    forward = True
            else:
                forward = True
        #else:   
            #print("There is no legal way...")
    # ============================================================
    # Change THIS FUNCTION to use different path finding algorithm.
    # Must return an array of node coordinates within MAP,
    # where st comes first and end comes last.
    # elements are COORDINATES
    #path = shortest_path()

    # example path
    #path = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)]
    #output_path(path)



# "unique" to coordinate: converts the unique node number to a coordinate tuple
def u2c(u):
    if (len(MAP) == 0):
        raise ValueError("MAP not yet initialized")
    x = u % WIDTH
    y = u // WIDTH
    return (x, y)

# coordinate to "unique": converts coordinate tuple (x, y) to its unique equivalent
def c2u(c):
    return (c[1] * WIDTH) + c[0]

# initialises the global variable MAP with a node graph
# based on the adjacency matrix
def to_graph(adj):
    global MAP
    MAP = [[None for x in range(WIDTH)] for y in range(HEIGHT)]

    # initialize the nodes
    for y in range(0, HEIGHT):
        for x in range(0, WIDTH):
            MAP[y][x] = Node(x, y)

    for j in range(0, HEIGHT):
        for i in range(0, WIDTH):
            MAP[j][i].set_neighbours()

    for b in range(0, HEIGHT * WIDTH):
        for a in range(0, b):
            if (adj[b][a]):
                if (a == b - WIDTH):
                    neigh_pos = "N"
                elif (a == b + WIDTH):
                    neigh_pos = "S"
                elif (a == b + 1):
                    neigh_pos = "E"
                elif (a == b - 1):
                    neigh_pos = "W"
                else:
                    raise ValueError("neighbour not in right place")

                coords = u2c(b)
                if not (MAP[coords[1]][coords[0]].connected[neigh_pos]):
                    MAP[coords[1]][coords[0]].connect(neigh_pos)


def set_weighted(st, end):
    global WEIGHTED_GRAPH
    # start and end must both be connected to the road network
    if (
            (st[0] < 0) or (st[0] >= WIDTH) or (st[1] < 0) or (st[1] >= HEIGHT)
            or (end[0] < 0) or (end[0] >= WIDTH) or (end[1] < 0) or (end[1] >= HEIGHT)
            or not ((MAP[st[1]][st[0]].deg() > 1) and (MAP[end[1]][end[0]].deg() > 1))
    ):
        print("Please select a start and end point within the road network.")
        sys.exit()

    # do this because adjacency lists require unique nodes
    # and coordinates don't work correctly in dictionaries
    st_1d  = c2u(st)
    end_1d = c2u(end)

    # list of nodes
    to_visit = [MAP[st[1]][st[0]]]

    visited = []

    # element format: <(y*WIDTH) + x>: [(x1, y1, d1), (x2, y2, d2), ...]
    #           curr 1d coords    neighbour coords, distance
    adj_list  = {
        st_1d: [],
        end_1d: []
    }

    curr = MAP[st[1]][st[0]]
    while(len(to_visit) > 0):
        # check neighbour in each direction to find next node
        for i in range(0, 4):
            direction = ["N", "E", "S", "W"][i]
            if (curr.connected[direction]):
                sub = curr.neighbours[direction]
                distance = 1
                last_dir = find_rel_dirs(direction)["f"]
                while(sub.deg() < 3 and not ((sub.x == end[0]) and (sub.y == end[1]))):
                    rel_dirs = find_rel_dirs(last_dir)
                    if (sub.connected[rel_dirs["f"]]):
                        sub = sub.neighbours[rel_dirs["f"]]
                        last_dir = rel_dirs["b"]
                        distance += 1
                    elif (sub.connected[rel_dirs["l"]]):
                        sub = sub.neighbours[rel_dirs["l"]]
                        last_dir = rel_dirs["r"]
                        distance += 1
                    elif (sub.connected[rel_dirs["r"]]):
                        sub = sub.neighbours[rel_dirs["r"]]
                        last_dir = rel_dirs["l"]
                        distance += 1

                if (sub.deg() > 2) or ((sub.x == end[0]) and (sub.y == end[1])):
                    if (sub.x, sub.y, distance) not in adj_list[c2u((curr.x, curr.y))]:
                        adj_list[c2u((curr.x, curr.y))].append((sub.x, sub.y, distance))
                    if c2u((sub.x, sub.y)) not in adj_list:
                        adj_list[c2u((sub.x, sub.y))] = []

                    if (sub not in visited):
                        to_visit.append(sub)

        to_visit.remove(curr)
        visited.append(curr)
        if (len(to_visit) > 0):
            curr = to_visit[0]

    WEIGHTED_GRAPH = adj_list


def shortest_path():
    pass

# takes array containing node order of path and
# outputs a .yaml file
def output_path(path):
    with open('path.yaml', 'w+') as f:
        for j in range(0, len(MAP)):
            line = ""
            # fill up arrays
            for i in range(0, len(MAP[0])):
                if ((i, j) == path[0]):
                    line += "S "
                elif((i, j) == path[len(path) - 1]):
                    line += "X "
                elif (i, j) in path:
                    line += "1 "
                else:
                    line += "0 "
            # [:-1] takes away the trailing space
            f.write(line[:-1] + "\n")
    print("Path written to path.yaml")


def ascii_map():
    way = []
    global MAP
    for j in range(0, len(MAP)):
        down_edges = []
        right_edges = []
        # fill up arrays
        for i in range(0, len(MAP[0])):
            if (MAP[j][i].connected["E"] == 1):
                right_edges.append(str(MAP[j][i].deg()) + "-")
            else:
                right_edges.append(str(MAP[j][i].deg()) + " ")

            if (MAP[j][i].connected["S"] == 1):
                down_edges.append("| ")
            else:
                down_edges.append("  ")
            if (str(MAP[j][i].deg()) == '0'):
                way.append([i,j])

        # print right edges
        for k in range(0, len(right_edges)):
            sys.stdout.write(right_edges[k])
        sys.stdout.write("\n")

        # print down edges
        for m in range(0, len(down_edges)):
            sys.stdout.write(down_edges[m])
        sys.stdout.write("\n")

    sys.stdout.write("\n")
    sys.stdout.flush()
    return way

# returns set of relative directions on a degree 1 node
# according to its only connection
def find_rel_dirs(leading_connection):
    if (leading_connection == "S"):
        return {"f": "N",  # relative forward "f" is north "N"
                "r": "E",  # relative right "r" is east "E"
                "b": "S",  # relative backwards "b" is south "S"
                "l": "W"}  # relative left "l" is west "W"
    elif (leading_connection == "W"):
        return {"f": "E",
                "r": "S",
                "b": "W",
                "l": "N"}
    elif (leading_connection == "N"):
        return {"f": "S",
                "r": "W",
                "b": "N",
                "l": "E"}
    elif (leading_connection == "E"):
        return {"f": "W",
                "r": "N",
                "b": "E",
                "l": "S"}
    else:
        raise ValueError("ERROR: unrecognized direction string")
# ============================= NODE CLASS ====================================
class Node():
    global MAP
    def __init__(self, x, y):
        # cartesian coordinates in the map
        self.x = x
        self.y = y

        # the neighbouring Nodes
        self.neighbours = {"N":None, "E":None, "S":None, "W":None}

        # represents if self is connected to neighbours
        self.connected = {"N":0, "E":0, "S":0, "W":0}

    def deg(self):
        return self.connected["N"] + self.connected["E"] + \
               self.connected["S"] + self.connected["W"]

    def connect(self, direction):
        if (self.neighbours[direction] is not None):
            global ONES
            # connect on self's side
            if (self.deg() == 0):
                ONES.append((self.x, self.y))
            elif (self.deg() == 1):
                ONES.remove((self.x, self.y))
            self.connected[direction] = 1

            # connect on the conectee's side
            if (self.neighbours[direction].deg()==0):
                ONES.append((self.neighbours[direction].x, self.neighbours[direction].y))
            elif (self.neighbours[direction].deg()==1):
                ONES.remove((self.neighbours[direction].x, self.neighbours[direction].y))

            if   direction == "N":
                self.neighbours[direction].connected["S"] = 1
            elif direction == "E":
                self.neighbours[direction].connected["W"] = 1
            elif direction == "S":
                self.neighbours[direction].connected["N"] = 1
            elif direction == "W":
                self.neighbours[direction].connected["E"] = 1
        else:
            raise ValueError('ERROR: tried to connect "{}" to non-existant node from ({}, {})'.format(direction, str(self.x), str(self.y)))


    def disconnect(self, direction):
        if (self.neighbours[direction] is not None):
            global ONES
            # disconnect on self's side
            if (self.deg() == 1):
                ONES.remove((self.x, self.y))
            elif (self.deg() == 2):
                ONES.append((self.x, self.y))
            self.connected[direction] = 0


            # disconnect on the disconectee's side
            if (self.neighbours[direction].deg() == 1):
                ONES.remove((self.neighbours[direction].x, self.neighbours[direction].y))
            elif (self.neighbours[direction].deg() == 2):
                ONES.append((self.neighbours[direction].x, self.neighbours[direction].y))

            if   direction == "N":
                self.neighbours[direction].connected["S"] = 0
            elif direction == "E":
                self.neighbours[direction].connected["W"] = 0
            elif direction == "S":
                self.neighbours[direction].connected["N"] = 0
            elif direction == "W":
                self.neighbours[direction].connected["E"] = 0
        else:
            raise ValueError("ERROR: tried to connect to non-existant node")



    # IMPORTANT: Y-axis comes before X-axis due to how 2D arrays are represented
    # ( MAP[y][x] );    ex. MAP[5][0] => cartesian coordinates (0, 5)
    def set_neighbours(self):
        # north neighbour
        # if node is in the top row (y == 0), self.neighbours["N"] remains None
        try:
            if (self.y > 0):
                self.neighbours["N"] = MAP[self.y - 1][self.x]
        except:
            raise ValueError("ERROR: undefined north neighbour")

        # east neighbour
        # if node is in the rightmost column (x == len(MAP[0])-1), self.neighbours["E"] remains None
        try:
            if (self.x < len(MAP[0])-1):
                self.neighbours["E"] = MAP[self.y][self.x + 1]
        except:
            raise ValueError("ERROR: undefined east neighbour")

        # south neighbour
        # if node is in the bottom row (y == len(MAP)), self.neighbours["S"] remains None
        try:
            if (self.y < len(MAP)-1):
                self.neighbours["S"] = MAP[self.y + 1][self.x]
        except:
            raise ValueError("ERROR: undefined south neighbour")

        # west neighbour
        # if node is in the leftmost column (x == 0), self.neighbours["W"] remains None
        try:
            if (self.x > 0):
                self.neighbours["W"] = MAP[self.y][self.x - 1]
        except:
            raise ValueError("ERROR: undefined west neighbour")

##________________________________A_STAR_______________________________________
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A.Akdogan
"""
from random import randint
import argparse


class Nodea:

    def __init__(self, x, y):

        self.x = x 
        self.y = y
        self.f = 0
        self.g = 0
        self.h = 0
        self.neighbors = []
        self.previous = None
        self. obstacle = False

    def add_neighbors(self,grid, columns, rows):

        neighbor_x = self.x
        neighbor_y = self.y
    
        if neighbor_x < columns - 1:
            self.neighbors.append(grid[neighbor_x+1][neighbor_y])
        if neighbor_x > 0:
            self.neighbors.append(grid[neighbor_x-1][neighbor_y])
        if neighbor_y < rows -1:
            self.neighbors.append(grid[neighbor_x][neighbor_y +1])
        if neighbor_y > 0: 
            self.neighbors.append(grid[neighbor_x][neighbor_y-1])
        #diagonals
        """ if neighbor_x > 0 and neighbor_y > 0:
            self.neighbors.append(grid[neighbor_x-1][neighbor_y-1])
        if neighbor_x < columns -1 and neighbor_y > 0:
            self.neighbors.append(grid[neighbor_x+1][neighbor_y-1])
        if neighbor_x > 0 and neighbor_y <rows -1:
            self.neighbors.append(grid[neighbor_x-1][neighbor_y+1])
        if neighbor_x < columns -1 and neighbor_y < rows -1:
            self.neighbors.append(grid[neighbor_x+1][neighbor_y+1]) """


        
class AStar:

    def __init__(self, cols, rows, start, end, obstacle_ratio = False, obstacle_list = False):

        self.cols = cols
        self.rows = rows
        self.start = start
        self.end = end
        self.obstacle_ratio = obstacle_ratio
        self.obstacle_list = obstacle_list

    @staticmethod
    def clean_open_set(open_set, current_node):

        for i in range(len(open_set)):
            if open_set[i] == current_node:
                open_set.pop(i)
                break

        return open_set

    @staticmethod
    def h_score(current_node, end):

        distance =  abs(current_node.x - end.x) + abs(current_node.y - end.y)
        
        return distance

    @staticmethod
    def create_grid(cols, rows):

        grid = []
        for _ in range(cols):
            grid.append([])
            for _ in range(rows):
                grid[-1].append(0)
        
        return grid

    @staticmethod
    def fill_grids(grid, cols, rows, obstacle_ratio = False, obstacle_list = False):

        for i in range(cols):
            for j in range(rows):
                grid[i][j] = Nodea(i,j)
                if obstacle_ratio == False:
                    pass
                else:
                    n = randint(0,100)
                    if n < obstacle_ratio: grid[i][j].obstacle = True
        if obstacle_list == False:
            pass
        else:
            for i in range(len(obstacle_list)):
                grid[obstacle_list[i][0]][obstacle_list[i][1]].obstacle = True

        return grid

    @staticmethod
    def get_neighbors(grid, cols, rows):
        for i in range(cols):
            for j in range(rows):
                grid[i][j].add_neighbors(grid, cols, rows)
        return grid
    
    @staticmethod
    def start_path(open_set, closed_set, current_node, end):

        best_way = 0
        for i in range(len(open_set)):
            if open_set[i].f < open_set[best_way].f:
                best_way = i

        current_node = open_set[best_way]
        final_path = []
        if current_node == end:
            temp = current_node
            while temp.previous:
                final_path.append(temp.previous)
                temp = temp.previous
            
            #print("Done !!")

        open_set = AStar.clean_open_set(open_set, current_node)
        closed_set.append(current_node)
        neighbors = current_node.neighbors
        for neighbor in neighbors:
            if (neighbor in closed_set) or (neighbor.obstacle == True):
                continue
            else:
                temp_g = current_node.g + 1
                control_flag = 0
                for k in range(len(open_set)):
                    if neighbor.x == open_set[k].x and neighbor.y == open_set[k].y:
                        if temp_g < open_set[k].g:
                            open_set[k].g = temp_g
                            open_set[k].h= AStar.h_score(open_set[k], end)
                            open_set[k].f = open_set[k].g + open_set[k].h
                            open_set[k].previous = current_node
                        else:
                            pass
                        control_flag = 1
  
                if control_flag == 1:
                    pass
                else:
                    neighbor.g = temp_g
                    neighbor.h = AStar.h_score(neighbor, end)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.previous = current_node
                    open_set.append(neighbor)

        return open_set, closed_set, current_node, final_path

    def main(self):

        grid = AStar.create_grid(self.cols, self.rows)
        grid = AStar.fill_grids(grid, self.cols, self.rows, obstacle_ratio = self.obstacle_ratio, obstacle_list = self.obstacle_list)
        grid = AStar.get_neighbors(grid, self.cols, self.rows)
        open_set  = []
        closed_set  = []
        current_node = None
        final_path  = []
        open_set.append(grid[self.start[0]][self.start[1]])
        self.end = grid[self.end[0]][self.end[1]]
        while len(open_set) > 0:
            open_set, closed_set, current_node, final_path = AStar.start_path(open_set, closed_set, current_node, self.end)
            if len(final_path) > 0:
                break

        return final_path


       



#====================================== END ==============================================
#===========================================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outputs the shortest path from startpoint to \
                        endpoint in the specified adjacency matrix .npz file")
    parser.add_argument("file", help="enter path of the.npz file generated by generator.py")
    parser.add_argument("rand_number", help="number of generating random coordinates", type=int)
    parser.add_argument("startx", help="start x coordinate", type=int)
    parser.add_argument("starty", help="start y coordinate", type=int)

    args = parser.parse_args()

    main( args.file, args.rand_number, [args.startx, args.starty])
