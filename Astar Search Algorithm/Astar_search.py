#rdmaze.py
#
# Implementation of A* algorithm, to determine path from start location to Goal in a maze/puzzle grid.
#In the graph, places avaiable to move are represented by '.' or 'S' or 'G'.
#Obstacle are represented as '*'.
#
# Input graph is present in the a file
#File name to be provided by user as an command line argument
# Usage: python rdmaze.py puzzleFilename
#
# Author: Ritvik Joshi, RIT, Oct. 2016
# Author: Rahul Dashora, RIT, Oct. 2016
# Author: Amruta Deshpande, RIT, Oct. 2016
#######################################################################################################################

import copy
import math
import numpy as np
from matplotlib import pyplot as plt
import sys

class dice:
    """
    Dice
    -Face_Value - Top value of the dice
    -North - Value north of the Dice current orientation
    -South - Value south of the Dice current orientation
    -East - Value East of the Dice current oreientation
    -West - Value West of the Dice current oreientation
    -dist_travelled - Distance travelled by the dice
    """
    __slots__ = "North","South","East","West","Face_value","dist_travelled"

    def __init__(self,North,South,West,East,Face_value,dist_travelled):
        self.North=North
        self.South =South
        self.East=East
        self.West =West
        self.Face_value=Face_value
        self.dist_travelled=dist_travelled


def display_dice(dice):
    """
    Display dice values in current orientation
    :param dice: current Orientation
    :return: String of values of dice
    """
    #print("F:",dice.Face_value," N:",dice.North," S:",dice.South," E:",dice.East," W:",dice.West)
    return "F:"+str(dice.Face_value)+" N:"+str(dice.North)+" S:"+str(dice.South)+" E:"+str(dice.East)+" W:"+str(dice.West)

class node:
    """
    Search Node
    -x : x- axis(row) location of the dice
    -y: y-axis(col) location of the dice
    -Dice: Current orientation of the dicce
    -Graph: Current state of the Puzzle
    -Parent: Previous node of current search node
    """
    __slots__ = "x","y","value","dice","graph","parent"

    def __init__(self,x,y,value,dice,graph,parent):
        self.x=x
        self.y=y
        self.value=value
        self.dice =dice
        self.graph=graph
        self.parent=parent


    def __str__(self):
        result=""
        result+="x:"+str(self.x)+" y:"+str(self.y)+" v:"+str(self.value)+" F:"+display_dice(self.dice)+"P:"+str(self.parent[0])+","+str(self.parent[1])+"\n"

        return result


class priority_ll():
    """
    Priority Class
    -Create a List to Store the Search node in descending oreder based on there function value F(n) = G(n)+H(n)
    -Pop the Last element on the list
    """

    __slots__ = "list"

    def __init__(self):
        self.list=[]

    def insert(self,node):
        """
        Insert the search node in the priority queue base on F(n) = G(n)+H(n)
        """
        if(self.list.__len__()==0):
            self.list.append(node)
        else:
            last=True
            for i in range(self.list.__len__()):
                if(self.list[i].x==node.x) and (self.list[i].y==node.y) and (self.list[i].value==node.value) and (self.list[i].dice.Face_value==node.dice.Face_value):
                    last=False
                    break
            if(last):
                self.list.append(node)
                self.list.sort(key=lambda x: x.value,reverse=True)

    def pop(self):
        """
        Reutrn Last element of the list
        """
        return self.list.pop()


    def __str__(self):
        result=""
        for i in range(self.list.__len__()):
            result+="x:"+str(self.list[i].x)+" y:"+str(self.list[i].y)+" v:"+str(self.list[i].value)+" F:"+str(self.list[i].dice.Face_value)+"\n"

        return result





def move_dice(dice,move):
    """
    Change the orientation of the dice based on the move made
    :param dice: current orientation of dice
    :param move: Move to be made
    :return:    Dice with updated orientation
    """

    if move=='North':
        dice.North = dice.Face_value
        dice.Face_value = dice.South
        dice.South = 7-dice.North
    elif move=='South':
        dice.South = dice.Face_value
        dice.Face_value = dice.North
        dice.North = 7-dice.South
    elif move=='East':
        dice.East = dice.Face_value
        dice.Face_value = dice.West
        dice.West = 7-dice.East
    elif move=='West':
        dice.West = dice.Face_value
        dice.Face_value = dice.East
        dice.East = 7-dice.West

    return dice


def Move_valid(rolling_dice,Move):
    """
    Check if the move by the dice is valid or not
    Basically if top faceValue of dice is not 6.
    :param rolling_dice: dice current orientation
    :param Move: Move to be made
    :return:    True - if valid, False - if not valid
    """
    if Move=='North':
        if(rolling_dice.South==6):
            return False
    elif Move=='South':
        if(rolling_dice.North==6):
            return False
    elif Move=='East':
        if(rolling_dice.West==6):
            return False
    elif Move=='West':
        if(rolling_dice.East==6):
            return False

    return True



def display(graph):
    """
    Displays puzzle
    :param graph: puzzle
    :return:
    """
    for i in graph:
        print(i)


def heuristic(point1,point2,type):
    """
    heuristic helper function- decides  which heuristic function to call
    :param point1: current location
    :param point2: goal location
    :param type: heuristic function indicator
    :return:    heuristic function result
    """
    if type==1:
        return (calc_euclidean_dist(point1,point2))
    if type==2:
        return (calc_manhattan_dist(point1,point2))
    if type==3:
        return (calc_Normed_Vector_dist(point1,point2))




def calc_manhattan_dist(point1,point2):
    """
    Heuristic function Manhattan Distance
    :param point1: current location
    :param point2:  goal location
    :return:    heuristic function value
    """
    return (abs(point1[0]-point2[0])+abs(point1[1]-point2[1]))


def calc_euclidean_dist(point1,point2):
    """
    Heuristic function Euclidean Distance
    :param point1: current location
    :param point2:  goal location
    :return:    heuristic function value
    """

    return math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)


def calc_Normed_Vector_dist(point1,point2):
    """
    Heuristic function Normalizied Vector Space
    :param point1: current location
    :param point2:  goal location
    :return:    heuristic function value
    """
    return (max(abs(point1[0]-point2[0]),abs(abs(point1[1])-abs(point2[1]))))



def a_start(prio_queue,goal,type):
    """
    A* Start Algorithm to identify the path to the goal based on the heuristic function
    :param prio_queue: priority queue to store search nodes based on the F(n) = G(n)+H(n)
    :param goal: tuple with goal location
    :param type: type of heuristic function (1- Euclidean, 2- Manhattan, 3- Normalized Vector Space)
    :return:result(True-Path found, False-Not Found), Graph state, Number of visited search node, Number of generated node
    """
    count_visited=0                     #Intializing visited node
    count_generated=1                   #Intializing generated node
    while(prio_queue.list.__len__()>0):
        current_state = prio_queue.pop()                                    #Popping search node with minimum cost to goal based on F(n)=G(n)+H(n)
        curr_graph=current_state.graph[current_state.graph.__len__()-1][0]     #Current state of the puzzle
        curr_graph[current_state.x][current_state.y]=str(current_state.dice.Face_value)             #Assigning Dice FaceVaule to the graph location
        if current_state.x == goal[0] and current_state.y == goal[1] and current_state.dice.Face_value==1 :             #Checking base Condition for termintaion/ if goal condition statisfied
            return True,current_state,count_visited,count_generated                                 #If True rturn the results
        count_visited+=1

        if(current_state.x+1<curr_graph.__len__()):                         #Checking if North neigbhouring location available to move
            if(curr_graph[current_state.x+1][current_state.y]!='*'):        #Checking for obstacle
                if(Move_valid(copy.deepcopy(current_state.dice),'South')):  #Checking valid dice move
                    if(current_state.x+1==current_state.parent[0] and current_state.y==current_state.parent[1]):
                        pass                                                #Exculding move from child to parent to avoid infinite loop
                    else:
                        moved_dice=move_dice(copy.deepcopy(current_state.dice),'South')         #Moving dice to neigbhouring location
                        moved_dice.dist_travelled+=1
                        parent_x = copy.deepcopy(current_state.x)                               #Search node attributes
                        parent_y = copy.deepcopy(current_state.y)
                        current_state.graph.append((copy.deepcopy(curr_graph),moved_dice))
                        graph_list=copy.deepcopy(current_state.graph)

                        #Inserting serach node into priority queue
                        prio_queue.insert(node(current_state.x+1,current_state.y,
                                           (moved_dice.dist_travelled+heuristic((current_state.x+1,current_state.y),goal,type))
                                           ,moved_dice,graph_list,(parent_x,parent_y)))
                        count_generated+=1

        if(current_state.x-1>=0):                                               #Checking if South neigbhouring location available to move
            if(curr_graph[current_state.x-1][current_state.y]!='*'):            #Checking for obstacle
                if(Move_valid(copy.deepcopy(current_state.dice),'North')):      #Checking valide dice move
                    if(current_state.x-1==current_state.parent[0] and current_state.y==current_state.parent[1]):
                        pass
                    else:

                        moved_dice=move_dice(copy.deepcopy(current_state.dice),'North')         #Moving dice to neigbhouring location
                        moved_dice.dist_travelled+=1
                        parent_x = copy.deepcopy(current_state.x)                               #Search node attributes
                        parent_y = copy.deepcopy(current_state.y)
                        current_state.graph.append((copy.deepcopy(curr_graph),moved_dice))
                        graph_list=copy.deepcopy(current_state.graph)

                        #Inserting serach node into priority queue
                        prio_queue.insert(node(current_state.x-1,current_state.y,
                                               (moved_dice.dist_travelled+heuristic((current_state.x-1,current_state.y),goal,type))
                                               ,moved_dice,graph_list,(parent_x,parent_y)))
                        count_generated+=1

        if(current_state.y+1<curr_graph[0].__len__()):                          #Checking if East neigbhouring location available to move
            if(curr_graph[current_state.x][current_state.y+1]!='*'):            #Checking for obstacle
                if(Move_valid(copy.deepcopy(current_state.dice),'East')):       #Checking valid dice move
                    if(current_state.x==current_state.parent[0] and current_state.y+1==current_state.parent[1]):
                        pass
                    else:
                        moved_dice=move_dice(copy.deepcopy(current_state.dice),'East')           #Moving dice to neigbhouring location
                        moved_dice.dist_travelled+=1
                        parent_x = copy.deepcopy(current_state.x)                               #Search node attributes
                        parent_y = copy.deepcopy(current_state.y)
                        current_state.graph.append((copy.deepcopy(curr_graph),moved_dice))
                        graph_list=copy.deepcopy(current_state.graph)

                        #Inserting serach node into priority queue
                        prio_queue.insert(node(current_state.x,current_state.y+1,
                                               (moved_dice.dist_travelled+heuristic((current_state.x,current_state.y+1),goal,type)),
                                               moved_dice,graph_list,(parent_x,parent_y)))
                        count_generated+=1

        if(current_state.y-1>=0):                                           #Checking if West neigbhouring location available to move
            if(curr_graph[current_state.x][current_state.y-1]!='*'):         #Checking for obstacle
                if(Move_valid(copy.deepcopy(current_state.dice),'West')):   #Checking valid dice move
                    if(current_state.x==current_state.parent[0] and current_state.y-1==current_state.parent[1]):
                        pass
                    else:
                        moved_dice=move_dice(copy.deepcopy(current_state.dice),'West')          #Moving dice to neigbhouring location
                        moved_dice.dist_travelled+=1
                        parent_x = copy.deepcopy(current_state.x)                               #Search node attributes
                        parent_y = copy.deepcopy(current_state.y)
                        current_state.graph.append((copy.deepcopy(curr_graph),moved_dice))
                        graph_list=copy.deepcopy(current_state.graph)

                        #Inserting serach node into priority queue
                        prio_queue.insert(node(current_state.x,current_state.y-1,
                                               (moved_dice.dist_travelled+heuristic((current_state.x,current_state.y-1),goal,type)),
                                               moved_dice,graph_list,(parent_x,parent_y)))
                        count_generated+=1


    #Return if path is not present
    return False,None,count_visited,count_generated
        #print(prio_queue)


def plot_path(h_gen,h_vis):
    '''
    Plotting Bar graph of the result of all three heuristic
    :param h_gen: List of number of generated search nodes
    :param h_vis: List of number of visited search nodes
    :return:
    '''

    N = 3
    ind = np.arange(3)
    width = 0.25

    fig, vis_plot = plt.subplots()

    bar1 = vis_plot.bar(ind, h_vis, width, color='Y')   #Plot bar for visited search node

    vis_plot.set_ylabel('Number of states Visited')
    vis_plot.set_title('Heuristic Functions')
    vis_plot.set_xticks(ind + width/2)
    vis_plot.set_xticklabels(('H1(Euclidean) ', 'H2(Manhattan)', 'H3(Normalized Vector Space)'))

    for rect in bar1:                                      #Displaying numerical value of the result
        height = rect.get_height()
        vis_plot.text(rect.get_x() + rect.get_width()/2., height,
                '%d' % int(height),
                ha='center', va='bottom')


    fig, gen_plot = plt.subplots()
    bar2 = gen_plot.bar(ind + width, h_gen, width, color='G')   #plot bar for generated serach node

    gen_plot.set_ylabel('Number of states Generated')
    gen_plot.set_title('Heuristic Functions')
    gen_plot.set_xticks(ind+width + width/2)
    gen_plot.set_xticklabels(('H1(Euclidean) ', 'H2(Manhattan)', 'H3(Normalized Vector Space)'))

    for rect in bar2:
        height = rect.get_height()
        gen_plot.text(rect.get_x() + rect.get_width()/2., height,
                '%d' % int(height),
                ha='center', va='bottom')


    plt.show()


def main():
    '''
    Main function
    -create dice with Initial orientation
    -read input file for puzzle from command prompt
    -creates priority queue object to store search node in descending order based on f(n)
    -Determine the location of start node and goal node in the maze.
    -Call A_star algorithm with different heuristic function
    -Displays result of the heuristic search for the goal node
    :return:
    '''

    if len(sys.argv) != 2:
        print('Usage: python3 rdmaze.py puzzleFilename')        #Read the input filename
        return
    else:
        rolling_dice = dice(2,7-2,7-3,3,1,0)                    #Initial orientation of the dice
        #display_dice(rolling_dice)
        graph =[]
        h_visited=[]
        h_generated=[]
        prio_queue = priority_ll()                              #Initialize priority queue
        File = open(sys.argv[1])                                #Read puzzle data from input file
        for line in File:
            data = list(line.strip())
            graph.append(data)                                  #Puzzle strucutre stored in 2D list dataset

        start_x=0
        start_y=0
        goal_x=0
        goal_y=0

        for i in range(graph.__len__()):
            for j in range(graph[i].__len__()):
                if(graph[i][j]=='S'):                           #Determining start location in the puzzle
                    start_x=i
                    start_y=j
                if(graph[i][j]=='G'):                           #Determining goal location in the puzzle
                    goal_x = i
                    goal_y = j



        #Creating root node
        start_node_1=node(start_x,start_y,heuristic((start_x,start_y),(goal_x,goal_y),1),rolling_dice,[(graph,rolling_dice)],(-1,-1))

        prio_queue.insert(start_node_1)                         #Adding start/root search node into priority queue



        #For Heuristice 1 - Euclidean Distance is used

        result,state,count_v,count_g=a_start(prio_queue,(goal_x,goal_y),1)  #Calling A* algorithm for heuristic -1 (Euclidean)
        h_visited.append(count_v)                                           #Number of visited search node
        h_generated.append(count_g)                                         #Number of generated search node

        #Display result
        if(result==True):
            print('****************path found -Euclidean***************')
            print("Distance travelled:: ",state.dice.dist_travelled)
            print("Number of nodes visited:: ",count_v)
            print("Number of nodes generated:: ",count_g)

            for i in  range(0,state.graph.__len__()):
                if(i==1):
                    pass
                else:
                    print('*************Orientation',display_dice(state.graph[i][1]))
                    display(state.graph[i][0])


        else:
            print('****************path not found - Euclidean***************')
            print("Number of nodes visited:: ",count_v)
            print("Number of nodes generated:: ",count_g)

        print('******************************************************************************')

        #For Heuristice 2 - Manhattan Distance is used

        #Reinitializing for heuristic 2
        start_node_2=node(start_x,start_y,heuristic((start_x,start_y,2),(goal_x,goal_y),2),rolling_dice,[(graph,rolling_dice)],(-1,-1))
        prio_queue = priority_ll()
        prio_queue.insert(start_node_2)

        result,state,count_v,count_g=a_start(prio_queue,(goal_x,goal_y),2)         #Calling A* algorithm for heuristic -2 (Manhattan)
        h_visited.append(count_v)                                                   #Number of visited search node
        h_generated.append(count_g)                                                  #Number of generated search node
        #Displaying result for heuristic-2
        if(result==True):
            print('****************path found - Manhattan***************')
            print("distance travelled:: ",state.dice.dist_travelled)
            print("Number of nodes visited:: ",count_v)
            print("Number of nodes generated:: ",count_g)

            for i in  range(0,state.graph.__len__()):
                if(i==1):
                    pass
                else:

                    print('*************Orientation',display_dice(state.graph[i][1]))
                    display(state.graph[i][0])

        else:
            print('****************path not found - Manhattan***************')
            print("Number of nodes visited:: ",count_v)
            print("Number of nodes generated:: ",count_g)
        print('******************************************************************************')

        #For Heuristice 3 - Normalized Vector Space

        #Reinitializing for heuristic 3
        start_node_2=node(start_x,start_y,heuristic((start_x,start_y,3),(goal_x,goal_y),3),rolling_dice,[(graph,rolling_dice)],(-1,-1))
        prio_queue = priority_ll()
        prio_queue.insert(start_node_2)

        result,state,count_v,count_g=a_start(prio_queue,(goal_x,goal_y),3)              #Calling A* algorithm for heuristic -3 (Manhattan)
        h_visited.append(count_v)                                                       #Number of visited search node
        h_generated.append(count_g)                                                     #Number of generated search node

        #Display result for Heuristic -3
        if(result==True):
            print('****************path found - Normed Vector***************')
            print("distance travelled:: ",state.dice.dist_travelled)
            print("Number of nodes visited:: ",count_v)
            print("Number of nodes generated:: ",count_g)

            for i in  range(0,state.graph.__len__()):
                if(i==1):
                    pass
                else:
                    print('*************Orientation',display_dice(state.graph[i][1]))
                    display(state.graph[i][0])



        else:
            print('****************path not found - Normed Vector***************')
            print("Number of nodes visited:: ",count_v)
            print("Number of nodes generated:: ",count_g)

        #Plotting bar grap for result of all three heuristic
        plot_path(h_generated,h_visited)


main()