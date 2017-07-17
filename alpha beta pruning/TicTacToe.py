"""
Program to implement tic tac toe using Min Max algortihm and Alpha Beta Pruning
Author : Ritvik Joshi
"""

import copy
import time


"""
CLASS ttt_State:
Description:
It creates the game board and display it
"""


class ttt_state:
    __slots__="board","val_list","dimension"

    def __init__(self,dimension):
        """
        Intialize Game board
        :return:
        """
        self.dimension=dimension
        self.board=[['_' for _ in range(dimension)] for _ in range(dimension)]
        self.val_list=[]

    def display(self):
        """
        Display Game Board
        :return:
        """
        result=""
        for i in range(self.dimension):
            for j in range(self.dimension):
                result+=str(self.board[i][j])+"   "
            result+='\n'

        print(result)


def Successor(state,symbol):
    """
    Gives all the possible possible successor state for a given current state
    :param state: Current State of the game
    :param symbol: Indicate piece
    :return: list of Successor states
    """
    succ_list=[]
    for i in range(state.board.__len__()):
        for j in range(state.board.__len__()):
            if state.board[i][j] == '_':
                new_state = ttt_state(state.board.__len__())
                new_state.board = copy.deepcopy(state.board)
                new_state.board[i][j]=symbol
                succ_list.append(new_state)
                #print('Old board')
                #state.display()
                #print('New board')
                #new_state.display()


    return succ_list

def check_horizontal(state,x,y,symbol):
    """
    Check horizontal the Wining conditions
    :param state: Current state of the board game
    :param x: starting x co-ordinate
    :param y: starting y co-ordinate
    :param symbol: Indicate piece
    :return: if win than 1 else False
    """
    count=0
    if state.board[x][y]==symbol:
            count+=1
    if state.board[x][y+1]==symbol:
            count+=1
    if state.board[x][y+2]==symbol:
            count+=1

    if count ==3:
        return 1

    return False

def check_vertical(state,x,y,symbol):
    """
    Check vertical the Wining conditions
    :param state: Current state of the board game
    :param x: starting x co-ordinate
    :param y: starting y co-ordinate
    :param symbol: Indicate piece
    :return: if win than 1 else False
    """
    count=0
    if state.board[x][y]==symbol:
            count+=1
    if state.board[x+1][y]==symbol:
            count+=1
    if state.board[x+2][y]==symbol:
            count+=1

    if count ==3:
        return 1

    return False

def check_diagonal_1(state,x,y,symbol):
    """
    Check diagonal the Wining conditions
    :param state: Current state of the board game
    :param x: starting x co-ordinate
    :param y: starting y co-ordinate
    :param symbol: Indicate piece
    :return: if win than 1 else False
    """
    count=0

    if state.board[x][y]==symbol:
            count+=1
    if state.board[x+1][y+1]==symbol:
            count+=1
    if state.board[x+2][y+2]==symbol:
            count+=1

    if count ==3:
        return 1

    return False

def check_diagonal_2(state,x,y,symbol):
    """
    Check diagonal the Wining conditions
    :param state: Current state of the board game
    :param x: starting x co-ordinate
    :param y: starting y co-ordinate
    :param symbol: Indicate piece
    :return: if win than 1 else False
    """

    count=0
    if state.board[x][y]==symbol:
            count+=1
    if state.board[x+1][y-1]==symbol:
            count+=1
    if state.board[x+2][y-2]==symbol:
            count+=1

    if count ==3:
        return 1
    else:
        return 0


def check(state,symbol):
    """
    Call different checking functions
    :param state: Current state of the board game
    :param symbol: Indicate piece
    :return: max value of all the wining check functions
    """

    h_1=check_horizontal(state,0,0,symbol)
    h_2=check_horizontal(state,1,0,symbol)
    h_3=check_horizontal(state,2,0,symbol)
    d_1=check_diagonal_1(state,0,0,symbol)
    d_2=check_diagonal_2(state,0,state.board.__len__()-1,symbol)
    v_1 =check_vertical(state,0,0,symbol)
    v_2 =check_vertical(state,0,1,symbol)
    v_3 =check_vertical(state,0,2,symbol)

    return max(h_1,h_2,h_3,v_1,v_2,v_3,d_1,d_2)




def Utility(state):
    """
    Utility heuristic function
    :param state: current state
    :return: -1 if human wins, +1 if computer wins, 0 if tie
    """

    win= check(state,'O')
    loss=check(state,'X')

    if(win==1):
        return 1
    if(loss==1):
        return -1

    return 0


def Termination_Test(state):
    """
    GAme termination conditions
    :param state: Current state of the game
    :return: True if terminated else False
    """
    win=check(state,'O')
    if win==1:
        return True
    win=check(state,'X')
    if win==1:
        return True

    for i in range(state.board.__len__()):
        for j in range(state.board.__len__()):
            if state.board[i][j]=='_':
                return False

    return True





def Make_decision(state):
    """
    Calls Max_value  function and return the result
    :param state: Initial state
    :return: number of search node, choosen state, utility value, utility value list
    """
    value,move,val_list,count =Max_value(state,0)
    return move,value,val_list,count

def Max_value(state,count):
    """
    Max_value  function
    Choose best possible move that can be made
    :param state: Current state of the board
    :param count: count of search node
    :return: board, search node, utility value, utility value list
    """
    #print('Max')
    #state.display()
    if Termination_Test(state):
        #print('in Termination')
        return Utility(state),state,state.val_list,count
    else:
        value=-9999
        action=None
        for succ_state in Successor(state,'O'):
            temp=value
            count+=1
            value1,temp_state,child_list,count=Min_value(copy.deepcopy(succ_state),count)
            state.val_list.append(value1)
            value=max(value,value1)
            if value!=temp:
              action=succ_state
        #print(value,"max")
        return value,action,state.val_list,count

def Min_value(state,count):
    """
    Min_value  function
    Choose least favourable move that can be made
    :param state: Current state of the board
    :param count: count of search node
    :return: board, search node, utility value, utility value list
    """
    #print('Min')
    #state.display()
    value=9999
    if Termination_Test(state):
        return Utility(state),state,state.val_list,count
    else:
        action=state
        for succ_state in Successor(state,'X'):
            temp=value
            count+=1
            value1,temp_state,child_list,count=Max_value(copy.deepcopy(succ_state),count)
            state.val_list.append(value1)
            value=min(value,value1)
            if value!=temp:
              action=succ_state
        #print(value,"min")
        return value,action,state.val_list,count


def alpha_beta_pruning(state):
    """
    Calls alpha_beta_Max_value function
    :param state: current state
    :return: board, search node, utility value, utility value list
    """
    value,move,val_list,count =alpha_beta_Max_value(state,-999,999,0)
    return move,value,val_list,count

def alpha_beta_Max_value(state,alpha,beta,count):
    """
    alpha_beta_Max_value  function
    Choose most favourable move that can be made
    Donot search further if better move is already available
    :param state: Current state of the board
    :param count: count of search node
    :return: board, search node, utility value, utility value list
    """
    #print('Max')
    #state.display()
    if Termination_Test(state):
        #print('in Termination')
        return Utility(state),state,state.val_list,count
    else:
        value=-9999
        action=None
        for succ_state in Successor(state,'O'):
            temp=value
            count+=1
            value1,temp_state,child_list,count=alpha_beta_Min_value(succ_state,alpha,beta,count)
            state.val_list.append(value1)
            value=max(value,value1)
            if value!=temp:
              action=succ_state
            if value >= beta:
                return value,action,state.val_list,count
            alpha = max(alpha,value)

        #print(value,"max")
        return value,action,state.val_list,count

def alpha_beta_Min_value(state,alpha,beta,count):
    """
    alpha_beta_Min_value  function
    Choose least favourable move that can be made
    Donot search further if worst move is already available
    :param state: Current state of the board
    :param count: count of search node
    :return: board, search node, utility value, utility value list
    """

    #print('Min')
    #state.display()
    if Termination_Test(state):
        return Utility(state),state,state.val_list,count
    else:
        value=9999
        action=state
        for succ_state in Successor(state,'X'):
            temp=value
            count+=1
            value1,temp_state,child_list,count=alpha_beta_Max_value(succ_state,alpha,beta,count)
            state.val_list.append(value1)
            value=min(value,value1)
            if value!=temp:
              action=succ_state
            if value <= alpha:
                return value,action,state.val_list,count
            beta = min(beta,value)
        #print(value,"min")
        return value,action,state.val_list,count



"""
Main function
Take input from human
Calls computer's decision making function
Display board states, time consumed, utility values and number of search nodes
"""

def main():

    state = ttt_state(3)
    print("Inital Board")
    state.display()

    #Successor(state,'X')
    flag=False
    while(not Termination_Test(state)):
        print("+++++++++++ Human turn ++++++++++++++")
        row = int(input("Enter row "))
        col = int(input("Enter col "))
        if state.board[row][col] != '_':
            print("Enter valid row and col")
            continue
        state.board[row][col]='X'
        state.display()
        if(check(state,'X')==1):
            flag=True
            print("++++++++++++++++  Human Win's ++++++++++++++++++++++++")
            break
        print("+++++++++++ Computer's turn Using Min-Max ++++++++++++++")
        mm_state=copy.deepcopy(state)
        mm_state.val_list=[]
        start = time.time()
        min_max_state,value,val_list,count=Make_decision(mm_state)
        end = time.time()
        print("Time taken by Min-Max algo:: ",(end-start))
        print("Value List", val_list)
        print("Value choosen", value)
        print("Number of serch node::",count)
        min_max_state.display()

        print("+++++++++++ Computer's turn Using Alpha-Beta-Pruning ++++++++++++++")
        ab_state=copy.deepcopy(state)
        ab_state.val_list=[]
        start = time.time()
        alpha_beta_state,value,val_list,count=alpha_beta_pruning(ab_state)
        end = time.time()
        print("Time taken by Alpha-Beta pruning algo:: ",(end-start))
        print("Value List", val_list)
        print("Value choosen", value)
        print("Number of serch node::",count)
        alpha_beta_state.display()

        state = min_max_state
        if(check(state,'O')==1):
            flag=True
            print("++++++++++++++++  Computer Win's ++++++++++++++++++++++++")
            break

    if(not flag):
        print("+++++++++++++++++++++  Game tied ++++++++++++++++++++++")

main()