"""
Program to implement Logistic Regression for 2D Space data set.
The value of logistic function are rounded to 1( if value is great
than equal to 0.5 ) or to 0( if value is less than 0.5 )
The program takes input filename from user
Author :- Ritvik Joshi
"""

from csv import reader
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

def calc_sigmoid(val):
    '''
    logistic function, computes values between 0 and 1
    :param val: Solution of line function
    :return: sigmoid(val)
    '''
    return (np.exp(val)/(1+np.exp(val)))

def calc_line_funtio(value_x, weights):
    '''
    solve line equation (y=mx+c)
    :param value_x: numpy array contianing values of attributes
    :param weights: numpy array conatining coeff. for respective attributes
    :return: soln of line equation(y_cap)
    '''

    return (np.sum(value_x*weights))

def logistic_regression(data,weights,actual_class):
    """
    Trains regressor on the give input data
    :param data: input data
    :param weights: initial weights corresponding to respective attributes(all zero)
    :param actual_class: original class of the data(0 or 1)
    :return: Final weights(coeff), SSE(sum of squared error) for each epoch
    """
    #learning rate
    learning_constant = 0.03
    SSE_epoch=[]
    #number of iteration on the training data
    for epoch in range(70):
        SSE=0
        for r_index in range(len(data)):
            #computing y_cap and sigmoid function for each sample
            y_cap = calc_line_funtio(data[r_index],weights)
            y_sigmoid = calc_sigmoid(y_cap)

            #calculating SSE for the current epoch
            SSE+= (actual_class[r_index] - y_sigmoid)**2
            #updating weights for each attributes
            for weight_index in range(len(weights)):
                weights[weight_index] += learning_constant * (actual_class[r_index] - y_sigmoid) * (y_sigmoid*(1-y_sigmoid)) * data[r_index][weight_index]


        SSE_epoch.append(SSE)


    return weights,SSE_epoch


def prediction(data,weights):
     '''
     Predticion function
     :param data: test data
     :param weights: final weights after training
     :return: prediction of class(0 or 1)
     '''
     result=np.zeros((0))
     for r_index in range(len(data)):
        #computing y_cap and sigmoid function for each sample
        y_cap = calc_line_funtio(data[r_index],weights)
        y_sigmoid = calc_sigmoid(y_cap)
        #roundong sigmoid value
        if(y_sigmoid<0.5):
            y_sigmoid=0
        else:
            y_sigmoid=1
        #result list
        result=np.append(result,y_sigmoid)

     return result


def accuracy(actual_class,prediction):
    '''
    Comparing the result to actual class
    :param actual_class: Original classes
    :param prediction: Predicted classes
    :return:None
    '''
    count_match_zero=0
    count_match_one =0
    count_miss_zero=0
    count_miss_one = 0
    #Comparing classes
    for index in range(len(actual_class)):
        if(actual_class[index]==prediction[index]):
            if prediction[index]==1:
                count_match_one+=1
            else:
                count_match_zero+=1
        else:
            if prediction[index]==1:
                count_miss_one+=1
            else:
                count_miss_zero+=1

    print("********Confusion Matrix********\n")

    print(" \t 0\t      1")
    print("0\t",count_match_zero,"\t",count_miss_zero)
    print("1\t",count_miss_one,"\t",count_match_one)


    print("\n\nTotal Matched ::",(count_match_one+count_match_zero))
    print("Toatl MissMatched ::",(count_miss_one+count_miss_zero))
    print("Accuracy :: ",(((count_match_one+count_match_zero)/len(actual_class))*100))


def plot_SSE(SSE):
    '''
    plotting SSE vs Epoch graph
    :param SSE: List SSE for each epoch
    :return:
    '''

    plt.plot([x for x in range(1,len(SSE)+1)],SSE),plt.title("SSE vs Epoch"),plt.xlabel("Epoch"),plt.ylabel("SSE")
    plt.show()


def plot_data(data,weights):
    '''
    Plot data distribution and Decision boundary across data
    :param data: Input data
    :param weights: Final weights
    :return:None
    '''

    col=('blue','green')
    #Plotting data points onto the graph
    fig= plt.subplot()

    for index in range(len(data[0])):
          fig.scatter(data[0][index], data[1][index],c=col[int(data[2][index])],marker='x')
    plt.xlabel=("Attribute1"),plt.ylabel("Attribute2"),plt.title("Attribute and Class Distribution")
    blue_patch = mpatches.Patch(color='blue', label='Class 0')
    green_patch = mpatches.Patch(color='green', label='Class 1')
    red_patch = mpatches.Patch(color='red', label='Dec. Boundary')
    plt.legend(handles=[blue_patch,green_patch,red_patch],loc="upper left")
    #calculating min and max of attributes
    x1min=min(data[0])
    x1max=max(data[0])
    x1 = (data[0]-x1min)/(x1max+x1min)
    x2min=min(data[1])
    x2max=max(data[1])
    x2 = (data[1]-x2min)/(x2max+x2min)

    #Calculating line position
    ww = weights
    ex1 = np.linspace(x1min,x1max,endpoint=True)
    ex2 = -(ww[1]*ex1+ww[0])/ww[2]

    #plotting line
    plt.plot(ex1,ex2,c='r')

    plt.show()



def main():
    '''
    MAin function
    takes input file from user
    Store the data into a list of numpy array(for each row)
    Call all the other function
    :return:
    '''

    filename=input("Enter the filename")
    #filename='test2'
    file = open(filename, "r")
    attr_list = list(reader(file))

    data_list=attr_list[1:] #create the final list for passing
    data=[np.zeros((0)) for _ in range(len(data_list))]
    actual_class=np.zeros((0))
    weights=np.zeros((0))
    data2=[np.zeros((0)) for _ in range(len(data_list[0]))]
    #Stroing data into numpy array
    for r_index in range(len(data_list)):
        data[r_index]=np.append(data[r_index],1)

        for c_index in range(len(data_list[r_index])):
            data2[c_index]=np.append(data2[c_index],np.float(data_list[r_index][c_index]))
            if(c_index==len(data_list[r_index])-1):
                #Storing class in separate list
                 actual_class=np.append(actual_class,np.float(data_list[r_index][c_index]))
            else:
                #Storing data row wise
                data[r_index]=np.append(data[r_index],np.float(data_list[r_index][c_index]))

    #Intializing weight vector
    for _ in range(len(data_list[r_index])):
        weights = np.append(weights,0)

    #Calling training function
    weights,SSE = logistic_regression(data,weights,actual_class)

    #writting final weights onto a file
    target = open("weights_"+filename,"w")
    wt_string="Intercept:: "+str(weights[0])+" Coeff1:: "+str(weights[1])+" Coeff2::"+str(weights[2])
    target.write(wt_string)

    #Calling prediction function
    result=prediction(data,weights)

    #calling comparison function
    accuracy(actual_class,result)

    #plotting graphs
    plot_SSE(SSE)
    plot_data(data,weights)


main()