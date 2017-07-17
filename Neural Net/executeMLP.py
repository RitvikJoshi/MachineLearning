"""
Program to Execute Trained Multi layer perceptron Network
Takes input weight files
-{weights_0,weights_1,weights_2,weights_3,weights_4}
Take input data
-{test_data.csv}
Author : Ritvik Joshi, Rahul Dashora, Amruta Deshpande
"""

import numpy as np
import math
from csv import reader
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt

class confusion_matrix:

    __slots__='mat'

    def __init__(self):
        self.mat=[[0 for _ in range(4)] for _ in range(4)]
        #print(self.mat)

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



def prediction(data,weight_array):
     '''
     Predticion function
     :param data: test data
     :param weights: final weights after training
     :return: prediction of class(0 or 1)
     '''
     result=[]

     for r_index in range(len(data)):
        first_layer=np.array((1))
        output_layer=[]
        for weights in weight_array[0]:
            #computing y_cap and sigmoid function for each sample

            y_cap = calc_line_funtio(data[r_index],weights)
            y_sigmoid = calc_sigmoid(y_cap)
            first_layer=np.append(first_layer,y_sigmoid)

        for index in range(4):
            #print(first_layer,weight_array[1][index])
            y_cap = calc_line_funtio(first_layer,weight_array[1][index])

            y_sigmoid = calc_sigmoid(y_cap)

            output_layer.append(y_sigmoid)
        #print(output_layer)
        result.append(output_layer)

     return result

def accuracy(actual_class,prediction):
    '''
    Comparing the result to actual class
    :param actual_class: Original classes
    :param prediction: Predicted classes
    :return:None
    '''


    #Comparing classes
    conf_mat= confusion_matrix()
    for index in range(len(actual_class)):
        conf_mat.mat[int(actual_class[index]-1)][int(prediction[index]-1)]+=1

    #print(conf_mat.mat)
    print("********Confusion Matrix********\n")
    print(" \t1\t2\t3\t4")
    ind=1
    result=""
    for row in conf_mat.mat:
        result+=str(ind)+'\t'
        ind+=1
        for col in row:
            result+=str(col)+'\t'
        result+='\n'

    print(result)

    recognition=0

    for r_index in range(len(conf_mat.mat)):
        recognition+=conf_mat.mat[r_index][r_index]

    print("Recognition rate (%) :: ",(((recognition)/len(actual_class))*100))

    Cost_mat = [[20,-7,-7,-7],[-7,15,-7,-7],[-7,-7,5,-7],[-3,-3,-3,-3]]
    gain=0
    for r_index in range(4):
        for c_index in range(4):
            gain+=Cost_mat[r_index][c_index]* conf_mat.mat[r_index][c_index]

    print("Profit earned = ",gain, "cents")

def generatePlot(weight_total):
    xl = []
    yl = []
    data =[]
    dim = 100
    result = []
    for i in range(0, dim):
        for j in range(0, dim):
            xl.append(i / dim)
            yl.append(j / dim)
            #st = str((i / dim)) + "," + str((j / dim)) + "," + str(1) + '\n'
            data.append(np.array((1,(i/dim), (j/dim))))
           # print((np.array([(i/dim), (j/dim),1])))
   # print(weight_total)
    #print(data)
    result = prediction(data, weight_total)
    #print(result)
    x = np.array(xl)
    y = np.array(yl)
    finalres = []

    for records in result:
        max_index = records.index(max(records))
        finalres.append(max_index)
    res = np.array(finalres)
    #print(res)
    xx = x.reshape((dim, dim))

    yy = y.reshape((dim, dim))
    r2d = res.reshape((dim, dim))
    #print(r2d)
    return  xx,yy,r2d


def plot_data(data,contour):
    '''
    Plot data distribution and Decision boundary across data
    :param data: Input data
    :param weights: Final weights
    :return:None
    '''

    col=('blue','yellow','red','black')
    #Plotting data points onto the graph
    fig1= plt.subplot(3,2,1)
    plt.title('Weight-0')
    fig2= plt.subplot(3,2,2)
    plt.title('Weight-10')
    fig3= plt.subplot(3,2,3)
    plt.title('Weight-100')
    fig4= plt.subplot(3,2,4)
    plt.title('Weight-1000')
    fig5= plt.subplot(3,2,5)
    plt.title('Weight-10000')


    fig1.contourf(contour[0][0],contour[0][1],contour[0][2])
    fig2.contourf(contour[1][0],contour[1][1],contour[1][2])
    fig3.contourf(contour[2][0],contour[2][1],contour[2][2])
    fig4.contourf(contour[3][0],contour[3][1],contour[3][2])
    fig5.contourf(contour[4][0],contour[4][1],contour[4][2])
    for index in range(len(data[0])):
        fig1.scatter(data[0][index], data[1][index],c=col[int(data[2][index])-1],marker='x')
        fig2.scatter(data[0][index], data[1][index],c=col[int(data[2][index])-1],marker='x')
        fig3.scatter(data[0][index], data[1][index],c=col[int(data[2][index])-1],marker='x')
        fig4.scatter(data[0][index], data[1][index],c=col[int(data[2][index])-1],marker='x')
        fig5.scatter(data[0][index], data[1][index],c=col[int(data[2][index])-1],marker='x')


    blue_patch = mpatches.Patch(color='blue', label='Class 1')
    yellow_patch = mpatches.Patch(color='yellow', label='Class 2')
    red_patch = mpatches.Patch(color='red', label='Class 3')
    black_patch = mpatches.Patch(color='black', label='Class 4')

    #fig1.xlabel=("Attribute1"),fig1.ylabel("Attribute2"),fig1.title("Attribute and Class Distribution")
    fig1.legend(handles=[blue_patch,yellow_patch,red_patch,black_patch],loc="upper right")
    fig2.legend(handles=[blue_patch,yellow_patch,red_patch,black_patch],loc="upper right")
    fig3.legend(handles=[blue_patch,yellow_patch,red_patch,black_patch],loc="upper right")
    fig4.legend(handles=[blue_patch,yellow_patch,red_patch,black_patch],loc="upper right")
    fig5.legend(handles=[blue_patch,yellow_patch,red_patch,black_patch],loc="upper right")
    plt.show()



def main():
    filename=input("Enter the filename")
    #filename='test2'
    file = open(filename, "r")
    attr_list = list(reader(file))

    data_list=attr_list #create the final list for passing
    data=[np.zeros((0)) for _ in range(len(data_list))]
    actual_class=np.zeros((0))

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

    #print(data)
    region=[]
    for ind in range(5):
        print('reading:',"weights_"+str(ind)+".csv")
        weight_file = open("weights_"+str(ind)+".csv","r")

        counter=0
        weight_input_layer=[]
        weight_hidden_layer=[]
        #print(weight_file)
        for line in weight_file:
            #print(len(line))
            if len(line)==0:
                continue
            line=line.strip()
            line=line.split(',')
            #print(line)
            weights=np.zeros((0))
            if(counter<=4):
                counter+=1
                for value in line:
                    weights=np.append(weights,np.float(value))
                weight_input_layer.append(weights)
            else:
                for value in line:
                    weights=np.append(weights,np.float(value))
                weight_hidden_layer.append(weights)

        weight_total=[]
        weight_total.append(weight_input_layer)
        weight_total.append(weight_hidden_layer)

        #print('w',weight_total)
        xx,yy,finalres =generatePlot(weight_total)
        region.append((xx,yy,finalres))

        result=prediction(data,weight_total)


        final_classification=[]
        for records in result:
            max_index = records.index(max(records))
            #print(max_index+1)
            final_classification.append(max_index+1)

        accuracy(actual_class,final_classification)
        #print(final_classification)
    plot_data(data2,region)

main()