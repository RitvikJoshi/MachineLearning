from csv import reader
import numpy as np
import copy
from matplotlib import pyplot as plt
import random
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


def logistic_regression(data,actual_class):
    """
    Trains regressor on the give input data
    :param data: input data
    :param weights: initial weights corresponding to respective attributes(all zero)
    :param actual_class: original class of the data(0 or 1)
    :return: Final weights(coeff), SSE(sum of squared error) for each epoch
    """
    #learning rate
    weight_total=[]
    weight_input_layer=[]
    weight_hidden_layer=[]

    for _ in range(5):
        weights=np.zeros((0))
        for _ in range(len(data[0])):
            val =random.uniform(-1,1)

            weights=np.append(weights,val)
        weight_input_layer.append(weights)

    weight_total.append(weight_input_layer)

    for _ in range(4):
        weights=np.zeros((0))
        for _ in range(6):
            val = random.uniform(-1,1)

            weights=np.append(weights,val)
        weight_hidden_layer.append(weights)

    weight_total.append(weight_hidden_layer)


    learning_constant = 0.1
    SSE_epoch=[]
    weights_array=[]
    #number of iteration on the training data
    for epoch in range(10001):
        SSE=0

        for r_index in range(len(data)):
                #computing y_cap and sigmoid function for each sample
            #hiddenlayer calculation
            hidden_layer_output=np.array((1))
            for node in range(5):
                y_cap = calc_line_funtio(data[r_index],weight_input_layer[node])
                y_sigmoid = calc_sigmoid(y_cap)
                hidden_layer_output= np.append(hidden_layer_output,y_sigmoid)


            #output layer calculation
            output_layer_output=np.zeros((0))
            for node in range(4):
                y_cap = calc_line_funtio(hidden_layer_output,weight_hidden_layer[node])
                y_sigmoid = calc_sigmoid(y_cap)
                output_layer_output= np.append(output_layer_output,y_sigmoid)



            #output layer delta

            delta_output=(actual_class[r_index]-output_layer_output) * (output_layer_output*(1-output_layer_output))
            #calculating SSE
            SSE +=np.sum((actual_class[r_index]-output_layer_output)**2)



            #print('delta output',delta_output)
            #hidden layer delta
            delta_hidden_temp=np.zeros((0))

            for weight in range(6):
                delta_node=0
                for hidden_node in range(4):
                    delta_node+= weight_hidden_layer[hidden_node][weight] * delta_output[hidden_node]
                delta_hidden_temp=np.append(delta_hidden_temp,delta_node)

            delta_hidden= (hidden_layer_output*(1- hidden_layer_output)) * delta_hidden_temp


            #updating hidden_layer weight
            for weight in range(6):
                for output_node in range(4):

                    weight_hidden_layer[output_node][weight]+=learning_constant *(hidden_layer_output[weight] * delta_output[output_node])


            #updating input layer weight
            for weight in range(3):
                for hidden_node in range(5):

                    weight_input_layer[hidden_node][weight]+=learning_constant *(data[r_index][weight] * delta_hidden[hidden_node+1])
        SSE_epoch.append(SSE)
        if(epoch==0):
            weights_array.append(copy.deepcopy(weight_total))
        elif(epoch==10):
            weights_array.append(copy.deepcopy(weight_total))
        elif(epoch==100):
            weights_array.append(copy.deepcopy(weight_total))
        elif(epoch==1000):
            weights_array.append(copy.deepcopy(weight_total))
        elif(epoch==10000):
            weights_array.append(copy.deepcopy(weight_total))

    return weights_array,SSE_epoch


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
            #roundong sigmoid value
            # if(y_sigmoid<0.5):
            #     y_sigmoid=0
            # else:
            #     y_sigmoid=1
            #result list
            first_layer=np.append(first_layer,y_sigmoid)
        for index in range(4):
            y_cap = calc_line_funtio(first_layer,weight_array[1][index])
            y_sigmoid = calc_sigmoid(y_cap)
            output_layer.append(y_sigmoid)

        result.append(output_layer)
     return result

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

    col=('blue','green','red','black')
    #Plotting data points onto the graph
    fig= plt.subplot()

    for index in range(len(data[0])):
          fig.scatter(data[0][index], data[1][index],c=col[int(data[2][index])-1],marker='x')
    plt.xlabel=("Attribute1"),plt.ylabel("Attribute2"),plt.title("Attribute and Class Distribution")
    blue_patch = mpatches.Patch(color='blue', label='Class 1')
    green_patch = mpatches.Patch(color='green', label='Class 2')
    red_patch = mpatches.Patch(color='red', label='Class 3')
    black_patch = mpatches.Patch(color='black', label='Class 4')
    plt.legend(handles=[blue_patch,green_patch,red_patch,black_patch],loc="upper left")

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

    training_class=[]
    for classes in actual_class:
        if classes == 1:
            training_class.append(np.array((1,0,0,0)))
        elif classes ==2:
            training_class.append(np.array((0,1,0,0)))
        elif classes ==3:
            training_class.append(np.array((0,0,1,0)))
        elif classes ==4:
            training_class.append(np.array((0,0,0,1)))


    #Intializing weight vector

    #Calling training function
    weights_array,SSE = logistic_regression(data,training_class)
    #print(weights)
    #print(weights_array)
    ind=0
    for weight in weights_array:
        target = open("weights_"+str(ind)+".csv","w")
        ind+=1
        result=""
        for wt in weight:
            for w in wt:
                for v in range(len(w)):
                    result+=str(w[v])
                    if v+1==len(w):
                        pass
                    else:
                        result+=","
                result+="\n"
        #print("R",result)
        target.write(result)

    print("Weights files are created................")
    plot_SSE(SSE)
    plot_data(data2,weights_array)


main()