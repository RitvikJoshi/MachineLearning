from pylab import *
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def read_data_knn():
    """
    Read data for K NN classification
    Split the target variable from the input data
    return: Np array conatining attributes, Np array conatining result
    """
    data= load("data.npy")

    final_raw_data=np.zeros((data.shape[0],2))

    Orig_result=np.zeros(0)
    for i in range(data.shape[0]):
        final_raw_data[i]=data[i][:len(data[i])-1]
        Orig_result=np.append(Orig_result,data[i][len(data[i])-1])

    return (final_raw_data,Orig_result)

def dist_calc(point,data):
    """
    Calculate distance between point and the reaming data points in the input data vector
    :param point: Current point in data vector
    :param data: data Vector
    :return: distance Vector
    """
    distance_vector = []
    for points in range(len(data)):
        distance_vector.append((points,calc_distance(point,data[points])))
    nbr=Sort_nbr(distance_vector)

    return nbr


def calc_distance(point1, point2):
    '''
    euclidean distance
    :param point1:
    :param point2:
    :return:
    '''

    dist = (np.sum((point1-point2)**2))**(1/2)
    return dist

def Sort_nbr(dist_vector):
    """
    Sort the Distance vector
    return: Sorted Distance vector
    """
    dist_vector.sort(key=lambda x: x[1])
    #print(dist_vector)
    return dist_vector


def plot_knn(inp_data,result,k):
    """
    Plot classification region and Decision boundary and train data points for K-NN
    :param inp_data: Train data points
    :param result: result of K-NN
    :param k: Value of k
    :return: Final Result of k-nn
    """
    cols=['b','g']
    sym=['0','x']


    min_val = np.min(inp_data)
    max_val = np.max(inp_data)
    #print(min_val,max_val)

    points_Array = np.arange(min_val,max_val+0.2,0.1)

    final_result=np.zeros(0)
    #print(points_Array)

    for i in points_Array:
        counter=0
        for j in points_Array:
            point = np.array((i,j))
            dist = dist_calc(point,inp_data)
            k_nbr = get_k_nearest_nbr(dist,k)

            K_sum=0
            #print(len(k_nbr))
            for nbr in k_nbr:
                K_sum+=result[nbr[0]]
            K_sum=int(K_sum)
            Out_class=round(K_sum/k)
            final_result= np.append(final_result,Out_class)

            counter+=1
            plt.scatter(i,j,c=cols[Out_class],s=1)

    xx,yy=np.meshgrid(points_Array,points_Array)

    z= final_result.reshape(xx.shape)
    plt.contour(yy, xx, z, cmap=plt.cm.Paired)

    for i in range(inp_data.shape[0]):
        index = int(result[i])
        #print(index)
        plt.scatter(inp_data[i][0],inp_data[i][1],c=cols[index],s=15)

    final_result=np.zeros(0)
    for pair in inp_data:
        dist = dist_calc(pair,inp_data)
        k_nbr = get_k_nearest_nbr(dist,k)
        K_sum=0
        #print(len(k_nbr))
        for nbr in k_nbr:
            K_sum+=result[nbr[0]]
        K_sum=int(K_sum)
        Out_class=round(K_sum/k)
        final_result= np.append(final_result,Out_class)

    blue_patch = mpatches.Patch(color='blue', label='Class 0')
    green_patch = mpatches.Patch(color='green', label='Class 1')
    red_patch = mpatches.Patch(color='red', label='Dec. Boundary')

    plt.legend(handles=[blue_patch,green_patch,red_patch],loc="upper left")


    return final_result





def get_k_nearest_nbr(dist_vector,k):
    """
    Return k nearest neighbhours
    :param dist_vector:
    :param k:
    :return:
    """
    return dist_vector[0:k]


def read_data_linear():
    """
    Read data for linear model
    Split the target variable and append one in front of input attributes
    :return:
    """
    data= load("data.npy")

    final_raw_data=np.zeros((data.shape[0],3))
    Orig_result=np.zeros(0)
    for i in range(data.shape[0]):
        final_raw_data[i]=np.append(1,data[i][:len(data[i])-1])
        Orig_result=np.append(Orig_result,data[i][len(data[i])-1])

    return (final_raw_data,Orig_result)



def get_best_bias(inp_data,result):
    """
    Get the best bias
    :param inp_data: Input data
    :param result: Target variable
    :return: Best Bias
    """
    Bias = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(inp_data),inp_data)),np.transpose(inp_data)),result)

    return Bias

def calc_line_funtio(value_x, weights):
    """
    Calc line function y =mx+c
    :param value_x: Value of x
    :param weights: associated w
    :return:
    """
    return (np.sum(value_x*weights))

def linear_model(data,weights):

    out=[]
    #number of iteration on the training data
    for r_index in range(len(data)):
        #computing y_cap and sigmoid function for each sample
        y_cap = calc_line_funtio(data[r_index],weights)

        if(y_cap>0.5):
            result=1
        else:
            result=0

        out.append(result)

    return out


def accuracy(orig_class,pred_class):
    '''
    Calculating the accuracy of the linear classifier
    :param orig_class: Original classes
    :param pred_class: Predicted classes
    :return:None
    '''
    count_match_zero=0
    count_match_one =0
    count_miss_zero=0
    count_miss_one = 0
    #Comparing classes
    for index in range(len(orig_class)):
        if(orig_class[index]==pred_class[index]):
            if pred_class[index]==1:
                count_match_one+=1
            else:
                count_match_zero+=1
        else:
            if pred_class[index]==1:
                count_miss_one+=1
            else:
                count_miss_zero+=1

    print("********Confusion Matrix********\n")

    print(" \t 0\t      1")
    print("0\t",count_match_zero,"\t",count_miss_zero)
    print("1\t",count_miss_one,"\t",count_match_one)


    print("\n\nTotal Matched ::",(count_match_one+count_match_zero))
    print("Toatl MissMatched ::",(count_miss_one+count_miss_zero))
    print("Accuracy :: ",(((count_match_one+count_match_zero)/len(pred_class))*100))





def plot_linear(inp_data,bias,result):
    """
    Plot for Linear model
    Plot Classification region, Decision boundary, train data
    :param inp_data: input data
    :param bias: Bias weight
    :param result: Target variable
    :return:
    """
    cols=['b','g']
    data_max=np.max(inp_data)+0.2
    data_min=np.min(inp_data)

    points_array=np.arange(data_min,data_max,0.1)

    region=np.zeros(((len(points_array)*len(points_array)),3))
    counter=0
    for i in points_array:
        for j in points_array:
            region[counter]=np.array((1,i,j))
            counter+=1

    out=np.array(linear_model(region,bias))
    counter=0
    for i in points_array:
        for j in points_array:
            index=int(out[counter])
            plt.scatter(i,j,c=cols[index],s=1)
            counter+=1
    xx,yy=np.meshgrid(points_array,points_array)

    z=out.reshape(xx.shape)

    for i in range(inp_data.shape[0]):
        index = int(result[i])
        #print(index)
        plt.scatter(inp_data[i][1],inp_data[i][2],c=cols[index],s=15)

    blue_patch = mpatches.Patch(color='blue', label='Class 0')
    green_patch = mpatches.Patch(color='green', label='Class 1')
    red_patch = mpatches.Patch(color='red', label='Dec. Boundary')

    plt.legend(handles=[blue_patch,green_patch,red_patch],loc="upper left")

    plt.contour(yy,xx,z,cmap=plt.cm.Paired)

    plt.show()




def Run_linear():
    """
    Linear model function
    :return:
    """
    inp_data,result = read_data_linear()

    bias=get_best_bias(inp_data,result)

    out = linear_model(inp_data,bias)

    accuracy(result,out)
    plot_linear(inp_data,bias,result)


def Run_knn():
    """
    K-NN Function
    :return:
    """
    inp_data,result = read_data_knn()

    k = int(input("Enter value of k"))
    out = plot_knn(inp_data,result,k)

    accuracy(result,out)
    plt.show()


def main():
    """
    Main function
    Makes calls for Linear or K-NN function
    :return:
    """
    print('1. Linear Model')
    print('2. K-NN')

    choice = input("Enter your choice::")

    if choice=='1':
        Run_linear()
    else:
        Run_knn()



main()