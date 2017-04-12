
from pylab import *
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
"""
Program Description: Maximum a postirori

Author: Ritvik Joshi
"""


def read_data_map():
    '''
    Read the input data and return the result and data points
    :return:
    '''
    data= load("data.npy")
    cols=['m','y']


    result=np.zeros(0)

    for i in range(data.shape[0]):
        index = int(data[i][2])
        #print(index)
        plt.scatter(data[i][0],data[i][1],c=cols[index],s=15)

    patch1 = mpatches.Patch(color="Magenta", label='Class 0')
    patch2 = mpatches.Patch(color="Yellow", label='Class 1')
    plt.legend(handles=[patch1,patch2],loc="upper left")
    plt.title("MAP")
    for i in range(data.shape[0]):
        result=np.append(result,data[i][len(data[i])-1])

    return (data,result)


def covariance_matrix(input_data):
    '''
    Return the covariance matrix of the input data
    :param input_data: input data
    :return: covariance matrix
    '''
    mean_vector = np.array((np.average(np.hsplit(input_data,np.array([1,2]))[0]),
                                     np.average(np.hsplit(input_data,np.array([1,2]))[1])))
    value1 = input_data - mean_vector
    mat = np.asmatrix(value1)
    mat_t = np.transpose(mat)

    cov = (mat_t * mat)/(len(input_data)-1)

    return cov


def mean_vector_cal(input_data):
    '''
    Calculate mean vector
    :param input_data: input data
    :return: mean vector
    '''
    return np.array((np.average(np.hsplit(input_data,np.array([1,2]))[0]),
                                     np.average(np.hsplit(input_data,np.array([1,2]))[1])))


def class_prior(result):
    '''
    calculate prior probability of the classes
    :param result: class vector
    :return: prior probability vector
    '''
    n=np.unique(result)
    prior = np.zeros(0)
    for i in range(len(n)):
        prior = np.append(prior,(list(result).count(n[i])/len(result)))

    return prior

def Gaussain_cal(point,mean_vector,N,Cov_mat):
    '''
    Calculate Gaussian Probability of input vector
    :param point: input point
    :param mean_vector: mean vector of training data set
    :param N: number of classes
    :param Cov_mat: Covariance matrix of a class
    :return: conditional density
    '''
    value1= 1/np.sqrt((2*np.pi)**N * np.linalg.det(Cov_mat))
    value2= np.exp((-1*(np.dot(np.dot((point-mean_vector) , np.linalg.inv(Cov_mat)) , np.transpose(point-mean_vector)))/2))
    cond_density= value1 * value2

    return cond_density

def MAP(data,result):
    '''
    Maximum a posteriori
    :param data: input data
    :param result: class vector
    :return: Result for train data
    '''
    raw_data_Zero=[]
    raw_data_One=[]
    for i in range(data.shape[0]):
        if(data[i][len(data[i])-1]==0):
            raw_data_Zero.append(data[i][:len(data[i])-1])
        else:
            raw_data_One.append(data[i][:len(data[i])-1])



    input_data=np.array((np.array(raw_data_Zero),np.array(raw_data_One)))
    N = len(np.unique(result))
    min_val = np.min(input_data)
    max_val = np.max(input_data)

    cols=['b','g']
    sym=['0','x']

    points_Array = np.arange(min_val,max_val+0.2,0.1)

    final_result=np.zeros((len(points_Array)**2))

    Posterior_prob = []
    mean_vector=[]
    Cov_mat=[]
    #covariance and Mean vector calculation
    for i in range(input_data.shape[0]):
        mean_vector.append(mean_vector_cal(input_data[i]))
        Cov_mat.append(covariance_matrix(input_data[i]))
        #print("c::",Cov_mat[i])

    Cov_mat= np.array(Cov_mat)
    mean_vector=np.array(mean_vector)
    #Prior Probability calculation
    Prior_prob = class_prior(result)

    x_axis = []
    y_axis= []
    counter=0
    #Maximum A Postiori calculation
    for x in points_Array:
        for y in points_Array:
            for i in range(N):
                point = np.array((x,y))
                Post_prob= Gaussain_cal(point,mean_vector[i],N,Cov_mat[i]) * Prior_prob[i]
                Posterior_prob.append(Post_prob)

            final_result[counter]=Posterior_prob.index(max(Posterior_prob))
            x_axis.append(x)
            y_axis.append(y)

            Posterior_prob=[]
            counter+=1

    #plotting data points
    plt.scatter(x_axis,y_axis,c=final_result,s=1)
    #xx,yy=np.meshgrid(points_Array,points_Array)

    #z= final_result.reshape(xx.shape)
    #plt.contour(yy, xx, z, cmap=plt.cm.Paired)
    x_list=[]
    y_list=[]
    counter=0
    #plot function
    for x in points_Array:
        for y in points_Array:
            if ((counter)<(len(final_result)-1) and final_result[counter]!= final_result[counter+1] and y<max_val):
                    x_list.append(x)
                    y_list.append(y)
            counter+=1
    plt.scatter(x_list,y_list,c="black",s=3)
    final_result = [0 for _ in range(len(data))]
    counter=0

    #Model Testing
    for k in range(N):
        for point in input_data[k]:
            for i in range(N):
                Post_prob= Gaussain_cal(point,mean_vector[i],N,Cov_mat[i]) * Prior_prob[i]
                Posterior_prob.append(Post_prob)

            final_result[counter]=Posterior_prob.index(max(Posterior_prob))
            Posterior_prob=[]
            counter+=1


    return final_result

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
    print("Total MissMatched ::",(count_miss_one+count_miss_zero))
    print("Accuracy :: ",(((count_match_one+count_match_zero)/len(orig_class))*100))




def main():
    '''
    Main function
    Call's MAP function
    Call's accuracy function
    :return:
    '''
    input_data,Orig_result= read_data_map()
    result=MAP(input_data,Orig_result)
    accuracy(Orig_result,result)
    plt.savefig("Map.png")
    plt.show()






main()