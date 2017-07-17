__author__ = 'ritvi'

from pylab import *
from matplotlib import pyplot as plt

def read_data():
    data= load("data.npy")

    final_raw_data=np.zeros((data.shape[0],2))

    Orig_result=np.zeros(0)
    for i in range(data.shape[0]):
        final_raw_data[i]=data[i][:len(data[i])-1]
        Orig_result=np.append(Orig_result,data[i][len(data[i])-1])

    return (final_raw_data,Orig_result)

def dist_calc(point,data):
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
    dist_vector.sort(key=lambda x: x[1])
    #print(dist_vector)
    return dist_vector


def classification_Region(inp_data,result,k):
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

    plt.show()




def get_k_nearest_nbr(dist_vector,k):
    return dist_vector[0:k]



def main():
    inp_data,result = read_data()



    classification_Region(inp_data,result,1)
    #classification_Region(inp_data,result,15)
    #print(inp_data)
    #print(result)


main()