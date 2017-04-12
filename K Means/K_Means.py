"""
Program to implement k-means algorithm
Author - Ritvik Joshi
"""
__author__ = 'ritvi'

from csv import reader
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import copy


class node:
    '''
    Class node to store all the points
    '''
    __slots__='point','id'

    def __init__(self,point,id):
        self.point =point
        self.id=id


    def __repr__(self):
        result= str(self.id)
        for i in self.point:
            result+=" "+str(i)

        result+="\n"
        return result

def plot(cluster_list,centers):
    '''
    Plot the points into 3d space
    :param cluster_list:
    :param centers:
    :return:
    '''
    fig = plt.figure()
    img = fig.add_subplot(111, projection='3d')

    col=('b','r','g','c','m','y','k','w','gray','orange','pink','brown')
    symb = ('x','o','+','1','2','3','4','8','<','>','*','^')
    for cluster in range(len(cluster_list)):
        for points in cluster_list[cluster]:
            img.scatter(points.point[0], points.point[1], points.point[2], c=col[cluster], marker=symb[cluster])
            img.scatter(centers[cluster][0], centers[cluster][1], centers[cluster][2], c='k', marker='D')

    img.set_xlabel('X Label')
    img.set_ylabel('Y Label')
    img.set_zlabel('Z Label')

    plt.show()


def update_center(cluster):
    '''
    Update the center (medoid ) of the cluster
    :param cluster:
    :return:
    '''
    center = np.zeros((3))

    for points in cluster:
        for val_index in range(len(points.point)):
            center[val_index]+=points.point[val_index]

    center =center/len(cluster)
    min_dist=99999
    min_id=0
    for clust_index in range(len(cluster)):
        dist = calc_distance(cluster[clust_index].point,center)
        if(dist<min_dist):
            min_dist = dist
            min_id = clust_index

    medoid = cluster[min_id].point

    print(center,medoid)

    return medoid




def K_cluster(data,cluster_center):
    '''
    K means algorithm
    Cluster the points based on distance from k-centers
    Update the centers
    if centers not updated then stop
    :param data: points
    :param cluster_center:current cluster centers
    :return: final clusters and center
    '''
    clusters=[[] for _ in range(len(cluster_center))]
    distance_vector = [[] for _ in range(len(cluster_center))]

    for clust_center in range(len(cluster_center)):
        for points in data:
            distance_vector[clust_center].append(calc_distance(points.point,cluster_center[clust_center]))

    #print(len(distance_vector[0]),len(distance_vector))
    for dist_index in range(len(distance_vector[0])):
        min=999999
        clust_id=0
        for clust_center_index in range(len(distance_vector)):
            if(distance_vector[clust_center_index][dist_index]<min):
                min = distance_vector[clust_center_index][dist_index]
                clust_id = clust_center_index
        clusters[clust_id].append(data[dist_index])

    #print(clusters)
    new_cluster_center = cluster_center
    for clust_center_index in range (len(cluster_center)):
        print(clusters[clust_center_index])
        if (len(clusters[clust_center_index])>0):
            new_cluster_center[clust_center_index]=update_center(clusters[clust_center_index])

    print(new_cluster_center)

    if (termination_test(new_cluster_center,cluster_center)):
        return clusters,new_cluster_center
    else:
        clusters,new_cluster_center = K_cluster(data, new_cluster_center)
        return clusters,new_cluster_center


def termination_test(new_cluster_center,cluster_center):
    '''
    Termination condition for k- means
    if center donot change
    :param new_cluster_center:
    :param cluster_center:
    :return:
    '''
    flag=False
    for center_index in range(len(new_cluster_center)):
        if(new_cluster_center[center_index]==cluster_center[center_index]).all():
            flag=True
        else:
            return False

    return flag



def calc_distance(point1, point2):
    '''
    euclidean distance
    :param point1:
    :param point2:
    :return:
    '''
    dist = (np.sum((point1-point2)**2))**1/2
    return dist


def calc_SSE(clusters,center):
    '''
    Calculate SSE for the k-cluster
    :param clusters:
    :param center:
    :return:
    '''
    SSE=0
    for cluster_index in range(len(clusters)):
        for points in clusters[cluster_index]:
            SSE+= (calc_distance(points.point,center[cluster_index]))**2

    return SSE




def main():

    #filename=input("Enter the filename")
    filename='HW08_KMEANS_DATA_v300.csv'
    file = open(filename, "r")
    #reading data
    attr_list = list(reader(file))

    data_list=attr_list[1:] #create the final list for passing
    data=[]

    #print(data)
    for r_index in range(len(data_list)):
        temp=np.zeros((0))
        for c_index in range(len(data_list[r_index])):
            temp=np.append(temp,np.float(data_list[r_index][c_index]))
        data.append(node(temp,r_index))

    #print(data)


    clusters_list=[]
    centers_list=[]
    #Performing clustering
    for  i in range(1,13):
        nclust=i
        cluster_center=[]
        for i in range(nclust):
            cluster_center.append(data[random.randint(0, len(data)-1)].point)
        print(cluster_center)
        clusters,centers=K_cluster(copy.deepcopy(data),cluster_center)
        print(clusters,centers)
        clusters_list.append(clusters)
        centers_list.append(centers)
    #Calculating SSE
    SSE_list=[]
    for cluster_index in range(len(clusters_list)):
        SSE_list.append(calc_SSE(clusters_list[cluster_index],centers_list[cluster_index]))

    print(SSE_list)
    cluster_Count=[]

    print(SSE_list)
    #Determing best SSE and cluster
    best_SSE_index = SSE_list.index(min(SSE_list))
    best_cluster = clusters_list[best_SSE_index]
    best_centers = centers_list[best_SSE_index]

    temp=[]
    #Best cluster lengths
    for cluster in best_cluster:
        temp.append(len(cluster))

    print(temp)

    #plotting SSE - k(clusters) and best cluster 3d projection
    plt.plot([x for x in range(1,13)],SSE_list),plt.title('SSE-K(clusters)'),plt.xlabel('K -clusters'),plt.ylabel('SSE')
    plt.show()
    plot(best_cluster,best_centers)

main()
