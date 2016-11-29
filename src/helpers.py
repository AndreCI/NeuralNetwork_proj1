"""Python script for the Unsupervised and 
Reinforcement Learning.
"""

import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rewritedata(indata):
    data_matrix = np.ones((np.shape(indata)[0],28,28))
    for i in range (0,28):
        data_matrix[:,i,:] = indata[:,i*28:(i+1)*28]
    return data_matrix

def visualize(number,label):
    assert np.shape(number)==(28,28)
    #plt.title('Image of ',label)
    #plt.annotate(('label is ',label),xy=(2, 1))
    plt.figure(figsize=(6,2))
    #plt.axis("off")
    plt.title(("An hand-written %s"%label))
    plt.imshow(number,cmap='Greys')
    plt.savefig("plots_report/data_visu.png")
    
#return eucl distance between the data and a center
def distance(vec1,vec2):
    assert np.shape(vec1)==np.shape(vec2)
    #print("   Distance between",vec1,"and",vec2)
    term1 = np.power(vec1-vec2,2)
    #print("      term1 :",term1)
    term2 = np.sqrt(np.sum(term1,axis=1))
    #print("      term2 :",term2)
    return term2

def assign_cluster(centers, datas):
    #initialization
    cluster_assignment = np.zeros(np.shape(datas)[0]) - 1
    current_c = np.tile(centers[0], (np.shape(datas)[0],1))
    eucl_distance = distance(current_c,datas)
    min_distance = np.copy(eucl_distance)
    #find cluster for each entry in datas
    for i in range(0,np.shape(centers)[0]):
        #print("Center is : ",centers[i], "iteration : ",i)
        current_c = np.tile(centers[i], (np.shape(datas)[0],1))
        eucl_distance = distance(current_c,datas)
        #print("   MD :",min_distance)
        #print("   ED :",eucl_distance)
        bool_distance = eucl_distance <= min_distance
        #print("   BD :",bool_distance)
        min_distance = eucl_distance*(bool_distance) + min_distance*(1 - bool_distance)
        #print("   MD :",min_distance)        
        cluster_assignment = cluster_assignment*(1 - bool_distance) + i*bool_distance
        #print("   CA :",cluster_assignment)
    return cluster_assignment

def reconstruction_error(centers,datas):
    #E = SUM{k}(SUM{u in C_k}((Wk - Xu)^2) by the course
    #E is mean error of quantization
    #Show that error decrease to prove convergence
    #Th. After conv, each proto is at the center of his data cloud
    E = 0
    cluster_assignment = assign_cluster(centers,datas)
    #print("Assignement is :",cluster_assignment)
    for i in range(0, np.shape(centers)[0]):
        #print("Center is : ",centers[i], "iteration : ",i)
        C_k = cluster_assignment == i #bool_tab to see if Xu is in C_k
        #print("   C_k :",C_k)
        current_c = np.tile(centers[i], (np.shape(datas)[0],1))
        C_ks = np.tile(C_k,(np.shape(datas)[1],1)).T
        values = datas * C_ks + current_c * (1 - C_ks) #if Xu is in C_k, Xu = data, else Xu = centers so Wk-Xu = 0
        #print("   values :",values)
        #print("   curren_c :",current_c)
        d = distance(current_c,values) #compute SUM{u in C_k}((Wk - Xu)^2) and add it to current E
        #print("   d :",d)
        E = E + sum(d)
        #print("   E :",E)
    return E

def check_conv(losses, conv_fact,nbrValues):
    assert(np.shape(losses)[0]>=nbrValues*2)
    part1 = losses[np.shape(losses)[0]-nbrValues:np.shape(losses)[0]]
    part2 = losses[np.shape(losses)[0]-nbrValues*2:np.shape(losses)[0]-nbrValues]
    p1 = sum(part1)/nbrValues
    p2 = sum(part2)/nbrValues
    return p1, p2, p1/p2>conv_fact
