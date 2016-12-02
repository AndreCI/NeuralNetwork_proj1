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

def visualize_neurons(path,centers,eta,ite_num):
    plb.close('all')
    plb.figure(ite_num)
    for i in range(0,size_k**2):
        plb.subplot(size_k,size_k,i+1)
        plb.imshow(np.reshape(centers[i,:], [28, 28]),cmap='Greys',interpolation='bilinear')
        plb.axis('off')
    plb.annotate("plot_num:"+str(ite_num),xy=(0,27))
    plb.savefig((str(path)+"/plot_"+"_sigma"+str(sigma)+"_eta"+str(eta)+"_num"+str(ite_num)+".png"))
    plb.close(i)
    plb.show()
    plb.draw()
    
def get_attribute_visualize_label(centers_label):
    plb.plot(centers_label)
    
def visualize_label(path,centers_label,eta,ite_num):
    plb.close('all')
    plb.figure(ite_num)
    for i in range(0,size_k**2):
        labe = centers_label[i,:]
        lab = [value for value in labe if value != 0]
        
        plb.subplot(size_k,size_k,i+1)
        plb.axis([0,4,0,max(lab)+1])
        plb.axis('off')
        plb.bar([0,1,2,3],lab,1.0)
        for k in range(0,np.shape(targetdigits)[0]):
            plb.annotate(targetdigits[k],xy=(k,lab[k]+0.2))
            #if(targetdigits[k]==max(targetdigits)):
                #plb.annotate("WINNER",xy=(i,lab[k]+0.6))    
    plb.savefig((str(path)+"/labels_"+"_sigma"+str(sigma)+"_eta"+str(eta)+"_num"+str(ite_num)+".png"))
    plb.show()
    plb.plot()
