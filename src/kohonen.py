"""Python script for Exercise set 6 of the Unsupervised and 
Reinforcement Learning.
"""

import numpy as np
import matplotlib.pylab as plb

def kohonen():
    """Example for using create_data, plot_data and som_step.
    """
    plb.close('all')
    
    dim = 28*28
    data_range = 255.0
    
    # load in data and labels    
    data = np.array(np.loadtxt('data/data.txt'))
    labels = np.loadtxt('data/labels.txt')

    # select 4 digits    
    name = 'Andre Cibils' # REPLACE BY YOUR OWN NAME
    targetdigits = name2digits(name) # assign the four digits that should be used
    print(targetdigits) # output the digits that were selected
    # this selects all data vectors that corresponds to one of the four digits
    data = data[np.logical_or.reduce([labels==x for x in targetdigits]),:]
    
    dy, dx = data.shape
    
    #set the size of the Kohonen map. In this case it will be 6 X 6
    size_k = 6
    
    #set the width of the neighborhood via the width of the gaussian that
    #describes it
    sigma = 3.0
    
    #initialise the centers randomly
    centers = np.random.rand(size_k**2, dim) * data_range
    
    #build a neighborhood matrix
    neighbor = np.arange(size_k**2).reshape((size_k, size_k))

    #set the learning rate
    eta = 0.5 # HERE YOU HAVE TO SET YOUR OWN LEARNING RATE
    
    #set the maximal iteration count
    tmax = 5000 # this might or might not work; use your own convergence criterion
    
    #set the random order in which the datapoints should be presented
    i_random = np.arange(tmax) % dy
    np.random.shuffle(i_random)
    
    for t, i in enumerate(i_random):
        print(t)
        print(i)
        som_step(centers, data[i,:],neighbor,eta,sigma)


    # for visualization, you can use this:
    for i in range(0,size_k**2):
        plb.subplot(size_k,size_k,i+1)
        
        plb.imshow(np.reshape(centers[i,:], [28, 28]),cmap='Greys',interpolation='bilinear')
        plb.axis('off')
        
    # leave the window open at the end of the loop
    plb.show()
    plb.draw()
   
    

def som_step(centers,data,neighbor,eta,sigma):
    """Performs one step of the sequential learning for a 
    self-organized map (SOM).
    
      centers = som_step(centers,data,neighbor,eta,sigma)
    
      Input and output arguments: 
       centers  (matrix) cluster centres. Have to be in format:
                         center X dimension
       data     (vector) the actually presented datapoint to be presented in
                         this timestep
       neighbor (matrix) the coordinates of the centers in the desired
                         neighborhood.
       eta      (scalar) a learning rate
       sigma    (scalar) the width of the gaussian neighborhood function.
                         Effectively describing the width of the neighborhood
    """
    
    size_k = int(np.sqrt(len(centers)))
    new_centers = np.copy(centers)
    #find the best matching unit via the minimal distance to the datapoint
    b = np.argmin(np.sum(np.abs(centers - np.resize(data,(size_k**2,data.size))),1))
    #b = np.argmin(np.sum((centers - np.resize(data, (size_k**2, data.size)))**2,1))
    # find coordinates of the winner
    a,b = np.nonzero(neighbor == b)
    # update all units
    for j in range(size_k**2):
        # find coordinates of this unit
        a1,b1 = np.nonzero(neighbor==j)
        # calculate the distance and discounting factor
        disc=gauss(np.sqrt((a-a1)**2+(b-b1)**2),[0, sigma])
        # update weights        
        new_centers[j,:] += disc * eta * (data - centers[j,:])
    return new_centers
 
def som_step_labels(centers,centers_label,data,label,neighbor,eta,sigma):
    """Performs one step of the sequential learning for a 
    self-organized map (SOM).
    
      centers = som_step(centers,data,neighbor,eta,sigma)
    
      Input and output arguments: 
       centers  (matrix) cluster centres. Have to be in format:
                         center X dimension
       centers_label (matrix) centers label. Similar to centers.
       data     (vector) the actually presented datapoint to be presented in
                         this timestep
       label    (vector) the actually label assigned to the data
       neighbor (matrix) the coordinates of the centers in the desired
                         neighborhood.
       eta      (scalar) a learning rate
       sigma    (scalar) the width of the gaussian neighborhood function.
                         Effectively describing the width of the neighborhood
    """
    label_ = np.zeros(10)
    label_[label] = 1
    
    size_k = int(np.sqrt(len(centers)))
    new_centers = np.copy(centers)
    new_centers_label = np.copy(centers_label)
    #find the best matching unit via the minimal distance to the datapoint
    b = np.argmin(np.sum(np.abs(centers - np.resize(data,(size_k**2,data.size))),1))
    #b = np.argmin(np.sum((centers - np.resize(data, (size_k**2, data.size)))**2,1))
    # find coordinates of the winner
    a,b = np.nonzero(neighbor == b)
    # update all units
    for j in range(size_k**2):
        # find coordinates of this unit
        a1,b1 = np.nonzero(neighbor==j)
        # calculate the distance and discounting factor
        disc=gauss(np.sqrt((a-a1)**2+(b-b1)**2),[0, sigma])
        # update weights        
        new_centers[j,:] += disc * eta * (data - centers[j,:])
        new_centers_label[j,:] += disc * eta * (label_ - centers_label[j,:])
    return new_centers, new_centers_label 
 
def gauss(x,p):
    """Return the gauss function N(x), with mean p[0] and std p[1].
    Normalized such that N(x=p[0]) = 1.
    """
    return np.exp((-(x - p[0])**2) / (2 * p[1]**2))
    
#return eucl distance between the data and a center
def distance(vec1,vec2):
    """Return the euclidian distance between two same dimension vector.
    vec1 and vec2 must been NxD vector, and it returns a N dimensional distance
    vector
    """
    assert np.shape(vec1)==np.shape(vec2)
    #print("   Distance between",vec1,"and",vec2)
    term1 = np.power(vec1-vec2,2)
    #print("      term1 :",term1)
    term2 = np.sqrt(np.sum(term1,axis=1))
    #print("      term2 :",term2)
    return term2
    
def assign_cluster(centers, datas):
    """Return, for each lign in the data, the number of the corresponding 
    center it must be assigned to. Uses euclidian distance.
    """
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
    """Compute the mean error of vector quantization for current centers.
    Uses assign_cluster.
    """
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
    """ Return a boolean indication if convergence has occured for the current
    parameters.
    Losses is a vector containing the losses, conv_fact must be in [0,1] and
    nbrValues is the number of entry to consider in the vector losses.
    """
    assert(np.shape(losses)[0]>=nbrValues*2)
    part1 = losses[np.shape(losses)[0]-nbrValues:np.shape(losses)[0]]
    part2 = losses[np.shape(losses)[0]-nbrValues*2:np.shape(losses)[0]-nbrValues]
    p1 = sum(part1)/nbrValues
    p2 = sum(part2)/nbrValues
    return p1, p2, p1/p2>conv_fact
    
def update_sigma(sigma, phi, floor):
    new = sigma*phi
    if(new<floor):
        new = floor
    return new

def update_step(step, phi, ceil):
    new = step*phi
    if(new>ceil):
        new = ceil
    return new

def name2digits(name):
    """ takes a string NAME and converts it into a pseudo-random selection of 4
     digits from 0-9.
     
     Example:
     name2digits('Felipe Gerhard')
     returns: [0 4 5 7]
     """
    
    name = name.lower()
    
    if len(name)>25:
        name = name[0:25]
        
    primenumbers = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
    
    n = len(name)
    
    s = 0.0
    
    for i in range(n):
        s += primenumbers[i]*ord(name[i])*2.0**(i+1)

    import scipy.io.matlab
    Data = scipy.io.matlab.loadmat('hash.mat',struct_as_record=True)
    x = Data['x']
    t = np.mod(s,x.shape[0])

    return np.sort(x[t,:])


if __name__ == "__main__":
    kohonen()

