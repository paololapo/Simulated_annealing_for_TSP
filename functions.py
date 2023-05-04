import numpy as np
import osmnx as ox
import networkx as nx

# Defining the distance between two cities
def L2(cities, i, j):
    dx = cities["x"][i] - cities["x"][j]
    dy = cities["y"][i] - cities["y"][j]
    return (dx**2+dy**2)**0.5


# Building the distance matrix
def D_matrix(distance, cities):
    N = len(cities)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i][j] = distance(cities, i, j)
    return D


# Function to compute the total path length
def L(arr, D):
    L = 0
    for k in range(len(arr)-1):
        L += D[arr[k+1]][arr[k]]
    return L


# Function to transpose elements in the input array, leaving the first and last element unchanged
def transpose(arr):
    temp = arr.copy()
    idx = np.random.choice(np.arange(len(temp))[1:-1], size=2, replace=False)
    temp[idx[0]], temp[idx[1]] = temp[idx[1]], temp[idx[0]]
    return temp


# Simulated annealing algorithm
def SA(D, T0, T_f, alpha):
    # Create initial configuration by randomly shuffling the index vector
    idx = np.arange(D.shape[1])
    conf_i = np.random.permutation(idx)
    conf_i = np.append(conf_i, conf_i[0])
    L_i = L(conf_i, D)
    
    # Define arrays to store information about learning
    tested = np.array([]) 
    accepted = np.array([])
    best = np.array([L(conf_i, D)])
    
    # Initialize the temperature
    T = T0
    
    while(T > T_f):
        
        # Proposed configuration: transpose the current one
        conf_t = transpose(conf_i)

        # Compute total distances for the test configuration
        L_t = L(conf_t, D)

        # Save total distance of the tested configuration
        tested = np.append(tested, L_t)
        
        # Update configuration with probability given by Boltzmann distribution
        if L_t < L_i:
            # Update configuration
            conf_i = conf_t
            L_i = L_t
            
            # Save accepted and smallest total distances encountered
            accepted = np.append(accepted, L_t)
            best = np.append(best, np.min([L_t, best[-1]]))

        else:
            # Save smallest total distance encountered
            best = np.append(best, np.min([L_i, best[-1]]))
       
            r = np.random.uniform()
            
            if np.exp(-(L_t - L_i) / T) > r:
                # Update configuration
                conf_i = conf_t
                L_i = L_t
                
                # Save accepted total distance
                accepted = np.append(accepted, L_t)
            else:
                # Configuration unchanged
                # Save accepted total distance
                accepted = np.append(accepted, L_i)

        # Update temperature using cooling factor
        T = alpha * T
        
    return conf_i, accepted, best[1:], tested


# Define a faster version of SA to study the role of hyperparameters
# Returns: final configuration and its corresponding loss

def SA_light(D, T0, T_f, alpha):
    # Create initial configuration via random shuffling of the index vector
    idx = np.arange(D.shape[1])
    conf_i = np.random.permutation(idx)
    conf_i = np.append(conf_i, conf_i[0])
    L_i = L(conf_i, D)
    
    best = L(conf_i, D)
    
    # Initialize the temperature
    T = T0
    
    while(T>T_f):
        
        # Proposed configuration: transpose the current configuration
        conf_t = transpose(conf_i)

        # Compute total distances for the test configuration
        L_t = L(conf_t, D)
        
        # Update configuration with probability given by the Boltzmann distribution
        if(L_t < L_i):
            # Update configuration
            conf_i = conf_t
            L_i = L_t
            
            # Save the smallest distance encountered so far
            best = np.min([L_t, best])

        else:
            # Save the smallest distance encountered so far
            best = np.min([L_i, best])
       
            r = np.random.uniform()
            
            if(np.exp(-(L_t-L_i)/T) > r):
                # Update configuration
                conf_i = conf_t
                L_i = L_t
                         
        # Update temperature using the cooling factor
        T = alpha*T
        
    return conf_i, best


#Constant temperature learning algorithm
#Returns: ratio of tested cases to accepted cases
def acceptance_rate(D, T, it):
    # Create initial configuration through random shuffling of the index array
    idx = np.arange(D.shape[1])
    conf_i = np.random.permutation(idx)
    conf_i = np.append(conf_i, conf_i[0])
    L_i = L(conf_i, D)

    n_accepted = 0
    n_tested = 0

    for i in range(int(it)):
        
        # Proposed configuration: transpose the current one
        conf_t = transpose(conf_i)

        # Calculate the total distances for the test configuration
        L_t = L(conf_t, D)
        
        # Update the configuration with probability given by the Boltzmann distribution
        if(L_t < L_i):
            # Update the configuration
            conf_i = conf_t
            L_i = L_t
            
            n_accepted += 1
        
        else:
            r = np.random.uniform()
            
            if(np.exp(-(L_t-L_i)/T) > r):
                # Update the configuration
                conf_i = conf_t
                L_i = L_t
                
                n_accepted += 1
        
        n_tested += 1
    return n_accepted/n_tested


#SA version that saves every accepted configuration
#Returns: a list of lists
def SA_history(D, T0, T_f, alpha):
    # Create an initial configuration by randomly shuffling the index vector
    idx = np.arange(D.shape[1])
    conf_i = np.random.permutation(idx)
    conf_i = np.append(conf_i, conf_i[0])
    L_i = L(conf_i, D)

    # Define arrays that will contain information about the learning process
    history = []

    # Initialize the temperature
    T = T0

    while(T>T_f):
        
        # Proposed configuration: transpose the current one
        conf_t = transpose(conf_i)

        # Calculate the total distances for the test configuration
        L_t = L(conf_t, D)

        # Update the configuration with a probability given by the Boltzmann distribution
        if(L_t < L_i):
            # Update the configuration
            conf_i = conf_t
            L_i = L_t
            history.append(conf_t)
            
        else:
            r = np.random.uniform()
            
            if(np.exp(-(L_t-L_i)/T) > r):
                # Update the configuration
                conf_i = conf_t
                L_i = L_t
                history.append(conf_t)
            
            else:
                # Configuration unchanged
                history.append(conf_i)
                
        # Update the temperature using the cooling factor
        T = alpha*T
        
    return history