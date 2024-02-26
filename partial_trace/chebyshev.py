import numpy as np

def get_cheby(a,b,n):
    """
    get n Chebyshev nodes on [a,b]
    """
    
    i = np.arange(n)
    x = np.cos((2*i+1)*np.pi/(2*(n)))
    
    return 0.5*(b-a)*x+0.5*(b+a)


def uniform_cheby(arr, n):
    """
    get n Chebyshev nodes on each interval defined by arr
    """

    nodes = np.array([])
    
    # Iterate over the array by 2 steps
    for i in range(len(arr)-1):
        a = arr[i]
        b = arr[i+1]
        interval_nodes = get_cheby(a, b, n)
        nodes = np.concatenate((nodes, interval_nodes))
    nodes = np.concatenate((nodes, np.array(arr)))
    nodes.sort()
    
    return nodes


def scaled_cheby(arr, total_nodes):
    """
    get Chebyshev nodes on each interval defined by arr so that each interval has a number of poitns roughly proportional to its width and the total number of nodse is approxiamtely total_nodes
    """

    nodes = np.array([])
    num_nodes = []
    total_length = arr[-1]-arr[0]
    # Iterate over the array by 2 steps
    for i in range(len(arr)-1):
        a = arr[i]
        b = arr[i+1]
        interval_length = b - a
        interval_nodes_n = int((interval_length/total_length) * total_nodes)
        interval_nodes = get_cheby(a, b, interval_nodes_n)
        nodes = np.concatenate((nodes, interval_nodes))
        num_nodes.append(len(interval_nodes))
    nodes = np.concatenate((nodes, np.array(arr)))
    nodes.sort()
    
    return num_nodes,nodes 