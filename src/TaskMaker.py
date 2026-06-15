import numpy as np

""" Create DATA (set) based on # datapoints (count), and coded specs for inputs & outputs"""
def makeDataset(count, dataspecs):

    """ TODO: Code specs for outputs beyond random """
   
    DATA = np.zeros((count,len(dataspecs)))

    for idx,i in enumerate(dataspecs):

        if np.isscalar(i): #if simply a number
            DATA[:,idx] = i
            
        elif type(i) == list:
            if i[0] == 'UNIFORM' or i[0] == 'UNI' or i[0] == 'U': #[min,max]
                DATA[:,idx] = np.random.rand(count)*(i[2]-i[1]) + i[1]
            elif i[0] == 'NORMAL' or i[0] == 'NORM' or i[0] == 'N': #[mean, std]
                DATA[:,idx] = np.random.randn(count)*i[2] + i[1]
            elif i[0] == 'DISCRETE' or i[0] == 'DISC' or i[0] == 'D': #[min, max, step], as in range
                DATA[:,idx] = np.random.choice(np.arange(*i[1:]), size=count)
        elif type(i) == str:
            raise ValueError("have to program in grid functionality!")

    
    return DATA
