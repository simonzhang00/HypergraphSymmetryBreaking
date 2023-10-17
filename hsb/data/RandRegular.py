import glob
import sys
import networkx as nx
import pandas as pd
import os
import random
import numpy as np

class RandRegular():
    def __init__(self, PATH, d,n, seed):
        G= nx.random_regular_graph(d,n, seed= seed)
        maxcliques= []
        maxcliques_lens= []
        maxcliques_times= []
        t= 0
        cliques= list(nx.find_cliques(G))
        np.random.seed(seed)
        splitmax= np.random.binomial(size=len(cliques), n=1, p=1.0)
        for c,flip in zip(cliques,splitmax):
            if flip and len(c)>=3:
                for v in c:
                    maxcliques_lens.append(len(c)-1)
                    for u in c:
                        if(not u==v):
                            maxcliques.append(u)
                maxcliques_times.append(t)
                t+= 1
            else:
                maxcliques_lens.append(len(c))
                for u in c:
                    maxcliques.append(u)
                maxcliques_times.append(t)
                t+= 1
        df_maxcliques = pd.DataFrame(maxcliques, columns = ["vertices"])
        df_maxcliques_lens = pd.DataFrame(maxcliques_lens, columns = ["lengths"])
        df_maxcliques_times = pd.DataFrame(maxcliques_times, columns=["times"])
        PATH= PATH + "rand-regular/"
        os.makedirs(PATH, exist_ok=True)

        df_maxcliques.to_csv(PATH+'rand-regular-simplices.txt', sep='\n', line_terminator='\n', index=False, header= False)
        df_maxcliques_lens.to_csv(PATH+'rand-regular-nverts.txt', sep='\n', line_terminator='\n', index=False, header= False)
        df_maxcliques_times.to_csv(PATH+'rand-regular-times.txt', sep='\n', line_terminator='\n', index=False, header= False)

if __name__ == '__main__':
    PATH= "./processed-data/"
    RandRegular(PATH, 5,1000, seed=0)

