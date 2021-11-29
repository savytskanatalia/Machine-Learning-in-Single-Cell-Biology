# Final code piece for the script
import argparse
import argparse
from scipy import io, stats
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
from scipy.sparse import linalg, eye
from sklearn.manifold import LocallyLinearEmbedding
from umap import UMAP
import umap.plot




parser = argparse.ArgumentParser(description='Assignment 2, Savytska, Sarieva')
parser.add_argument('input_path', type=str,
                    help='Full input path, where the count and meta matrices are stored; no default')
parser.add_argument('n_neighbours', type=int,
                    help='n_neighbours for LLE')
parser.add_argument('--out_dim', type=int,default=2,
                    help='out_dim for LLE, default 2')

args = parser.parse_args()



# Final code for 2A




# Function that will obtain unique sorted (in ascencion) list of values from a series, assign ranks to them and then sustitute values with appropriate ranks in original series
# Needed for Quantile normalization (gene expression ranking and obtaining rank mx from this data)
def ranking_df(myseries):
  lst = list(set(myseries.tolist()))
  dic = dict(zip(lst, list(range(0, len(lst)))))
  lst2 = [dic[k] for k in list(myseries)]
  return(lst2)

# Final transformation - translating series (so sample column with all gene expression ranks) into a normalized value series
# With dictionary comprised of key-ranks of values-rowmeans obtained in final step of the quantile normalization
# Second input variable - the dictionary of rank-rowmean values is hardcoded. Sorry. Just make sure to run in last




def task_02(input_path):
    """
    This function performs quantile normalization.
    The following steps are needed:
    1) transpose our data to have genes in rows and samples in columns;
    2) Then we create a new dataframe with the gene ranks for each of the values in columns == "For each column determine a rank from lowest to highest and assign number 0-N" (0 instead of 1 because python?)
    3) Then we sort each column separately (so 'unlinked' way, we don`t rearrange the rest of the columns when one gets sorted) from lowest to highest
    4) Then we find the mean for each row to determine the ranks values. So ranks from first step will be our key and values will be mean values > which we will need to sort themselves by ranks. So whatever mean values will be the lowest, it gets the rank 0 and now corresponds to it
    5) Then take the ranking order and substitute in new values. So in the ranking table we match keys to new values. That's it. That's the correction.
    Quantile normalization of gene expression however has different flavors to it such as quantile normalization by classes, batches, both classes and batches, and finally with weighted data Reference: https://www.nature.com/articles/s41598-020-72664-6
    We are asked to implement the simplest one, so this will be the one we implement
    """

    # load count mx
    cnt_mx = pd.read_csv(input_path+'expression_data_2.txt',sep="\t")   
    # drop the rowname numbers
    cnt_mx = cnt_mx.drop(["Unnamed: 0"], axis=1)
    # transpose so that genes become rows and samples/cells become columns
    cnt_mx_t = cnt_mx.T
    # get a matrix which records ranks for each of the values in columns (ranks in ascencion)
    rank_mx = cnt_mx_t.apply(ranking_df, axis=0)
    # now sort each column independently in ascending order
    srt_mx = cnt_mx_t.apply(lambda x: x.sort_values().values)
    # get rowmeans for the sorted mx
    r_lst = list(set(list(srt_mx.mean(axis=1))))
    # get the dictionary, which represents RANKS for ROWMEANS this time
    r_dic = dict(zip(list(range(0, len(r_lst))),r_lst))
    def fin_norm(myseries):
      lst2 = [r_dic[k] for k in list(myseries)]
      return(lst2)
    # Translate the values in the rank matrix with the rowmeans rank dictionary
    # Et voila! Simple (naive) quantile normalization is finished for this cnt mx
    corrected_data = rank_mx.apply(fin_norm,axis=0).T


    # Visualize the results
    # Load meta data
    meta_mx = pd.read_csv(input_path+'metadata_2.txt',sep="\t")    

    
    np.random.seed(25)
    mapper = umap.UMAP(n_neighbors=20,
        min_dist=0.1).fit(corrected_data)
    np.random.seed(25)
    mapper2 = umap.UMAP(n_neighbors=20,
        min_dist=0.1).fit(cnt_mx)
    fig = plt.figure()
    umap.plot.points(mapper, labels=meta_mx.iloc[:,1]).set_title("Normalized, Batch")
    plt.savefig('QuantNorm_Batch.png')
    umap.plot.points(mapper, labels=meta_mx.iloc[:,2]).set_title("Normalized, Cell Types")
    plt.savefig('QuantNorm_CellType.png')
    umap.plot.points(mapper2, labels=meta_mx.iloc[:,1])
    plt.savefig('Raw_Batch.png')
    umap.plot.points(mapper2, labels=meta_mx.iloc[:,2])
    plt.savefig('Raw_CellType.png')
    fig.show()
    return corrected_data

  




def locally_linear_embedding(X, n_neighbors, 
                             out_dim=2):

    from sklearn.neighbors import NearestNeighbors
    from sklearn.manifold._locally_linear import barycenter_kneighbors_graph

    # Calculate weights
    W = barycenter_kneighbors_graph(X, n_neighbors=n_neighbors) 
    
    # M = (I-W)' (I-W) (see equation 3 in the paper)
    A = eye(*W.shape, format=W.format) - W
    M = (A.T).dot(A).toarray()

    eigen_values, eigen_vectors = np.linalg.eig(M)

    # Skip the smallest eigenvalue and its eigenvector
    index = np.argsort(eigen_values)[1:] 
    return eigen_vectors[:, index][:,:out_dim]
  
def task_01(input_path,n_neighbours,out_dim=2) -> np.array:
    """
    This function performs a locally-linear embedding.
    """
    count_mtx = np.loadtxt(input_path+'expression_data_1.txt')
 
    with open(input_path+"metadata_1.txt", "r") as cl_names:
     names = [line.strip(' \n') for line in cl_names.readlines()]
 
    names_arr = np.array(names)

    print("Computing LLE embedding")
    n_neighbors, out_dim = 30, 2
    X_r = locally_linear_embedding(count_mtx, n_neighbors)

    fig, ax = plt.subplots(figsize=(5, 5))
    for i, g in enumerate(np.unique(names_arr)):
        ix = np.where(names_arr == g)
        ax.scatter(X_r[ix,0], X_r[ix,1], label = g, color=plt.get_cmap('Spectral')(i*0.12), s = 25)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    fit = UMAP(n_neighbors=30,
            min_dist=0.1)
    u = fit.fit_transform(count_mtx)

    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize=(10, 5))
    for i, g in enumerate(np.unique(names_arr)):
        ix = np.where(names_arr == g)
        ax2.scatter(X_r[ix,0], X_r[ix,1], label = g, color=plt.get_cmap('Spectral')(i*0.12), s = 25)
    umap.plot.points(fit, labels=names_arr,ax = ax1)
    ax1.get_legend().remove()
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    plt.savefig('LLE_UMAP_compare.png')
    ### INSERT YOUR CODE HERE ###

    return X_r


def main():

    task_01_solution = task_01(args.input_path,args.n_neighbours,args.out_dim)
    task_02_solution = task_02(args.input_path)
    np.savetxt('task_01_solution.csv', task_01_solution, delimiter=',')
    np.savetxt('task_02_solution.csv', task_02_solution, delimiter=',')


if __name__ == "__main__":

    main()
