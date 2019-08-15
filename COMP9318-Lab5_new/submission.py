def spectral_clustering(G): # do not change the heading of the function
    import numpy as np
    import networkx as nx
    from networkx import laplacian_matrix
    import nltk
    import random
    #print('hey')
    L = nx.laplacian_matrix(G).todense()
    nodes = []
    for i in G:
        nodes.append(i)
    #print(L)
    eigenvalue, eigenvector = np.linalg.eig(L)
    eigenvector = np.transpose(eigenvector)

    dim=len(eigenvalue)
    cluster_num = 2
    dictEigval = dict(zip(eigenvalue,range(0,dim)))
    keig = np.sort(eigenvalue)[1:cluster_num + 1]
    ix = [dictEigval[k] for k in keig]

    vec_transpose = np.transpose(eigenvector[ix][0])
    vec_transpose = np.array(vec_transpose)
    #print('add',vec_transpose)
    #print(vec_transpose.shape)
    kmeans_=nltk.cluster.kmeans.KMeansClusterer(num_means=2,distance=nltk.cluster.util.euclidean_distance, repeats=50, normalise=True,rng=random.Random(10),avoid_empty_clusters=True)
    clusters_index = kmeans_.cluster(vec_transpose, assign_clusters=True)
    clusters = [[],[]]
    index = 0
    for i in clusters_index:
        if i == 0:
            clusters[0].append(nodes[index])
        if i == 1:
            clusters[1].append(nodes[index])
        index += 1
    return eigenvector[ix], clusters

