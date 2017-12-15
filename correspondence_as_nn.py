import numpy as np
from time import time
import nibabel as nib
from dissimilarity.dissimilarity import compute_dissimilarity, dissimilarity
import joblib
from scipy.spatial import cKDTree

try:
    import annoy
    has_annoy = True
except ImportError:
    has_annoy = False
    print("spotify/annoy not available!")

try:
    import nmslib
    has_nmslib = True
except ImportError:
    has_nmslib = False
    print("nmslib not available!")


if __name__ == '__main__':
    
    np.random.seed(0)
    memoize = True
    
    filename_A = 'data/HCP3/derivatives/deterministic_tracking_FNAD/sub-124422/sub-124422_var-FNAD_tract.tck'
    filename_B = 'data/HCP3/derivatives/deterministic_tracking_FNAD/sub-627549/sub-627549_var-FNAD_tract.tck'

    filename_A_dissimilarity = filename_A[:-3] + 'dissimilarity'
    filename_B_dissimilarity = filename_B[:-3] + 'dissimilarity'

    use_kdtree = True
    use_annoy = True
    use_nmslib = True

    limit_kdtree_queries = 10000


    print("Attempting to retrieve %s" % filename_A_dissimilarity)
    try:
        t0 = time()
        tmp = joblib.load(filename_A_dissimilarity)
        print("%s sec." % (time() - t0))
        dissimilarity_matrix_A = tmp['dissimilarity_matrix']
        prototype_idx_A = tmp['prototype_idx']
        prototype_A = tmp['prototype']
    except IOError:
        print("Not available!")
        print("Loading %s" % filename_A)
        t0 = time()
        data_A = nib.streamlines.load(filename_A)
        print("%s sec." % (time() - t0))
        streamlines_A = data_A.streamlines

        print("Computing the dissimilarity representation")
        t0 = time()
        dissimilarity_matrix_A, prototype_idx_A = compute_dissimilarity(streamlines_A,
                                                                    n_jobs=1,
                                                                    verbose=True)
        prototype_A = streamlines_A[prototype_idx_A]
        print("%s sec." % (time() - t0))
        if memoize:
            print("Saving %s" % filename_A_dissimilarity)
            t0 = time()
            joblib.dump({'dissimilarity_matrix': dissimilarity_matrix_A,
                         'prototype_idx': prototype_idx_A,
                         'prototype': streamlines_A[prototype_idx_A]},
                        filename_A_dissimilarity)
            print("%s sec." % (time() - t0))


    print("Attempting to retrieve %s" % filename_B_dissimilarity)
    try:
        t0 = time()
        tmp = joblib.load(filename_B_dissimilarity)
        print("%s sec." % (time() - t0))
        dissimilarity_matrix_B = tmp['dissimilarity_matrix']
        prototype_idx_B = tmp['prototype_idx']
        prototype_B = tmp['prototype']
    except IOError:
        print("Not available!")
        print("Loading %s" % filename_B)
        t0 = time()
        data_B = nib.streamlines.load(filename_B)
        print("%s sec." % (time() - t0))
        streamlines_B = data_B.streamlines

        print("Computing the dissimilarity representation with prototypes of A!")
        t0 = time()
        dissimilarity_matrix_B = dissimilarity(streamlines_B,
                                               prototype_A,
                                               n_jobs=1,
                                               verbose=True)
        print("%s sec." % (time() - t0))
        if memoize:
            print("Saving %s" % filename_B_dissimilarity)
            t0 = time()
            joblib.dump({'dissimilarity_matrix': dissimilarity_matrix_B,
                         'prototype_idx': prototype_idx_A,
                         'prototype': prototype_A},
                        filename_B_dissimilarity)
            print("%s sec." % (time() - t0))



    if use_kdtree:
        print("Building KDTree of the (approximate) streamlines of A.")
        t0 = time()
        kdt = cKDTree(dissimilarity_matrix_A)
        print("%s sec." % (time() - t0))

        print("Computing the (exact) nearest neighbor for %s (approximate) streamlines of B with scipy.cKDTree" % limit_kdtree_queries)
        t0 = time()
        distances_kdtree, correspondence_kdtree = kdt.query(dissimilarity_matrix_B[:limit_kdtree_queries])
        print("%s sec." % (time() - t0))
        

    if use_annoy and has_annoy:
        print("Computing the (approximate) nearest neighbor for each (approximate) streamline of B")
        print("Building Annoy index for large-scale (approximate) nearest neighbor.")
        n_trees = 30
        search_k = 60
        t0 = time()
        index = annoy.AnnoyIndex(dissimilarity_matrix_A.shape[1], metric='euclidean')
        for i, v in enumerate(dissimilarity_matrix_A):
            index.add_item(i, v)

        index.build(n_trees=n_trees)
        print("%s sec." % (time() - t0))

        print("Querying the (approximate) nearest neighbor of each (approximate) streamline of B")
        t0 = time()
        correspondence_annoy = np.zeros(dissimilarity_matrix_A.shape[0])
        for i, v in enumerate(dissimilarity_matrix_B):
            correspondence_annoy[i] = index.get_nns_by_vector(v, 1, search_k=search_k)[0]

        if use_kdtree:
            print("%s sec." % (time() - t0))
            print("Annoy accuracy: %s" % np.mean(correspondence_annoy[:limit_kdtree_queries] == correspondence_kdtree))


    if use_nmslib and has_nmslib:
        print("Computing the (approximate) nearest neighbor for each (approximate) streamline of B with nmslib")
        print("Building nmslib index for large-scale (approximate) nearest neighbor.")
        t0 = time()
        index_nmslib = nmslib.init(method='hnsw', space='l2')
        index_nmslib.addDataPointBatch(dissimilarity_matrix_A)
        index_nmslib.createIndex({'post': 2}, print_progress=True)
        print("%s sec." % (time() - t0))

        print("Querying the (approximate) nearest neighbor of each (approximate) streamline of B")
        t0 = time()
        correspondence_nmslib = index_nmslib.knnQueryBatch(dissimilarity_matrix_B, k=1, num_threads=4)
        correspondence_nmslib = np.array([i[0][0] for i in correspondence_nmslib])
        if use_kdtree:
            print("%s sec." % (time() - t0))
            print("Annoy accuracy: %s" % np.mean(correspondence_nmslib[:limit_kdtree_queries] == correspondence_kdtree))
