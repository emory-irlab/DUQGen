import json
import torch
import faiss
import argparse
import numpy as np
import seaborn as sns
from numpy import dot
from tqdm import tqdm
from kneed import KneeLocator
from numpy.linalg import norm
from scipy.spatial import distance
from collections import defaultdict




CLUSTER_LIST = [5, 10, 50, 100, 500, 1000, 5000, 10000]
CLUSTER_SEED = 560
niter = 100
verbose = True
sns.set(rc={'figure.figsize':(15, 6)})



def find_optimum_k(sse_keys, sse_values, increment=0, decrement=0):
    kn = KneeLocator(x=sse_keys,
                 y=sse_values,
                 curve='convex',
                 S=0.01,
                 direction='decreasing')
    k = kn.knee + increment - decrement
    return k


def cluster_for_different_num_of_clusters(dataset_name, dataset_text_fn, dataset_emb_fn, save_sse_values):
    filtered_docid_doctext_dict = {}
    for line in open(dataset_text_fn):
        d = json.loads(line)
        docid = d['docid']
        doctext = d['doctext']

        if dataset_name == 'signal1m' and len(doctext) < 3:
            continue
        elif dataset_name == 'hotpotqa' and len(doctext) < 150:
            continue

        if len(doctext) < 300:
            continue
        filtered_docid_doctext_dict[docid] = doctext


    dataset_doc_emb_data_raw_dict = torch.load(dataset_emb_fn)
    dataset_doc_emb_data_dict = {k: v for k, v in dataset_doc_emb_data_raw_dict.items() if filtered_docid_doctext_dict.get(k) }
    dataset_doc_emb_data_np = np.array([v.numpy() for k, v in dataset_doc_emb_data_dict.items() ])
    N_CORPUS = len(dataset_doc_emb_data_np)


    target_docids = list(dataset_doc_emb_data_dict.keys())
    target_docids_embs = torch.stack( [dataset_doc_emb_data_dict[docid] for docid in target_docids] )


    euc_distance_fn = lambda x, y: distance.euclidean(x, y)
    cosine_fn = lambda A, B: dot(A, B) / (norm(A)*norm(B))


    d = target_docids_embs.shape[1]
    clusternum_SSE_dict = {}
    for n_clusters in CLUSTER_LIST:

        kmeans = faiss.Kmeans(d, n_clusters, niter=niter, verbose=verbose, gpu=True, seed=CLUSTER_SEED)
        kmeans.train(target_docids_embs)

        D_inv, I_inv = kmeans.index.search(target_docids_embs, 1)

        docid_clusteridx_dict = {}
        clusteridx_docids_dict = defaultdict(list)
        for docid, cluster_idx in zip(target_docids, I_inv):
            docid_clusteridx_dict[docid] = cluster_idx[0]
            clusteridx_docids_dict[cluster_idx[0]].append( docid )

        W_k_list = []
        for cluster_idx, docids_list in tqdm(clusteridx_docids_dict.items(), total=len(clusteridx_docids_dict)):
            centroid_emb = kmeans.centroids[cluster_idx]
            for docid in docids_list:
                doc_emb = dataset_doc_emb_data_dict[docid].numpy()

                cos_distance = 1 - cosine_fn(doc_emb, centroid_emb)
                # eucl_distance = euc_distance_fn(doc_emb, centroid_emb)
                W_k_list.append( cos_distance**2 )

        SSE_value = np.sum(W_k_list)
        print(f'..... SSE value for num of clusters {n_clusters} : {SSE_value}')

        clusternum_SSE_dict[n_clusters] = float(SSE_value)

        with open(save_sse_values, 'w') as f:
            json.dump(clusternum_SSE_dict, f)


    with open(save_sse_values, 'w') as f:
        json.dump(clusternum_SSE_dict, f)

    return clusternum_SSE_dict



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Document Encoding')    
    parser.add_argument('--dataset_name', required=True, type=str,
                        help='dataset name')
    parser.add_argument('--dataset_text_fn', required=True, type=str,
                        help='file contains document text for the dataset or corpus')
    parser.add_argument('--dataset_emb_fn', required=True, type=str,
                        help='file contains document embedding for the dataset or corpus')
    parser.add_argument('--save_sse_values', required=True, type=str,
                        help='save json file to store the sum of squard distances values for each number of cluster - clustering')
    args = parser.parse_args()


    clusternum_SSE_dict = cluster_for_different_num_of_clusters(args.dataset_name, args.dataset_text_fn, args.dataset_emb_fn, args.save_sse_values)

    sse_keys, sse_values = [], []
    for k, v in clusternum_SSE_dict.items():
        sse_keys.append(k)
        sse_values.append(v)

    optim_K = find_optimum_k(sse_keys, sse_values)

    print('Optimum K = ', optim_K)
