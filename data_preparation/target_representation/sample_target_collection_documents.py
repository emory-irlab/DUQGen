import json
import math
import faiss
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance
from collections import OrderedDict, defaultdict
from pyserini.search.lucene import LuceneSearcher


SEED_LIST = [35, 745, 10, 6534, 2]
N_DOCTEXT_FILTER = 300
N_ITER = 100
T = 1.0
VERBOSE = True


special_datasets_list = ["signal1m", "hotpotqa", "quora", "climate-fever"]


cosine_fn = lambda A, B: dot(A, B) / (norm(A)*norm(B))


def argmax(keys, f):
    return max(keys, key=f)


def get_seeditem_item_simscore_dict(seed_docid, docids, dataset_doc_emb_data_dict):
    seeditem_item_rel_dict = defaultdict()
    for docid in docids:
        seeditem_item_rel_dict[docid] = float( cosine_fn(dataset_doc_emb_data_dict[docid].numpy(), dataset_doc_emb_data_dict[seed_docid].numpy()) )
    return seeditem_item_rel_dict


def get_item_item_simscore_dict(docids, dataset_doc_emb_data_dict):
    item_item_cos_dict = defaultdict(dict)
    for i in docids:
        for j in docids:
            cos_sim = float( cosine_fn(dataset_doc_emb_data_dict[i].numpy(), dataset_doc_emb_data_dict[j].numpy()) )
            item_item_cos_dict[i][j] = cos_sim
    return item_item_cos_dict


def mmr_sorted(docs, lambda_, similarity1, similarity2):
    selected = OrderedDict()
    doc_score_dict = defaultdict()
    while set(selected) != docs:
        remaining = docs - set(selected)

        mmr_score = lambda x: lambda_*similarity1[x] - (1-lambda_)*max([similarity2[x][y] for y in set(selected)-{x}] or [0])
        mmr_score_values = np.array([mmr_score(doc) for doc in remaining])

        next_selected = argmax(remaining, mmr_score)
        next_selected_score = np.max(mmr_score_values)

        doc_score_dict[next_selected] = next_selected_score

        selected[next_selected] = len(selected)

    return selected, dict(doc_score_dict)


def passage_selection(dataset_doc_emb_data_dict, get_seeditem_item_simscore_dict, get_item_item_simscore_dict,
                      n_selection, seed_docid, docids, lambda_val=0.5):
    seeddoc_doc_rel_dict = get_seeditem_item_simscore_dict(seed_docid, docids, dataset_doc_emb_data_dict)

    doc_doc_rel_dict = get_item_item_simscore_dict(docids, dataset_doc_emb_data_dict)

    _, output_scores_dict = mmr_sorted(set(docids), lambda_val, seeddoc_doc_rel_dict, doc_doc_rel_dict)

    sort_orders = sorted(output_scores_dict.items(), key=lambda x: x[1], reverse=True)
    return sort_orders[:n_selection]


def get_doc_text(docid, searcher):
    try:
        doctext = searcher.doc(docid).raw()

        doctext_str = doctext.split('"text" : "')[-1].split('"metadata')[0].strip()
        if doctext_str[-2:] == '",':
            doctext_str = doctext_str.replace('",', '').strip()

        doctitle_str = doctext.split('"title" : "')[-1].split('"text')[0].strip()
        if doctitle_str[-2:] == '",':
            doctitle_str = doctitle_str.replace('",', '').strip()

        return doctitle_str + ' ' + doctext_str
    except AttributeError:
        return None


def main_run(dataset_name, collection_text_filepath, collection_embedding_filepath , save_sampled_documents_filepath,
             n_clusters=1000, n_train=1000, lambda_val=1.0):
    lexical_index_name = f'beir-v1.0.0-{dataset_name}.multifield' # pyserini multi-field index name

    ########################################################################
    #                   Load collection embedding
    ########################################################################
    # 1. Load collection documents and filter out noisy documents
    filtered_docid_doctext_dict = {}
    for line in open(collection_text_filepath):
        data = json.loads(line)
        docid = data['docid']
        doctext = data['doctext']

        if dataset_name == 'signal1m' and len(doctext) < 3:
            continue
        elif dataset_name == 'hotpotqa' and len(doctext) < 150:
            continue
        elif dataset_name == 'quora' and len(doctext) < 15:
            continue
        elif dataset_name == 'climate-fever' and len(doctext) < 3:
            continue

        elif dataset_name not in special_datasets_list and len(doctext) < N_DOCTEXT_FILTER:
            continue



        # if len(doctext) < N_DOCTEXT_FILTER:
        #     continue
        filtered_docid_doctext_dict[docid] = doctext

    # 2. Load document embedding
    dataset_doc_emb_data_dict = {k: v for k, v in torch.load(collection_embedding_filepath).items() if filtered_docid_doctext_dict.get(k) }
    dataset_doc_emb_data_np = np.array([v.numpy() for k, v in dataset_doc_emb_data_dict.items() ])
    n_corpus = len(dataset_doc_emb_data_np)

    target_docids = list(dataset_doc_emb_data_dict.keys())
    target_docids_embs = torch.stack( [dataset_doc_emb_data_dict[docid] for docid in target_docids] )
    print('... completed collection loading')

    ########################################################################
    #                   Apply clustering on collection
    ########################################################################
    # 3. Train and index clustering algorithm
    d = target_docids_embs.shape[1]
    kmeans = faiss.Kmeans(d, n_clusters, niter=N_ITER, verbose=VERBOSE, gpu=True, seed=420)
    kmeans.train(target_docids_embs)

    index = faiss.IndexFlatL2 (d)
    index.add ( target_docids_embs )
    D, I = index.search (kmeans.centroids, 1)

    # 4. Find docids closes to each cluster centroid
    centroid_nearest_docids_list = []
    for centroid_nearest_idxs, _ in zip(I, D):
        centroid_nearest_docid = target_docids[centroid_nearest_idxs[0]]
        centroid_nearest_docids_list.append( centroid_nearest_docid )

    D_inv, I_inv = kmeans.index.search(target_docids_embs, 1)

    # 5. Find cluster size (number of documents in each cluster)
    docid_clusteridx_dict = {}
    clusteridx_docids_dict = defaultdict(list)
    cluster_size_dict = defaultdict(int)
    for docid, cluster_idx in zip(target_docids, I_inv):
        docid_clusteridx_dict[docid] = cluster_idx[0]
        clusteridx_docids_dict[cluster_idx[0]].append( docid )
        cluster_size_dict[ cluster_idx[0] ] += 1

    cluster_size_dict_sorted = sorted(cluster_size_dict.items(), key=lambda x: x[1], reverse=False)
    print('... completed collection clustering')

    ########################################################################
    #                   Determine sample size for each cluster
    ########################################################################
    # 6. Find initial round of sample size for each cluster
    clusteridx_samplesize_dict = defaultdict(int)
    sample_cnt = 0
    for _, (cluster_idx, size) in enumerate(cluster_size_dict_sorted):
        sample_size = 1 + int(math.floor( (n_train - n_clusters) * (size / n_corpus) ))
        clusteridx_samplesize_dict[cluster_idx] = sample_size
        sample_cnt += sample_size

    # 7. Sample more from highly populated clusters
    n_remaining = n_train - sample_cnt
    if n_train != n_clusters:
        for _, (cluster_idx, size) in enumerate(cluster_size_dict_sorted[-n_remaining::]):
            clusteridx_samplesize_dict[cluster_idx] += 1
            sample_cnt += 1

    # 8. Derive cosine similarity (~distance) for each documents
    clusterids_docids_dist_cosdict = defaultdict(dict)
    for docid, cluster_idx in zip(target_docids, I_inv):
        cluster_centroid = kmeans.centroids[cluster_idx[0]]
        docemb = dataset_doc_emb_data_dict[docid].numpy()

        cos_distance = cosine_fn(docemb, cluster_centroid)
        clusterids_docids_dist_cosdict[cluster_idx[0]][docid] = float(cos_distance)

    cluster_idx_list = list(clusteridx_samplesize_dict.keys())
    random.shuffle(cluster_idx_list)

    ########################################################################
    #                   Sample documents from each cluster
    ########################################################################
    # 9. Probabilistic sampling based on document distance
    total_sample_size = 0
    cluster_idx_set = set()
    docids_set = set()
    all_sampled_docids_set = set()
    for cluster_idx in tqdm(cluster_idx_list, total=len(clusteridx_samplesize_dict)):
        sample_size = clusteridx_samplesize_dict[cluster_idx]
        total_sample_size += sample_size
        cluster_idx_set.add(cluster_idx)

        # (1) find docids belong to cluster
        curr_docids = clusteridx_docids_dict[cluster_idx]
        for docid in curr_docids:
            docids_set.add(docid)

        # (2) get cosine-similarity for each docid
        curr_docid_distances = [clusterids_docids_dist_cosdict[cluster_idx][docid] for docid in curr_docids]

        # (3) define probabilities based on distance
        prob_values = [e/T for e in curr_docid_distances]
        curr_docid_probs = np.exp(prob_values) / np.sum(np.exp(prob_values), axis=0)
        assert np.sum(curr_docid_probs) >= 0.99

        # (4) random sample with probabilities
        curr_selected_docids_pool = set()
        for seed in SEED_LIST:
            np.random.seed(seed)
            sampled_docids = np.random.choice(curr_docids, p=curr_docid_probs, size=sample_size, replace=False)
            for docid in sampled_docids:
                curr_selected_docids_pool.add(str(docid))

        # (5) apply MMR and pick top-sample_size documents
        closest_to_centroid_docid = centroid_nearest_docids_list[cluster_idx]
        final_sampled_docids_info = passage_selection(dataset_doc_emb_data_dict, get_seeditem_item_simscore_dict, get_item_item_simscore_dict,
                                                      n_selection=sample_size, seed_docid=closest_to_centroid_docid, docids=curr_selected_docids_pool,
                                                      lambda_val=lambda_val)
        final_sampled_docids = [e[0] for e in final_sampled_docids_info]

        all_sampled_docids_set = all_sampled_docids_set.union(set(final_sampled_docids))
    print('... completed document sampling')


    ########################################################################
    #                   Find document-text for the sampled documents
    ########################################################################
    # 10. Look-up Pyserini index to fetch the document-text
    searcher = LuceneSearcher.from_prebuilt_index(lexical_index_name)

    target_docid_doctext_list = []
    for docid in all_sampled_docids_set:
        doctext = get_doc_text( docid, searcher=searcher)
        if not doctext:
            continue
        target_docid_doctext_list.append( {'docid': docid, 'doctext': doctext} )
    print('... completed document text lookup')


    ########################################################################
    #                   Save sampled documents with text
    ########################################################################
    # 11. Save the documents sampled from collection
    with open(save_sampled_documents_filepath, 'w') as f:
        for _, line in enumerate(target_docid_doctext_list):
            json.dump(line, f)
            f.write('\n')
    print('... completed saving sampled documents')



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Document Sampling')
    parser.add_argument('--dataset_name', required=True, type=str,
                        help='dataset name to be specific')
    parser.add_argument('--collection_text_filepath', required=True, type=str,
                        help='file path to document collection text')
    parser.add_argument('--collection_embedding_filepath', required=True, type=str,
                        help='file path to document collection embedding')
    parser.add_argument('--save_sampled_documents_filepath', required=True, type=str,
                        help='file path to save output of the script: sampled documents')
    parser.add_argument('--n_clusters', type=int, default=1000, required=False,
                        help='number of clusters')
    parser.add_argument('--n_train', type=int, default=1000, required=False,
                        help='number of training examples')
    parser.add_argument('--lambda_val', type=float, default=1.0, required=False,
                        help='lambda value used in MMR diversify measure')
    args = parser.parse_args()


    main_run(args.dataset_name, args.collection_text_filepath, args.collection_embedding_filepath , args.save_sampled_documents_filepath,
             args.n_clusters, args.n_train, args.lambda_val)
