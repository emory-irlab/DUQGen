import os
import json
import argparse
import ir_measures
import numpy as np
from tqdm import tqdm
from ir_measures import *
from pyserini import search
from collections import defaultdict
from pyserini.search.lucene import LuceneSearcher
# set your custom local cache directory to store pyserini downloads
# os.environ['PYSERINI_CACHE'] = "/local/scratch/guest/pyserini_cache"


METRICS_LIST = [nDCG@10, R@100]


def main_run(dataset_name, fields, save_bm25_results_filepath, save_testdata_filepath, save_qrels_filepath, topk=100):
    lexical_index_name = f'beir-v1.0.0-{dataset_name}.{fields}'
    topics_filename = f"beir-v1.0.0-{dataset_name}-test"
    qrels_filename = f'beir-v1.0.0-{dataset_name}-test'
    searcher = LuceneSearcher.from_prebuilt_index(lexical_index_name)


    topics_dict = search.get_topics(topics_filename)
    qrels = search.get_qrels(qrels_filename)
    qrels_formatted = {}
    for k, v in qrels.items():
        qrels_formatted[str(k)] = {str(k): int(v) for k, v in v.items()}
    # 1. Save qrels in the trec-eval format
    with open(save_qrels_filepath, 'w') as f:
        for qid, pairs in qrels.items():
            for docid, label in pairs.items():
                f.write(f"{qid} 0 {docid} {label}\n")


    # 2. Run BM25 search
    python_cmd = f"python -m pyserini.search.lucene \
        --index {lexical_index_name} \
        --topics {topics_filename} \
        --output {save_bm25_results_filepath} \
        --output-format trec \
        --batch 36 --threads 12 \
        --hits {topk} --bm25 --fields contents=1.0 title=1.0"
    os.system(python_cmd)


    # 3. Print the evaluation scores
    run = ir_measures.read_trec_run(save_bm25_results_filepath)
    res = ir_measures.calc_aggregate(METRICS_LIST, qrels_formatted, run)
    res = {str(k): v for k, v in res.items()}

    print(f"Evaluation results: \n\tnDCG@10: {res['nDCG@10']}\n\tR@100: {res['R@100']}")


    # 4. Load first-stage retrieval results and verify top-k depth
    results_dict = defaultdict(dict)
    for line in open(save_bm25_results_filepath):
        qid, _, docid, rank, score, _ = line.split(' ')
        results_dict[qid][docid] = float(score)

    n_not_topK = 0
    for k, v in results_dict.items():
        if len(v) >= 100:
            n_not_topK += 1
    print('Numeber of queries not returning top-k depth of documents : ', n_not_topK)


    # 5. Pool documents and look-up document text
    docid_set = set()
    n_doc_occurance = 0
    for qid, info in results_dict.items():
        sorted_info = sorted(info.items(), key=lambda x: x[1], reverse=True)
        for docid, docscore in sorted_info[:topk]:
            docid_set.add(docid)
            n_doc_occurance += 1

    docid_doctext_dict = {}
    n_missing_docs = 0
    for docid in tqdm(list(docid_set), total=len(docid_set)):
        try:
            doctext = searcher.doc(docid).raw()
        except AttributeError:
            n_missing_docs += 1
            continue

        # =============================================================================
        doctext_str = doctext.split('"text" : "')[-1].split('"metadata')[0].strip()
        if doctext_str[-2:] == '",':
            doctext_str = doctext_str.replace('",', '').strip()
        # =============================================================================

        doctitle_str = doctext.split('"title" : "')[-1].split('"text')[0].strip()
        if doctitle_str[-2:] == '",':
            doctitle_str = doctitle_str.replace('",', '').strip()

        # =============================================================================
        if fields == "multifield":
            docid_doctext_dict[docid] = doctitle_str + '. ' + doctext_str
        else:
            docid_doctext_dict[docid] = doctext_str

    assert len(docid_set) >= len(docid_doctext_dict)


    # 6. Prepare test data in a unified format
    test_data = []
    n_doc_occurances_list = []
    for qid, info in results_dict.items():
        try:
            qtext = topics_dict[qid]
        except KeyError:
            qtext = topics_dict[int(qid)]

        docs_info = []
        n_doc = 0
        sorted_info = sorted(info.items(), key=lambda x: x[1], reverse=True)

        for docid, docscore in sorted_info[:topk]:
            if not docid_doctext_dict.get(docid):
                continue
            doctext = docid_doctext_dict[docid]
            docs_info.append( [docid, doctext] )
            n_doc += 1

        n_doc_occurances_list.append( n_doc )

        test_data.append( {'qid': qid, 'qtext': qtext, 'passages': docs_info} )


    # 7. Save test data
    with open(save_testdata_filepath, 'w') as f:
        json.dump(test_data, f)


    print(f'Number of documents per query: minimum={np.min(n_doc_occurances_list)} || maximum={np.max(n_doc_occurances_list)}')


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Generate test reranking data of BM25')
    parser.add_argument('--dataset_name', required=True, type=str,
                        help='dataset name')
    parser.add_argument('--save_bm25_results_filepath', required=True, type=str,
                        help='file to save bm25 initial retrieval results')
    parser.add_argument('--save_testdata_filepath', required=True, type=str,
                        help='file to save test reranking data')
    parser.add_argument('--save_qrels_filepath', required=True, type=str,
                        help='file to save test qrel into trec-eval format')
    parser.add_argument('--fields', required=False, default='multifield', type=str,
                        help='whether single field or multifield option in pyserini to include text with title or not')
    parser.add_argument('--topk', type=int, default=100, required=False,
                        help='initial retrieval top-k depth')
    args = parser.parse_args()


    main_run(args.dataset_name, args.fields, args.save_bm25_results_filepath, args.save_testdata_filepath, args.save_qrels_filepath, args.topk)