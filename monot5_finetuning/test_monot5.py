import os
import sys
import json
import torch
import argparse
import ir_measures
import pandas as pd
import pyterrier as pt
from ir_measures import *
from collections import defaultdict
from torch.nn import functional as F
from pyterrier.model import add_ranks
from pyterrier.transformer import TransformerBase
from transformers import T5Tokenizer, T5ForConditionalGeneration


METRICS_LIST = [nDCG@10, R@100]
pt.init()


# Most of the codes were extracted from Pyterrier codebase
class MonoT5ReRanker(TransformerBase):
    def __init__(self,
                tok_model='t5-base',
                model="castorini/monot5-3b-msmarco-10k",
                cache_dir=None,
                batch_size=4,
                text_field='text',
                verbose=True):
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(tok_model, cache_dir=cache_dir)
        self.model_name = model
        self.model = T5ForConditionalGeneration.from_pretrained(model, cache_dir=cache_dir)
        self.model.to(self.device)
        self.model.eval()
        self.text_field = text_field
        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]

    def __str__(self):
        return f"MonoT5({self.model_name})"

    def transform(self, run):
        scores = []
        queries, texts = run['query'], run[self.text_field]
        it = range(0, len(queries), self.batch_size)
        prompts = self.tokenizer.batch_encode_plus([f'Relevant:' for _ in range(self.batch_size)], return_tensors='pt', padding='longest')
        max_vlen = self.model.config.n_positions - prompts['input_ids'].shape[1]
        if self.verbose:
            it = pt.tqdm(it, desc='monoT5', unit='batches')
        for start_idx in it:
            rng = slice(start_idx, start_idx+self.batch_size) # same as start_idx:start_idx+self.batch_size
            enc = self.tokenizer.batch_encode_plus([f'Query: {q} Document: {d}' for q, d in zip(queries[rng], texts[rng])], return_tensors='pt', padding='longest')
            for key, enc_value in list(enc.items()):
                enc_value = enc_value[:, :-1] # chop off end of sequence token-- this will be added with the prompt
                enc_value = enc_value[:, :max_vlen] # truncate any tokens that will not fit once the prompt is added
                enc[key] = torch.cat([enc_value, prompts[key][:enc_value.shape[0]]], dim=1) # add in the prompt to the end
            enc['decoder_input_ids'] = torch.full(
                (len(queries[rng]), 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                result = self.model(**enc).logits
            result = result[:, 0, (self.REL, self.NREL)]
            scores += F.log_softmax(result, dim=1)[:, 0].cpu().detach().tolist()
        run = run.drop(columns=['score', 'rank'], errors='ignore').assign(score=scores)
        run = add_ranks(run)
        return run


def test(model_name_or_checkpoint_path, cache_dir, batch_size, save_predictions_fn, test_filename_path, qrle_filename):
    # 1. Load and prepare test data
    test_data = json.load(open(test_filename_path))

    test_dict = {'qid': [], 'query': [], 'docno': [], 'text': []}
    for e in test_data:
        qid = e['qid']
        qtext = e['qtext']['title']
        for pass_info in e['passages']:
            docid = pass_info[0]
            doctext = pass_info[1]
            test_dict['qid'].append( qid )
            test_dict['query'].append( qtext )
            test_dict['docno'].append( docid )
            test_dict['text'].append( doctext )

    test_df = pd.DataFrame(test_dict)


    # 2. Load MonoT5 model
    monot5_model = MonoT5ReRanker(tok_model=model_name_or_checkpoint_path, model=model_name_or_checkpoint_path, cache_dir=cache_dir, batch_size=batch_size)


    # 3. Test inference
    output_df = monot5_model.transform(test_df)


    # 4. Reformat test results and save
    output_scores_dict = defaultdict(dict)
    for _, row in output_df.iterrows():
        qid = row['qid']
        docid = row['docno']
        score = row['score']
        output_scores_dict[qid][docid] = score

    with open(save_predictions_fn, 'w') as f:
        json.dump(output_scores_dict, f)


    # 5. Print re-ranking evaluation metrics
    qrels = list(ir_measures.read_trec_qrels(qrle_filename))
    res = ir_measures.calc_aggregate(METRICS_LIST, qrels, output_scores_dict)
    res = {str(k): v for k, v in res.items()}

    print(f"Evaluation results: \n\tnDCG@10: {res['nDCG@10']}\n\tR@100: {res['R@100']}")



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Train MonoT5-3B')
    parser.add_argument('--model_name_or_checkpoint_path', required=True, type=str,
                        help='directory where saved trained model')
    parser.add_argument('--save_predictions_fn', required=True, type=str,
                        help='file to save predictions output')
    parser.add_argument('--test_filename_path', required=True, type=str,
                        help='input file with test reranking data')
    parser.add_argument('--qrle_filename', required=True, type=str,
                        help='output directory to save the trained model')
    parser.add_argument('--cache_dir', required=True, type=str,
                        help='cache dir to download model and tokenizer')
    parser.add_argument('--batch_size', type=int, default=32, required=False,
                        help='batch size for inference')
    args = parser.parse_args()


    test(args.model_name_or_checkpoint_path, args.cache_dir, args.batch_size, args.save_predictions_fn, args.test_filename_path, args.qrle_filename)