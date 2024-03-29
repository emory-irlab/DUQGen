import json
import torch
import argparse
from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, AutoModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def encode_text(model, max_seq_length, tokenizer, text):
    input = tokenizer.batch_encode_plus(
            text,
            truncation="longest_first",
            max_length = max_seq_length,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
            )
    input = { k: v.to(device) for k, v in input.items() }
    return model(**input), input['attention_mask']


def main_run(collection_data_filepath,  save_collection_embedding_filepath, cache_dir,
             batch_size=32, encoder_mode_name_or_path='facebook/contriever-msmarco', max_seq_length=512):
    # 1. Load collection documents
    docid_doctext_list = []
    for line in open(collection_data_filepath):
        data = json.loads(line)
        docid_doctext_list.append( {'docid': data['docid'], 'doctext': data['doctext']} )


    # 2. Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_mode_name_or_path, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(encoder_mode_name_or_path, cache_dir=cache_dir)
    model.to(device)
    print("... Encoder Model Loaded into GPU")


    text_encoder = partial(encode_text, model, max_seq_length, tokenizer)


    # 3. Encode document text
    dataset_doc_emb_dict = {}
    for i in tqdm(range(0, len(docid_doctext_list), batch_size)):
        batch_data = docid_doctext_list[i: i+batch_size]

        batch_q_text = [e['doctext'] for e in batch_data]
        batch_qid = [e['docid'] for e in batch_data]

        text_emb_output, attention_mask = text_encoder(batch_q_text)
        text_emb = mean_pooling(text_emb_output[0], attention_mask).detach().cpu()

        for qid, qemb in zip(batch_qid, text_emb):
            dataset_doc_emb_dict[str(qid)] = qemb


    # 4. Save document
    torch.save(dataset_doc_emb_dict, save_collection_embedding_filepath)



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Document Encoding')
    parser.add_argument('--collection_data_filepath', required=True, type=str,
                        help='file path to document collection text')
    parser.add_argument('--save_collection_embedding_filepath', required=True, type=str,
                        help='file path to save output of the script: document embedding')
    parser.add_argument('--cache_dir', required=True, type=str,
                        help='cache dir to download model and tokenizer')
    parser.add_argument('--batch_size', type=int, default=32, required=False,
                        help='batch size for inference')
    parser.add_argument('--encoder_mode_name_or_path', type=str, default='facebook/contriever-msmarco', required=False,
                        help='pre-trained encoder model name or path')
    parser.add_argument('--max_seq_length', type=int, default=512, required=False,
                        help='maximum sequence length to encode text')
    args = parser.parse_args()


    main_run(args.collection_data_filepath,  args.save_collection_embedding_filepath, args.cache_dir,
             args.batch_size, args.encoder_mode_name_or_path, args.max_seq_length)