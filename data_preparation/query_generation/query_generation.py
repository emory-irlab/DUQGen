import yaml
import json
import torch
import argparse
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def main_run(model_name_or_path, prompt_template_filepath, sampled_documents_filepath, save_generated_queries_filepath, cache_dir,
             max_input_seq_length=512, max_input_to_llm_seq_length=4096, max_new_tokens=64, temperature=1e-20, do_sample=False):
    # 1. Load LLM and tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, use_fast=True)
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16).to(device)


    # 2. Load prompt template
    templates = yaml.safe_load(open(prompt_template_filepath))
    input_prompt = templates['3-shot']['template']


    # 3. Load sampled documents
    docid_doctext_list = [json.loads(line) for line in open(sampled_documents_filepath)]


    # 4. Generate queries using LLM prompting
    with open(save_generated_queries_filepath, 'w') as f:
        for i in tqdm(range(0, len(docid_doctext_list))):
            batch_doc = docid_doctext_list[i]
            batch_docid = batch_doc['docid']
            batch_doctext = ' '.join( batch_doc['doctext'].split()[:max_input_seq_length] )
            batch_prompted_doctext = input_prompt.format(document=batch_doctext, query="").rstrip()

            input_tokens = tokenizer.batch_encode_plus([batch_prompted_doctext], return_tensors="pt", truncation=True, max_length=max_input_to_llm_seq_length)
            input_tokens = {k: v.to(device) for k, v in input_tokens.items()}
            input_tokens = {k: v[:, :max_input_to_llm_seq_length] for k, v in input_tokens.items()} # for a safety option to never throw exceptions while running


            batch_output = model.generate(**input_tokens, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample)
            batch_output_text = tokenizer.batch_decode(batch_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # some string cleanings for unexpected LLM generated texts
            postprocessed_generated_quries = batch_output_text.replace(batch_prompted_doctext, '').split('Example 5:')[0].strip()\
                                                .split('In each of these examples, the AI model is trained on a large corpus of text data,')[0].strip()\
                                                    .split('Answer:')[0].strip().split('\n')[0].strip().split('?')[0].strip()

            entry_dict = {'docid': batch_docid, 'doctext': batch_doctext, 'question': postprocessed_generated_quries}
            json.dump(entry_dict, f)
            f.write('\n')


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Query Generation')
    parser.add_argument('--model_name_or_path', required=True, type=str,
                        help='model name or path for LLM to generate query')
    parser.add_argument('--prompt_template_filepath', required=True, type=str,
                        help='prompt file with few-shot examples for each target domain')
    parser.add_argument('--sampled_documents_filepath', required=True, type=str,
                        help='sampled documents input to the query generation stage')
    parser.add_argument('--save_generated_queries_filepath', required=True, type=str,
                        help='file to save the generated queries for each dataset')
    parser.add_argument('--cache_dir', required=True, type=str,
                        help='cache dir to download model and tokenizer')
    parser.add_argument('--max_input_seq_length', type=int, default=512, required=False,
                        help='maximum sequence length for documents')
    parser.add_argument('--max_input_to_llm_seq_length', type=int, default=4096, required=False,
                        help='maximum input sequence length to parse into LLM')
    parser.add_argument('--max_new_tokens', type=int, default=64, required=False,
                        help='maximum new tokens to allow enough generated text for queries')
    parser.add_argument('--temperature', type=float, default=1e-20, required=False,
                        help='temperature value for LLM generation')
    args = parser.parse_args()


    main_run(args.model_name_or_path, args.prompt_template_filepath, args.sampled_documents_filepath, args.save_generated_queries_filepath, args.cache_dir,
             args.max_input_seq_length, args.max_input_to_llm_seq_length, args.max_new_tokens, args.temperature, do_sample=False)