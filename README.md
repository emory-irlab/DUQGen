# 🦆 DUQGen

![](https://img.shields.io/badge/PRs-welcome-brightgreen)
<img src="https://img.shields.io/badge/Version-1.0-lightblue.svg" alt="Version">
[![arXiv](https://img.shields.io/badge/arXiv-2311.11226-pink.svg)](https://arxiv.org/abs/2404.02489)
![Python version](https://img.shields.io/badge/lang-python-important)
![License: Apache](https://img.shields.io/badge/License-Apache2.0-yellow.svg)

This is the code for our paper -- [DUQGen: Effective Unsupervised Domain Adaptation of Neural Rankers by Diversifying Synthetic Query Generation (NAACL 2024)](https://arxiv.org/abs/2404.02489).

# How is DUQGen different from Prior Works ?

We provide an effective solution to unsupervised domain-adaptation for ranking models. Existing line of works proposed multiple synthetic data generation frameworks (InPars, DocGen-RL, and Promptagator) to generate target examples to fine-tune a pre-trained ranker (pre-trained on MS-MARCO data). However, often the fine-tuned performance drops below zero-shot performance and the data generation process is heavily exhaustive and intensive. Therefore, we propose **DUQGen**, a cost-effective solution that can improve consistently over the zero-shot baselines and substancially improve over the existing baselines in most cases.




# How to run

**DUQGen** framework consists of 2 stages: data augmentation (generation) and fine-tuning.

## Data Generation
### Step-1: Encoding target corpus text
Prepare any target corpus documents in the below jsonl file.

```
{'docid: <document-id-1>, 'doctext': <document-text-1>}
{'docid: <document-id-2>, 'doctext': <document-text-2>}
...
```

Then run the below command with the script found in `data_preparation/target_representation`.

```
python generate_document_embedding.py \
    --collection_data_filepath <path_to_document_collection_file.jsonl> \
    --save_collection_embedding_filepath <path_to_save_document_embeddings_file.pt> \
    --cache_dir <path_to_download_models_cache_directory>
```

### Step-2: Target corpus document sampling

Run the below command with the script found in `data_preparation/target_representation`.

```
python sample_target_collection_documents.py \
    --dataset_name <dataset-name> \
    --collection_text_filepath <path_to_document_collection_file.jsonl> \
    --collection_embedding_filepath <path_to_document_embeddings_file.pt> \
    --save_sampled_documents_filepath <path_to_save_sampled_documents_file.jsonl>
```

### Step-3: Query generation

Run the below command with the script found in `data_preparation/query_generation`. We used [**LLAMA2-7B-chat**](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) found in HuggingFace for our query generation task. All BEIR target dataset prompt templates can be found in `data_preparation/prompt_templates` folder.


```
python query_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --prompt_template_filepath ../prompt_templates/template_<dataset>.yaml \
    --sampled_documents_filepath <path_to_sampled_documents_file.jsonl> \
    --save_generated_queries_filepath <path_to_save_generated_queries.jsonl> \
    --cache_dir <path_to_download_models_cache_directory>
```

### Step-4: Generate hard negative pairs

Run the below command with the script found in `data_preparation/hardnegative_mining`. We used contriever to return top-100 initial rank list and picked bottom-x (x=4) as the hard negatives.

```
python train_data_generation.py \
    --dataset_name <dataset-name> \
    --generated_queries_filepath <path_to_generated_queries.jsonl> \
    --save_traindata_filepath <path_to_save_generated_traindata.jsonl>
```


## Fine-tuning

We fine-tuned both MonoT5-3B and ColBERT, namedly **DUQGen-reranker** and **DUQGen-retriever**. But our approach will work for any ranking model.


### Step-5: Train a neural ranker
#### Train MonoT5-3B

Run the below command with the script found in `monot5_finetuning`.

```
python train_monot5.py \
    --base_modename_or_path castorini/monot5-3b-msmarco-10k \
    --train_data_filepath <input_training_data_file_path.jsonl> \
    --save_model_path <directory_to_save_trained_model> \
    --cache_dir <path_to_download_models_cache_directory>
```


#### Train ColBERT

Run the below command with the bash script found in `colbert_finetuning`. All the variables to change are descripted in the bash file itself. The bash script was developed to run on a SLURM system, but a simple bash call can run it.

```
sbatch train_colbert.sh
```


### Step-6: Evaluate the trained neural ranker

#### Prepare Test Data

We have to generate or format test data in order to do the re-ranking and dense retrieval. Please refer [https://github.com/thakur-nandan/beir-ColBERT](https://github.com/thakur-nandan/beir-ColBERT) to prepare test data for ColBERT dense retriever. To generate top-100 BM25 re-ranking data, please run the below command with the script found in `data_preparation/testdata_preparation`.

```
python generate_bm25_reranking_data.py \
    --dataset_name <dataset-name> \
    --save_bm25_results_filepath <file_to_save_bm25_results.txt> \
    --save_testdata_filepath <file_to_save_test_data.json> \
    --save_qrels_filepath <file_to_save_qrel_data_in_treceval_format.txt>
```


#### Evaluate **DUQGen-reranker** (MonoT5-3B) performance

Run the below command with the script found in `monot5_finetuning`.

```
python test_monot5.py \
    --model_name_or_checkpoint_path <path_directory_to_saved_model/checkpoint-*> \
    --save_predictions_fn <file_to_save_predictions.json> \
    --test_filename_path <input_file_having_test_data.json> \
    --qrle_filename <qrel_file_saved_in_treceval_format.txt> \
    --cache_dir <path_to_download_models_cache_directory>
```


#### Evaluate **DUQGen-retriever** (ColBERT) performance

Run the below command with the bash script found in `colbert_finetuning`. All the variables to change are descripted in the bash file itself. The bash script was developed to run on a SLURM system, but a simple bash call can run it.

```
sbatch test_colbert.sh
```


# Models
All the models are available on HuggingFace at this [link](https://huggingface.co/cramraj8).


# Citation

To cite 🦆 **DUQGen** in your work,

```bibtex
@inproceedings{chandradevan-etal-2024-duqgen,
    title = "{DUQG}en: Effective Unsupervised Domain Adaptation of Neural Rankers by Diversifying Synthetic Query Generation",
    author = "Chandradevan, Ramraj  and
      Dhole, Kaustubh  and
      Agichtein, Eugene",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.413",
    pages = "7430--7444",
}
```


# Contact
1. Ramraj Chandradevan (rchan31@emory.edu)
2. Kaustubh Dhole (kdhole@emory.edu)
3. Eugene Agichtein (yagicht@emory.edu)

