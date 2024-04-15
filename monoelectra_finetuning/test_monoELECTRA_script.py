import os
import sys
import json
import torch
import pickle
import logging
import datasets
import argparse
import warnings
import ir_measures
import transformers
import torch.nn as nn
from ir_measures import *
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
    AutoModel,
    Trainer)
from tqdm.auto import tqdm
from typing import List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from transformers import DataCollatorWithPadding
from data_args import ModelArguments, DataTrainingArguments, IRTrainingArguments
device = torch.device('cuda')
torch.manual_seed(123)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


METRICS_LIST = [nDCG@10, R@100]


class IRModel(nn.Module):
    def __init__(self, base_model_name, cache_dir=None, hidden_size=768):
        super(IRModel, self).__init__()
        self.lm_model = AutoModel.from_pretrained(base_model_name, cache_dir=cache_dir)
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(self.hidden_size, 1)

        self.loss_mse_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        output = self.lm_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = output.last_hidden_state

        logits = torch.mean(last_hidden_state, dim=1)

        score = self.linear1(logits).squeeze(-1)

        loss = self.loss_mse_fn(score, labels)

        return {'loss': loss, 'logits': score}


    @classmethod
    def from_pretrained(cls, base_model_name, cache_dir, saved_path):
        model = cls(base_model_name, cache_dir)
        if os.path.exists(os.path.join(saved_path, 'pytorch_model.bin')):
            print('>>> Loaded pytorch_model.bin from local files')
            model_dict = torch.load(os.path.join(saved_path, 'pytorch_model.bin'), map_location="cpu")
            model.load_state_dict(model_dict, strict=False)
        return model


@dataclass
class IRCollator(DataCollatorWithPadding):
    max_seq_length: int =180
    eval_mode: bool = False

    def __post_init__(self):
        pass

    def __call__(self, examples: List[Tuple[str, str]]) -> None:

        if not self.eval_mode:
            query_text_list = []
            passage_text_list = []
            labels_list = []
            for example in examples:
                query_text_list.append( example[0] )
                passage_text_list.append( example[1] )
                labels_list.append( example[2] )
        else:
            qid_list = []
            query_text_list = []
            passageid_list = []
            passage_text_list = []
            labels_list = []
            for example in examples:
                qid_list.append(example[0])
                query_text_list.append(example[1])
                passageid_list.append(example[2])
                passage_text_list.append(example[3])
                labels_list.append( 1 )

        text_encoded = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=list(zip(query_text_list, passage_text_list)),
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
            max_length=self.max_seq_length
            )

        batch = {
            "input_ids": text_encoded['input_ids'],
            "attention_mask": text_encoded['attention_mask'],
            "token_type_ids": text_encoded['token_type_ids'],
            "labels": torch.tensor(labels_list, dtype=torch.float)
        }
        return batch


class IRTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, args, tokenizer=None, max_seq_length=128, filenames_path_list=None):
        self.data_dir = data_dir
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.filenames_path_list = [[filenames_path_list, 100000000]] if type(filenames_path_list) == str else filenames_path_list
        if self.filenames_path_list:
            self.read_input_train_data()
        else:
            logger.error('Error in provided filenames_path_list')


    def read_input_train_data(self):
        self.data_train_list = []
        num_samples = 0
        for train_filename, num_examples in self.filenames_path_list:

            train_list = open(train_filename).readlines()

            for line in train_list[:num_examples]:
                data = json.loads(line)

                qtext, doctext, label = data

                self.data_train_list.append( [qtext, doctext, label] )
                num_samples += 1


        assert len(self.data_train_list) == num_samples, "Mismatch in loaded data"
        logger.info(f"Read {len(self.data_train_list)} samples from list of files")
        print(f"Read {len(self.data_train_list)} samples from list of files")

    def __len__(self):
        return len(self.data_train_list)

    def __getitem__(self, idx):
        index = idx % self.__len__()
        return self.data_train_list[index]


class IRTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, args, tokenizer=None, max_seq_length=128, filenames_path_list=None):
        self.data_dir = data_dir
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.filenames_path_list = [[filenames_path_list, 100000000]] if type(filenames_path_list) == str else filenames_path_list
        self.read_input_test_data()

    def read_json_data(self, filename):
        return json.load(open(filename))

    def read_input_test_data(self):
        self.data_test_list = []
        num_samples = 0
        for test_filename_path, num_examples in self.filenames_path_list:
            test_dict = self.read_json_data(test_filename_path)

            for line in test_dict[:num_examples]:
                qid = line['qid']
                qtext = line['qtext']['title']
                for pinfo in line['passages']:
                    self.data_test_list.append( [qid, qtext, pinfo[0], pinfo[1]] )
                    num_samples += 1

        assert len(self.data_test_list) == num_samples, "Mismatch in loaded test data"
        logger.info(f"Read {len(self.data_test_list)} samples from list of test files")
        print(f"Read {len(self.data_test_list)} samples from list of test files")

    def __len__(self):
        return len(self.data_test_list)

    def __getitem__(self, idx):
        index = idx % self.__len__()
        return self.data_test_list[index]


def main_test(model_args, data_args, training_args):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    # ============================================================================================================================================
    #                                       1. Setup the loadings
    # ============================================================================================================================================
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_type, cache_dir=model_args.cache_dir)
    model = IRModel.from_pretrained(base_model_name=model_args.model_type, cache_dir=model_args.cache_dir,
                                    saved_path=model_args.model_name_or_path)
    model.to(device)
    print("Loaded base model from : ", model_args.model_name_or_path, ' into device : ', device)

    # ============================================================================================================================================
    #                                       2. Define dataset & collator
    # ============================================================================================================================================

    test_dataset = IRTestDataset(data_args.data_dir, training_args, max_seq_length=data_args.max_seq_length, filenames_path_list=data_args.test_file)

    data_collator = IRCollator(tokenizer=tokenizer, max_seq_length=data_args.max_seq_length, eval_mode=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    model.eval()
    logger.info("*** Predict ***")
    trainer_predictions = trainer.predict(test_dataset)
    predictions = trainer_predictions.predictions
    pickle.dump(predictions, open(os.path.join(training_args.output_dir, "predictions.pkl"), 'wb'))

    output_scores_dict = defaultdict(dict)
    for example, score in tqdm(zip(trainer.eval_dataset, predictions)):
        qid = example[0]
        passageid = example[2]
        output_scores_dict[qid][passageid] = float(score)
    print("output_scores_dict : ", len(output_scores_dict))

    with open(os.path.join(training_args.output_dir, "prediction_scores.json"), 'w') as f:
        json.dump(output_scores_dict, f, indent=4)

    print('>>> Using qrel file : ', data_args.qrel_file_path)
    qrels = list(ir_measures.read_trec_qrels(data_args.qrel_file_path))
    results_dict = ir_measures.calc_aggregate(METRICS_LIST, qrels, output_scores_dict)

    results_dict = {str(k): v for k, v in results_dict.items()}
    ids_list = sorted(list(results_dict.keys()))
    for k in ids_list:
        print(results_dict[k])

    logger.info("done model testing")
    return trainer


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Test MonoELECTRA')
    parser.add_argument('--dataset_name', required=True, type=str,
                        help='dataset name')
    parser.add_argument('--model_name_or_path', required=True, type=str,
                        help='base model name or path to be fine-tuned upon')
    parser.add_argument('--test_filename_path', required=True, type=str,
                        help='input file with test reranking data')
    parser.add_argument('--qrle_filename', required=True, type=str,
                        help='output directory to save the trained model')
    parser.add_argument('--model_type', default='google/electra-base-discriminator', required=False, type=str,
                        help='model_type to be fine-tuned upon')
    parser.add_argument('--cache_dir', required=True, type=str,
                        help='cache dir to download model and tokenizer')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32, required=False,
                        help='batch size for testing')
    args = parser.parse_args()


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, IRTrainingArguments))
    model_args, data_args, training_args, additiona_args_list = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.per_device_eval_batch_size = args.per_device_eval_batch_size
    training_args.dataloader_num_workers = 8
    training_args.seed = 1111
    model_args.model_type = args.model_type
    model_args.cache_dir = args.cache_dir
    data_args.do_eval = False
    data_args.do_train = False
    data_args.do_predict = True
    data_args.max_seq_length = 512
    data_args.qrel_file_path = args.qrle_filename
    data_args.test_file = args.test_filename_path

    training_args.output_dir = args.model_name_or_path
    model_args.model_name_or_path = args.model_name_or_path
    epochs_to_take = 1

    # ===================================================================================================
    ##                              1. Identify non-checkpoint files and folders (in case of epochs-2)
    # ===================================================================================================
    list_of_files_and_folders = os.listdir(model_args.model_name_or_path)
    list_of_checkpoint_folders = [f for f in list_of_files_and_folders if "checkpoint" in f]
    list_of_checkpoint_nums = sorted([int(f.split('-')[1]) for f in list_of_checkpoint_folders])
    if len(list_of_checkpoint_nums) > 1:
        choice_of_checkpoint_num = list_of_checkpoint_nums[epochs_to_take-1]
        choice_of_checkpoint_folder = f"checkpoint-{choice_of_checkpoint_num}"
    else:
        choice_of_checkpoint_num = list_of_checkpoint_nums[0]
        choice_of_checkpoint_folder = f"checkpoint-{choice_of_checkpoint_num}"

    # ===================================================================================================
    ##                              2. Select checkpoint and define variables
    # ===================================================================================================
    save_predictions_fn = os.path.join(model_args.model_name_or_path, f"predictions_epochs{epochs_to_take}.json")
    model_args.model_name_or_path = os.path.join(model_args.model_name_or_path, choice_of_checkpoint_folder)
    if not os.path.exists(model_args.model_name_or_path):
        sys.exit("Model checkpoint file does not exist : ", model_args.model_name_or_path)
    print('>>> Saving in the folder : ', training_args.output_dir)


    main_test(model_args, data_args, training_args)