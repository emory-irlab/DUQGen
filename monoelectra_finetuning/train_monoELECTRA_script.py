import os
import sys
import json
import torch
import logging
import datasets
import argparse
import transformers
import torch.nn as nn
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
    AutoModel,
    Trainer)
from typing import List, Tuple
from dataclasses import dataclass
from transformers import DataCollatorWithPadding
from data_args import ModelArguments, DataTrainingArguments, IRTrainingArguments
device = torch.device('cuda')
torch.manual_seed(123)
logger = logging.getLogger(__name__)


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


def main_train(model_args, data_args, training_args):
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
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    model = IRModel.from_pretrained(base_model_name=model_args.model_type,
                                    saved_path=model_args.model_name_or_path,
                                    cache_dir=model_args.cache_dir)
    model.to(device)
    print("Loaded base model from : ", model_args.model_name_or_path, ' into device : ', device)

    # ============================================================================================================================================
    #                                       2. Define dataset & collator
    # ============================================================================================================================================
    train_dataset = IRTrainDataset(data_args.data_dir, training_args, max_seq_length=data_args.max_seq_length, filenames_path_list=data_args.train_file)

    data_collator = IRCollator(tokenizer=tokenizer, max_seq_length=data_args.max_seq_length, eval_mode=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # ============================================================================================================================================
    #                                       3. Train
    # ============================================================================================================================================
    train_result = trainer.train()
    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("done model training")
    return trainer




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Train MonoELECTRA')
    parser.add_argument('--dataset_name', required=True, type=str,
                        help='dataset name')
    parser.add_argument('--model_name_or_path', required=True, type=str,
                        help='base model name or path to be fine-tuned upon')
    parser.add_argument('--cache_dir', required=True, type=str,
                        help='cache dir to download model and tokenizer')
    parser.add_argument('--per_device_train_batch_size', type=int, default=32, required=False,
                        help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5, required=False,
                        help='learning rate for optimization')
    parser.add_argument('--model_type', default='google/electra-base-discriminator', required=False, type=str,
                        help='model_type to be fine-tuned upon')
    parser.add_argument('--train_file', required=True, type=str,
                        help='training data file path')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='output directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=1, required=False,
                        help='number of epochs to fine-tune')
    parser.add_argument('--logging_steps', type=int, default=10, required=False,
                        help='logging steps to verbose the loss convergence')
    args = parser.parse_args()




    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, IRTrainingArguments))
    model_args, data_args, training_args, additiona_args_list = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.save_steps = 10e10
    training_args.logging_steps = args.logging_steps
    training_args.per_device_train_batch_size = args.per_device_train_batch_size
    training_args.per_device_eval_batch_size = 16
    training_args.fp16 = True
    training_args.dataloader_num_workers = 8
    training_args.weight_decay = 0.01
    training_args.learning_rate = args.learning_rate
    training_args.optim = "adamw_hf"
    training_args.lr_scheduler_type = "inverse_sqrt"
    training_args.warmup_ratio = 0.1
    training_args.seed = 1111
    training_args.save_strategy = "epoch"
    training_args.num_train_epochs = args.epochs
    model_args.model_type = args.model_type
    model_args.cache_dir = args.cache_dir
    model_args.model_name_or_path = args.model_name_or_path
    data_args.do_eval = False
    data_args.do_train = True
    data_args.do_predict = False
    data_args.max_seq_length = 512
    data_args.train_file = args.train_file
    training_args.output_dir = args.output_dir

    print('>>> Saving in the folder : ', training_args.output_dir)

    main_train(model_args, data_args, training_args)