from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from transformers import TrainingArguments


@dataclass
class IRTrainingArguments(TrainingArguments):
    do_eval_only: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    output_dir: str = field(
        default="__pycache__", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="bert-base-uncased", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
        default="bert-base-uncased", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    adapter_model_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The data folder contains all files"}
    )
    inputs_train: Optional[str] = field(
        default=None,
        metadata={"help": "The data folder contains all files"}
    )
    inputs_test: Optional[str] = field(
        default=None,
        metadata={"help": "The data folder contains all files"}
    )
    qrel_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "The data folder contains all files"}
    )
    input_filenames_config_file: Optional[str] = field(
        default=None,
        metadata={"help": "The file that contains list of files to be loaded"}
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The data folder contains all files"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    no_save_model: Optional[bool] = field(
        default=False
    )
    max_seq_length: Optional[int] = field(
        default=180
    )
    decode_early_stopping: Optional[int] = field(
        default=1
    )


    def __post_init__(self):
        pass
