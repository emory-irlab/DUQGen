import json
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments)
from torch.utils.data import Dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.manual_seed(123)


# Most of the codes were extracted from Pyterrier codebase
class MonoT5Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f'Query: {sample[0]} Document: {sample[1]} Relevant:'
        return {
          'text': text,
          'labels': sample[2],
        }


def train(base_modename_or_path, train_data_filepath, save_model_path, cache_dir,
          per_device_train_batch_size=8, gradient_accumulation_steps=16, learning_rate=2e-5, epochs=1, logging_steps=2):
    # 1. Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_modename_or_path, use_fast=False, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_modename_or_path, cache_dir=cache_dir)


    def smart_batching_collate_text_only(batch):
        texts = [example['text'] for example in batch]
        tokenized = tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
        tokenized['labels'] = tokenizer([example['labels'] for example in batch], return_tensors='pt')['input_ids']

        for name in tokenized:
            tokenized[name] = tokenized[name].to(device)

        return tokenized


    # 2. Load training data
    train_samples = []
    for line in open(train_data_filepath).readlines():
        data = json.loads(line)

        qtext, doctext, label = data

        if label == 1.0:
            train_samples.append((qtext, doctext, 'true'))
        else:
            train_samples.append((qtext, doctext, 'false'))


    # 3. Prepare dataset
    dataset_train = MonoT5Dataset(train_samples)


    # 4. Define training arguments
    train_args = Seq2SeqTrainingArguments(
            output_dir=save_model_path,
            do_train=True,
            save_strategy='epoch',
            logging_steps=logging_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=5e-5,
            num_train_epochs=epochs,
            warmup_ratio=0.1,
            adafactor=True,
            seed=1,
            disable_tqdm=False,
            load_best_model_at_end=False,
            predict_with_generate=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False
        )


    # 5. Train
    trainer = Seq2SeqTrainer(
            model=model,
            args=train_args,
            train_dataset=dataset_train,
            tokenizer=tokenizer,
            data_collator=smart_batching_collate_text_only,
    )
    trainer.train()


    # 6. Save the trained model
    trainer.save_model(save_model_path)
    trainer.save_state()



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Train MonoT5-3B')
    parser.add_argument('--base_modename_or_path', required=True, type=str,
                        help='base model name or path to be fine-tuned upon')
    parser.add_argument('--train_data_filepath', required=True, type=str,
                        help='training data file path')
    parser.add_argument('--save_model_path', required=True, type=str,
                        help='output directory to save the trained model')
    parser.add_argument('--cache_dir', required=True, type=str,
                        help='cache dir to download model and tokenizer')

    parser.add_argument('--per_device_train_batch_size', type=int, default=8, required=False,
                        help='batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, required=False,
                        help='gradient accumulation steps to train with small batch sizes')
    parser.add_argument('--learning_rate', type=float, default=2e-5, required=False,
                        help='learning rate for optimization')
    parser.add_argument('--epochs', type=int, default=1, required=False,
                        help='number of epochs to fine-tune')
    parser.add_argument('--logging_steps', type=int, default=2, required=False,
                        help='logging steps to verbose the loss convergence')
    args = parser.parse_args()


    train(args.base_modename_or_path, args.train_data_filepath, args.save_model_path, args.cache_dir,
          args.per_device_train_batch_size, args.gradient_accumulation_steps, args.learning_rate, args.epochs, args.logging_steps)