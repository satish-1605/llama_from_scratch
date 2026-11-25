from datasets import load_dataset
from loguru import logger
from src.tokenizer.hf_tokenizer import HFTokenizer
from src.data.sequence_packing import pack_sequences

from pathlib import Path
from datasets import load_from_disk

from src.evaluation.data_checks import (
    validate_packed_dataset,
    inspect_packed_dataset
)

TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_NAME = "roneneldan/TinyStories"

def load_tokenizer():
    logger.info("Tokenizer loading: {}", TOKENIZER_NAME)

    tokenizer = HFTokenizer(TOKENIZER_NAME)

    logger.success("tokenizer loaded successfully.")
    return tokenizer

def tokenize_batch(batch, tokenizer):
    #convert raw stories into token ids
    
    input_ids = [tokenizer.encode_with_eos(text)
                 for text in batch['text']
                 ]

    return {
        "input_ids": input_ids,
        "attention_mask": [[1] * len(ids) for ids in input_ids]
    }

def tokenize_dataset(train_data, val_data, tokenizer):
    logger.info("tokenizing training dataset...")
    tokenized_train = train_data.map(
        tokenize_batch,
        batched=True,
        batch_size = 1000,
        fn_kwargs= {"tokenizer": tokenizer},
        remove_columns = train_data.column_names,
        desc = "tokenizing train data"
    )

    logger.info("tokenizing val dataset...")
    tokenized_val = val_data.map(
            tokenize_batch,
            batched=True,
            batch_size = 1000,
            fn_kwargs= {"tokenizer": tokenizer},
            remove_columns = val_data.column_names,
            desc = "tokenizing val data"
        )

    logger.success("Tokenization completed")
    return tokenized_train, tokenized_val


def load_pretraining_dataset():
    logger.info("Loading the dataset: {}", DATASET_NAME)

    dataset = load_dataset(DATASET_NAME)

    logger.success("dataset loaded successfully")

    logger.info("Dataset loaded succesfully")
    logger.info("{}", dataset)

    return dataset

def inspect_dataset(dataset):
    train_data = dataset['train']
    val_data = dataset['val']

    logger.info("Training Docs: {}", len(train_data))
    logger.info("val Docs: {}", len(val_data))

    logger.info("Training columns: {}", train_data.column_names)

    logger.info("First Training example:")
    logger.info("{}", train_data[0])

    logger.info("First val example:")
    logger.info("{}", val_data[0])

def select_dataset_subset(dataset, train_size=50000, val_size = 5000):
    train_data = dataset['train'].shuffle(seed=42).select(
        range(train_size)
        )

    val_data = dataset['validation'].shuffle(seed=42).select(
            range(val_size)
            )

    logger.info("Selected training stories: {:,}", len(train_data))
    logger.info("Selected val stories: {:,}", len(val_data))

    return train_data, val_data

def save_processed_dataset(train_dataset, val_dataset):
    output_dir = Path("artifacts/processed/pretraining")

    train_path = output_dir / "train"
    val_path = output_dir / "val"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving processed training dataset...")
    train_dataset.save_to_disk(str(train_path))

    logger.info("Saving processed val dataset....")
    val_dataset.save_to_disk(str(val_path))

    logger.success("Processed datasets saved successfully.")
    return train_path, val_path

def load_processed_dataset(train_path, val_path):

    logger.info("Loading saved datasets...")

    train_dataset = load_from_disk(str(train_path))
    val_dataset = load_from_disk(str(val_path))

    logger.success("Datasets loaded successfully")

    return train_dataset, val_dataset

if __name__ == "__main__":
    dataset = load_pretraining_dataset()

    train_data, val_data = select_dataset_subset(dataset)

    tokenizer = load_tokenizer()

    tokenized_train, tokenized_val = tokenize_dataset(train_data, val_data, tokenizer)

    logger.info("Train tokenized columns {}", tokenized_train.column_names)
    logger.info("val tokenized columns {}", tokenized_val.column_names)

    logger.info("First tokenized story")
    logger.info("{}", tokenized_train[0]['input_ids'])


    logger.info("Decoded story")
    logger.info("{}", tokenizer.decode(tokenized_train[0]['input_ids']))

    logger.info("First story token count: {}",
                len(tokenized_train[0]['input_ids']))

    logger.info("First story attention count: {}",
                    len(tokenized_train[0]['attention_mask']))

    packed_train = pack_sequences(dataset= tokenized_train)
    packed_val = pack_sequences(dataset=tokenized_val)

    logger.info("Packed train sequnces {}", len(packed_train))
    logger.info("Packed val sequnces {}", len(packed_val))

    logger.info("First input shape {}", len(packed_train[0]['input_ids']))
    logger.info("First labels shape {}", len(packed_train[0]['labels']))

    # validate before saving
    validate_packed_dataset(packed_train,
                            vocab_size= tokenizer.vocab_size)

    validate_packed_dataset(packed_val,
                            vocab_size= tokenizer.vocab_size)

    inspect_packed_dataset(packed_train)
    # Save
    train_path, val_path = save_processed_dataset(
        packed_train,
        packed_val
    )

    # Reload to verify persistence
    packed_train, packed_val = load_processed_dataset(
    train_path,
    val_path
    ) 

    logger.success("Final pretraining data pipeline completed!")  


    





