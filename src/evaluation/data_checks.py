from loguru import logger


def validate_packed_dataset(dataset, vocab_size, context_length=512):
    assert len(dataset) > 0, "dataset is empty"

    logger.info("Validating sequences {}", len(dataset))

    for index, example in enumerate(dataset):
        input_ids = example['input_ids']
        labels = example['labels']

        assert len(input_ids) == context_length, (f"Invalid input length at index {index}")
        assert len(labels) == context_length, (f"Invalid label length at index {index}")

        assert all(0 <= token_id < vocab_size for token_id in input_ids), (f"Invalid input token id at index {index}")
        assert all(0 <= token_id < vocab_size for token_id in labels), (f"Invalid label token id at index {index}")

    logger.success("Packed dataset validation passed!")
    logger.info("Total sequences: {:,}", len(dataset))
    logger.info("Context length: {}", context_length)
    logger.info("Vocabulary size: {}", vocab_size)

def inspect_packed_dataset(dataset):

    example = dataset[0]

    logger.info("First input IDs: {}", example["input_ids"][:20])
    logger.info("First labels: {}", example["labels"][:20])

    logger.info("Input length: {}", len(example["input_ids"]))
    logger.info("Label length: {}", len(example["labels"]))