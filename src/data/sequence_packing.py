from datasets import Dataset
from loguru import logger

def pack_sequences(dataset, context_length = 512):
    block_size = context_length + 1

    input_blocks = []
    label_blocks = []

    token_buffer = []

    logger.info("Starting sequnece packing")
    logger.info("Content length {}", context_length)
    logger.info("Block or chunk size {}", block_size)

    for example in dataset:
        token_buffer.extend(example['input_ids'])

        while len(token_buffer) >= block_size:
            chunk = token_buffer[:block_size]

            # remove consumed tokens
            token_buffer = token_buffer[block_size:]

            # next token prediction right shift
            input_ids = chunk[:-1]
            labels = chunk[1:]

            input_blocks.append(input_ids)
            label_blocks.append(labels)

    logger.info("Remaining tokens dropped {}", len(token_buffer))
    packed_dataset = Dataset.from_dict({
        "input_ids": input_blocks,
        "labels": label_blocks
    })

    logger.success("sequence packing completed")
    logger.info("No of packed sequences {}", len(packed_dataset))

    if len(packed_dataset) > 0:
        logger.info("First input sequence length {}", len(packed_dataset[0]['input_ids']))
        logger.info("First label seqeunce length {}", len(packed_dataset[0]['labels']))

    return packed_dataset