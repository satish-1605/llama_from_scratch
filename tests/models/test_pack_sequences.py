from src.data.sequence_packing import pack_sequences
from datasets import Dataset

def test_seq_pack():
    dummy_data = Dataset.from_dict(
        {
            "input_ids": [[10, 20, 30, 2], [40, 50, 60, 2], [70, 80, 90, 2]]
        }
    )

    packed = pack_sequences(
        dummy_data,
        context_length= 5
    )

    assert len(packed) == 2
    assert len(packed[0]["input_ids"]) == 5
    assert len(packed[0]["labels"]) == 5

    assert packed[0]["input_ids"] == [10, 20, 30, 2, 40]

    assert packed[0]["labels"] == [20, 30, 2, 40, 50]

    assert packed[1]["input_ids"] == [60, 2, 70, 80, 90]

    assert packed[1]["labels"] == [2, 70, 80, 90, 2]


if __name__ == "__main__":

    test_seq_pack()

    print("All sequence packing tests passed!")