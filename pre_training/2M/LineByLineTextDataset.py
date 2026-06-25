import json
import logging
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

DEPRECATION_WARNING = (
    "This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: {0}"
)

# class LineByLineTextDataset(Dataset):
#     """
#     This will be superseded by a framework-agnostic approach soon.
#     """
#
#     def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
#         warnings.warn(
#             DEPRECATION_WARNING.format(
#                 "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
#             ),
#             FutureWarning,
#         )
#         assert os.path.isfile(file_path), f"Input file path {file_path} not found"
#         # Here, we do not cache the features, operating under the assumption
#         # that we will soon use fast multithreaded tokenizers from the
#         # `tokenizers` repo everywhere =)
#         logger.info(f"Creating features from dataset file at {file_path}")
#
#         with open(file_path, encoding="utf-8") as f:
#             lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
#
#         batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
#         self.examples = batch_encoding["input_ids"]
#         self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]
#
#     def __len__(self):
#         return len(self.examples)
#
#     def __getitem__(self, i) -> Dict[str, torch.tensor]:
#         return self.examples[i]
#
#


import os
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from joblib import Parallel, delayed

class LineByLineTextDataset(Dataset):
    """
    This class processes text files line by line using a tokenizer,
    leveraging multiple CPU cores to speed up the tokenization process.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, n_jobs: int = -1):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        logger.info(f"Creating features from dataset file at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # Parallel tokenization of lines
        self.examples = Parallel(n_jobs=n_jobs)(
            delayed(self.tokenize)(line, tokenizer, block_size) for line in lines
        )

        # self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples if e is not None]

    @staticmethod
    def tokenize(line, tokenizer, block_size):
        """Helper function to tokenize a line."""
        try:
            return tokenizer.encode_plus(line, add_special_tokens=True, truncation=True, max_length=block_size)[
                "input_ids"]
        except Exception as e:
            return None
        # return tokenizer.encode_plus(line, add_special_tokens=True, truncation=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
