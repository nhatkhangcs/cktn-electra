import pytest
import torch
from transformers import RemBertConfig

# Vocabulary used by the fake tokenizer: ids 0-3 are special tokens.
VOCAB = [
    "[PAD]", "[CLS]", "[SEP]", "[MASK]",
    "▁nhà", "▁noọng", "▁pỉ", "▁bruk",  # Latin (Tay-Nung / Cham)
    "▁ក", "▁ខ្មែរ", "▁ភាសា", "▁ជា",  # Khmer
    "▁,", "▁.", "▁123", "▁%",  # noise / script-neutral
]


class FakeTokenizer:
    pad_token_id, cls_token_id, sep_token_id, mask_token_id = 0, 1, 2, 3
    all_special_ids = [0, 1, 2, 3]
    vocab_size = len(VOCAB)
    name_or_path = "fake"

    def convert_ids_to_tokens(self, ids):
        return [VOCAB[i] for i in ids]

    def encode(self, text, add_special_tokens=False, truncation=False):
        return [4 + (ord(ch) % 12) for ch in text if not ch.isspace()]


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def tiny_config():
    return RemBertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        input_embedding_size=16,
        output_embedding_size=16,
        max_position_embeddings=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
    )


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
