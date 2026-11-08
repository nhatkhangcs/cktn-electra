import json

import numpy as np
import torch

from cktn_electra.data import ELECTRADataset, load_or_build_chunks, read_records, split_files, tokenize_and_chunk
from cktn_electra.downstream.classification import CategoryDataset, balanced_class_weights, build_label_map
from cktn_electra.downstream.retrieval import GroupedBatchSampler, build_eval_sets, compute_ranks, ranks_to_metrics
from cktn_electra.hub import load_hf_token


def _write_corpus(root):
    for split in ("train", "dev"):
        (root / "corpus" / split).mkdir(parents=True)
        for language in ("cham", "khmer", "tay_nung"):
            records = [
                {"title": f"{language} t{i}", "content": f"{language} content {i} " * 5, "summary": f"s{i}" if i % 2 else "",
                 "tags": ["tag"], "category": f"{language}-cat{i % 2}"}
                for i in range(4)
            ]
            (root / "corpus" / split / f"{language}_{split}.json").write_text(json.dumps(records), encoding="utf-8")
    return root / "corpus"


def test_chunks_have_boundaries_and_max_length(tokenizer):
    chunks = tokenize_and_chunk(["a" * 25, "bb"], tokenizer, max_len=10)
    assert all(c[0] == 1 and c[-1] == 2 and len(c) <= 10 for c in chunks)
    assert sum(len(c) - 2 for c in chunks) == 27


def test_chunk_cache_roundtrip(tmp_path, tokenizer):
    corpus = _write_corpus(tmp_path)
    files = split_files(corpus, "train")
    first = load_or_build_chunks("train", files, tokenizer, 16, tmp_path / "cache", "fake")
    assert len(list((tmp_path / "cache").glob("train_*.pkl"))) == 1
    assert load_or_build_chunks("train", files, tokenizer, 16, tmp_path / "cache", "fake") == first


def test_masking_rate_and_specials(tokenizer):
    np.random.seed(0)
    chunk = [1] + list(range(4, 16)) * 3 + [2]
    item = ELECTRADataset([chunk], tokenizer, max_len=48, mask_rate=0.15)[0]
    masked = item["labels"] != -100
    assert masked.sum().item() == round(36 * 0.15)
    assert not masked[0] and not masked[len(chunk) - 1]
    assert (item["input_ids"][masked] == 3).all()
    assert item["attention_mask"].sum().item() == len(chunk)


def test_read_records_tags_language(tmp_path):
    records = read_records(_write_corpus(tmp_path), "dev")
    assert {r["_language"] for r in records} == {"cham", "khmer", "tay_nung"}
    assert records[0]["_id"] == "cham_dev:0"


def test_class_weights_balanced(tmp_path):
    records = read_records(_write_corpus(tmp_path), "train")
    label_to_id = build_label_map(records)
    weights = balanced_class_weights(CategoryDataset(records, label_to_id, ["title"]), len(label_to_id))
    assert torch.allclose(weights, torch.ones(len(label_to_id)))


def test_retrieval_metrics_and_eval_sets(tmp_path):
    records = read_records(_write_corpus(tmp_path), "dev")
    sets = build_eval_sets(records, ["summary"], [], ["title", "content"])
    assert len(sets["documents"]) == 12 and len(sets["queries"]) == 6
    ranks = compute_ranks(torch.eye(3), torch.eye(3), torch.tensor([0, 1, 2]))
    assert ranks.tolist() == [1, 1, 1]
    metrics = ranks_to_metrics(torch.tensor([1, 2, 20]))
    assert abs(metrics["mrr_at_10"] - (1 + 0.5) / 3) < 1e-6
    assert abs(metrics["recall_at_10"] - 2 / 3) < 1e-6


def test_grouped_batches_stay_within_group():
    examples = [{"group": g} for g in "aaaaabbbbb"]
    sampler = GroupedBatchSampler(examples, batch_size=2, drop_last=True)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 4
    assert all(len({examples[i]["group"] for i in b}) == 1 for b in batches)


def test_load_hf_token_from_env_file(tmp_path, monkeypatch):
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_HUB"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nHUGGINGFACE_HUB = "hf_abc"\n')
    assert load_hf_token(env) == "hf_abc"
    env.write_text("HF_TOKEN=hf_xyz\n")
    assert load_hf_token(env) == "hf_xyz"
