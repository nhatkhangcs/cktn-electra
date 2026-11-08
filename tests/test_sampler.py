import torch

from cktn_electra.config import SamplerConfig
from cktn_electra.modeling import DifficultyCalibratedSampler

LATIN = [4, 5, 6, 7]
KHMER = [8, 9, 10, 11]
NOISE = [12, 13, 14, 15]
SPECIAL = [0, 1, 2, 3]


def _embedding(vocab_size: int) -> torch.Tensor:
    # All candidate embeddings have cosine 0.5 with every other token (inside the band).
    base = torch.ones(vocab_size, 1)
    unique = torch.eye(vocab_size)
    return torch.cat([base, unique], dim=1)


def _run(tokenizer, original_id: int, preferred_ids, script_aware=True, rho=(0.15, 0.95), trials=200):
    vocab = tokenizer.vocab_size
    sampler = DifficultyCalibratedSampler(
        vocab, SamplerConfig(top_k=vocab, temperature=1.0, rho_min=rho[0], rho_max=rho[1], script_aware=script_aware), tokenizer
    )
    logits = torch.full((trials, 1, vocab), -5.0)
    logits[..., preferred_ids] = 5.0
    original = torch.full((trials, 1), original_id)
    out = sampler(logits, original, torch.ones(trials, 1, dtype=torch.bool), torch.ones(trials, 1), _embedding(vocab))
    return out


def test_replacements_respect_script_and_noise(tokenizer):
    # The generator prefers Khmer, special and noise tokens for a Latin original.
    out = _run(tokenizer, LATIN[0], KHMER + SPECIAL + NOISE)
    replaced = out["corrupted_ids"][out["replacement_mask"]]
    assert out["replacement_mask"].any()
    assert set(replaced.tolist()) <= set(LATIN[1:])  # only other Latin tokens survive
    assert not (out["corrupted_ids"] == LATIN[0]).logical_and(out["replacement_mask"]).any()


def test_disabling_script_filter_allows_cross_script(tokenizer):
    out = _run(tokenizer, LATIN[0], KHMER, script_aware=False)
    replaced = set(out["corrupted_ids"][out["replacement_mask"]].tolist())
    assert replaced & set(KHMER)
    assert not replaced & set(SPECIAL + NOISE)  # well-formedness filter still active


def test_noise_original_may_be_replaced_by_noise(tokenizer):
    out = _run(tokenizer, NOISE[0], NOISE[1:])
    assert set(out["corrupted_ids"][out["replacement_mask"]].tolist()) <= set(NOISE[1:])


def test_empty_candidate_set_keeps_token(tokenizer):
    # Similarity band excludes every candidate -> nothing is replaced.
    out = _run(tokenizer, LATIN[0], LATIN, rho=(0.9, 0.95))
    assert not out["replacement_mask"].any()
    assert (out["corrupted_ids"] == LATIN[0]).all()
    assert out["valid_candidate_rate"].item() == 0.0


def test_unmasked_positions_never_replaced(tokenizer):
    vocab = tokenizer.vocab_size
    sampler = DifficultyCalibratedSampler(vocab, SamplerConfig(top_k=vocab), tokenizer)
    logits = torch.randn(4, 6, vocab)
    original = torch.full((4, 6), LATIN[0])
    is_masked = torch.zeros(4, 6, dtype=torch.bool)
    out = sampler(logits, original, is_masked, torch.ones(4, 6), _embedding(vocab))
    assert not out["replacement_mask"].any()
