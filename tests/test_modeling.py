import torch

from cktn_electra.checkpoints import RunPaths, build_hf_encoder, load_model_weights, save_epoch
from cktn_electra.config import PretrainConfig
from cktn_electra.modeling import CKTNElectra, pretrain_parameter_groups


def _model(tokenizer, tiny_config, **overrides):
    return CKTNElectra(PretrainConfig(**overrides), tokenizer=tokenizer, load_pretrained=False, disc_config=tiny_config)


def _batch(vocab_size, batch=2, length=12):
    input_ids = torch.randint(4, vocab_size, (batch, length))
    attention_mask = torch.ones(batch, length, dtype=torch.long)
    attention_mask[1, -3:] = 0
    labels = torch.full_like(input_ids, -100)
    labels[:, 2::3] = input_ids[:, 2::3]
    input_ids[:, 2::3] = 3  # [MASK]
    return input_ids, attention_mask, labels


def test_generator_depth_is_a_quarter(tokenizer, tiny_config):
    model = _model(tokenizer, tiny_config)
    assert len(model.generator.encoder.layers) == tiny_config.num_hidden_layers // 4


def test_forward_lambda_zero_skips_discriminator(tokenizer, tiny_config):
    model = _model(tokenizer, tiny_config)
    ids, mask, labels = _batch(tiny_config.vocab_size)
    out = model(ids, mask, labels=labels, lam=0.0)
    assert out["disc_logits"] is None
    assert out["loss"].item() == out["loss_mlm"].item()
    assert out["gen_logits"].shape == (2, 12, tiny_config.vocab_size)


def test_forward_joint_loss_and_gradients(tokenizer, tiny_config):
    model = _model(tokenizer, tiny_config)
    ids, mask, labels = _batch(tiny_config.vocab_size)
    out = model(ids, mask, labels=labels, lam=50.0)
    assert out["disc_logits"].shape == (2, 12, 1)
    torch.testing.assert_close(out["loss"], out["loss_mlm"] + 50.0 * out["loss_disc"])
    out["loss"].backward()
    assert model.shared_embeddings.word_embeddings.weight.grad is not None
    assert model.discriminator.rtd_head.out.weight.grad is not None
    # RTD labels: replaced positions are 0 and only occur at masked positions.
    assert not out["replacement_mask"][labels == -100].any()
    assert torch.equal(out["rtd_labels"], (~out["replacement_mask"]).float())


def test_parameter_groups_cover_all_parameters_once(tokenizer, tiny_config):
    model = _model(tokenizer, tiny_config)
    groups = pretrain_parameter_groups(model, 0.01)
    grouped = [id(p) for g in groups for p in g["params"]]
    assert len(grouped) == len(set(grouped)) == sum(1 for _ in model.parameters())


def test_checkpoint_roundtrip_and_hf_export(tmp_path, tokenizer, tiny_config):
    model = _model(tokenizer, tiny_config).eval()
    paths = RunPaths(tmp_path)
    paths.make_dirs()
    save_epoch(paths, model, 1, {"f1": 0.5}, True, {"epoch": 1})

    restored = _model(tokenizer, tiny_config).eval()
    assert load_model_weights(restored, paths.disc_path("best"), paths.gen_path("best")) == (1, 1)
    for (name, a), (_, b) in zip(model.state_dict().items(), restored.state_dict().items()):
        assert torch.equal(a, b), name

    # The exported HF encoder must reproduce shared embeddings + discriminator encoder.
    hf = build_hf_encoder(paths.disc_path("best"), base_model="unused", config=tiny_config)
    ids = torch.randint(4, tiny_config.vocab_size, (2, 10))
    with torch.no_grad():
        reference = model.discriminator.encoder(hidden_states=model.shared_embeddings(ids))[0]
        exported = hf(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
    torch.testing.assert_close(reference, exported, atol=1e-5, rtol=1e-5)


def test_state_dict_keys_match_legacy_format(tokenizer, tiny_config):
    """Checkpoints written by earlier versions of this code must still load."""
    model = _model(tokenizer, tiny_config)
    keys = set(model.state_dict())
    for expected in (
        "shared_embeddings.word_embeddings.weight",
        "shared_embeddings.LayerNorm.bias",
        "generator.embeddings_project.weight",
        "generator.encoder.layers.0.self_attn.in_proj_weight",
        "generator.mlm_head.decoder.weight",
        "discriminator.encoder.embedding_hidden_mapping_in.weight",
        "discriminator.encoder.layer.0.attention.self.query.weight",
        "discriminator.rtd_head.out.weight",
    ):
        assert expected in keys, expected
