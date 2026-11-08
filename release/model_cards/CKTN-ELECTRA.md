---
language:
  - cjm
  - km
  - tyz
  - nut
license: apache-2.0
library_name: transformers
pipeline_tag: feature-extraction
base_model: ducanhdinh/CKTN-ELECTRA
tags:
  - rembert
  - electra
  - low-resource
  - cham
  - khmer
  - tay-nung
  - vietnam
---

# CKTN-ELECTRA

CKTN-ELECTRA is a text encoder for three ethnic minority languages of Vietnam: **Cham**, **Khmer** and **Tay-Nung**. It is introduced in the paper
[*Echoes Across Vietnam's Highlands, Delta, and Coast: A Multilingual Corpus for Cham, Khmer, and Tay-Nung*](https://arxiv.org/abs/2607.08362).

This checkpoint is the **discriminator** after continued pre-training on the CKTN corpus with script-aware calibrated replaced-token detection. It is a standard `RemBertModel` with no task head.

| | |
|---|---|
| Architecture | RemBERT, 32 layers, hidden size 1152, 18 heads, 256-d input embeddings |
| Vocabulary | 254,513 SentencePiece pieces: RemBERT's 250,300 plus 4,213 CKTN pieces |
| Initialisation | [`ducanhdinh/CKTN-ELECTRA`](https://huggingface.co/ducanhdinh/CKTN-ELECTRA) (vocabulary-augmented RemBERT) |
| Pre-training data | CKTN corpus, 35,492 training documents (Cham 9,184 · Khmer 22,246 · Tay-Nung 4,062) |
| Max sequence length | 512 |
| Code | [github.com/TracDucAnh/CKTN-ELECTRA](https://github.com/TracDucAnh/CKTN-ELECTRA) |

## Usage

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nhatkhangdtp/CKTN-ELECTRA", use_fast=False)
model = AutoModel.from_pretrained("nhatkhangdtp/CKTN-ELECTRA")

inputs = tokenizer("Noọng pỉ hẩu chương trình dân tộc", return_tensors="pt", truncation=True, max_length=512)
hidden = model(**inputs).last_hidden_state  # (1, seq_len, 1152)
```

Fine-tune the whole encoder for downstream tasks.

## Training

The model is trained jointly with an 8-layer generator. The two share token, position and segment embeddings, and are trained with `L = L_MLM + λ(t) · L_RTD`:

- **Calibrated replacements.** Replacements are drawn from the generator's top-64 candidates at temperature 1.25. A candidate is kept only if it:
  - differs from the original token;
  - is well formed: no special tokens, and no punctuation, digit or symbol pieces unless the original is one too;
  - is script compatible: Khmer-script pieces never replace Latin-script Cham or Tay-Nung pieces, and vice versa;
  - has embedding cosine similarity to the original in [0.15, 0.95].
- **Linear schedule.** λ = 0 in epochs 1–2, rises linearly to 50 during epoch 3, and is fixed for epochs 4–6.
- **Optimisation.** AdamW, lr 2e-5, warmup 6%, weight decay 0.01, gradient clipping 1.0, 15% masking, batch size 8.

The released weights are from the epoch with the best development replaced-token-detection F1: epoch 4, with RTD F1 0.870 and accuracy 0.967. The generator and RTD head are not included.

## Results (from the paper)

After continued pre-training, all encoders were fine-tuned for 5 epochs at max length 512.

| Model | Category acc. | Category macro-F1 | Retrieval MRR@10 | Retrieval R@10 |
|---|---:|---:|---:|---:|
| **CKTN-ELECTRA** | **0.9214** | **0.7103** | 0.7718 | 0.9039 |
| RemBERT | 0.1454 | 0.0053 | **0.7787** | **0.9049** |
| XLM-R-base | 0.8169 | 0.5617 | 0.7361 | 0.8789 |
| mBERT | 0.8024 | 0.5054 | 0.3794 | 0.4504 |

Retrieval pairs each summary with its own article, so it largely rewards lexical overlap. Classification is the better test of document-level representations.

## Limitations

- The corpus comes from government and state-affiliated news outlets, so it is formal news register.
- Tay-Nung data is especially scarce.
- The models have not been audited for social bias.

## Citation

```bibtex
@misc{dinh2026echoes,
  title         = {Echoes Across Vietnam's Highlands, Delta, and Coast: A Multilingual Corpus for Cham, Khmer, and Tay-Nung},
  author        = {Dinh, Anh Trac Duc and Vo, Khang Nhat Hoang and Doan, Vinh Cong and Ta, Tai Tien and Lam, Khoa Duc Anh},
  year          = {2026},
  eprint        = {2607.08362},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2607.08362}
}
```
