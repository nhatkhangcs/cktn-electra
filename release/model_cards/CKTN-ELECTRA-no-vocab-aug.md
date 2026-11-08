---
language:
  - cjm
  - km
  - tyz
  - nut
license: apache-2.0
library_name: transformers
pipeline_tag: feature-extraction
base_model: google/rembert
tags:
  - rembert
  - electra
  - low-resource
  - cham
  - khmer
  - tay-nung
  - vietnam
  - ablation
---

# CKTN-ELECTRA (ablation: no vocabulary augmentation)

This is an **ablation checkpoint** from the paper [*Echoes Across Vietnam's Highlands, Delta, and Coast: A Multilingual Corpus for Cham, Khmer, and Tay-Nung*](https://arxiv.org/abs/2607.08362). It is released so that the ablation study can be reproduced. For downstream use, take the full model, [`nhatkhangdtp/CKTN-ELECTRA`](https://huggingface.co/nhatkhangdtp/CKTN-ELECTRA).

It was trained exactly like CKTN-ELECTRA, except that the discriminator, tokenizer and shared embeddings start from the **original `google/rembert`** (250,300 pieces) instead of the vocabulary-augmented `ducanhdinh/CKTN-ELECTRA`. All other settings are the same: 8-layer generator with shared embeddings, top-64 candidates at temperature 1.25, similarity band [0.15, 0.95], AdamW at lr 2e-5, 6 epochs, 15% masking and sequence length 512.

The weights are the discriminator from the epoch with the best development replaced-token-detection F1: epoch 6, with RTD F1 0.882 and accuracy 0.970. It is a plain `RemBertModel` with no task head.

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nhatkhangdtp/CKTN-ELECTRA-no-vocab-aug", use_fast=False)
model = AutoModel.from_pretrained("nhatkhangdtp/CKTN-ELECTRA-no-vocab-aug")
```

Code and training configuration: [github.com/TracDucAnh/CKTN-ELECTRA](https://github.com/TracDucAnh/CKTN-ELECTRA), config `configs/pretrain/no_vocab_aug.json`.

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
