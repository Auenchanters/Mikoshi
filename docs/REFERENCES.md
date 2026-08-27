# References for the Dense Control and Reproducibility Stack

These sources inform Tasks 01–04 only. The recurrent-depth and neural-memory
papers in `plan.md` remain future architectural references and were not used as
permission to add those components to the dense control.

## Model components

1. Vaswani et al., **Attention Is All You Need**, arXiv:1706.03762.
   https://arxiv.org/abs/1706.03762
2. Zhang and Sennrich, **Root Mean Square Layer Normalization**,
   arXiv:1910.07467. https://arxiv.org/abs/1910.07467
3. Su et al., **RoFormer: Enhanced Transformer with Rotary Position
   Embedding**, arXiv:2104.09864. https://arxiv.org/abs/2104.09864
4. Shazeer, **GLU Variants Improve Transformer**, arXiv:2002.05202.
   https://arxiv.org/abs/2002.05202
5. Ainslie et al., **GQA: Training Generalized Multi-Query Transformer Models
   from Multi-Head Checkpoints**, arXiv:2305.13245.
   https://arxiv.org/abs/2305.13245
6. Loshchilov and Hutter, **Decoupled Weight Decay Regularization**,
   arXiv:1711.05101. https://arxiv.org/abs/1711.05101

## Software and reproducibility references

7. PyTorch, **Reproducibility Notes**.
   https://docs.pytorch.org/docs/stable/notes/randomness.html
8. PyTorch, **Scaled Dot Product Attention**.
   https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
9. Hugging Face, **Tokenizers Components**.
   https://huggingface.co/docs/tokenizers/components

## Deferred AURORA references

Titans, ATLAS, Nested Learning/HOPE, Huginn, Gated Recurrent Transformers,
MeSH, LoopFormer, STARS, Mixture-of-Recursions, and MoNe are catalogued in
`plan.md` section 28. They become relevant in Task 05 or later and are not
implemented in this phase.
