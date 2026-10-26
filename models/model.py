"""
models/model.py — PEFT-UMamba (implementation matching the manuscript)

Paper section → code
────────────────────
  §3.2  VMamba-Tiny encoder, depths {2,2,9,2}, dims {96,192,384,768}, K=16,
        ImageNet-1K pretrained, ALL weights frozen          → VMambaEncoder
        (parameter names follow the original VMamba release, so the official
        checkpoint loads by name; loading is verified and fails loudly)
  §3.3–3.4  Supplementary Scan: K → K+K′ inside every SS2D, block-diagonal
        Â=[A_frz 0; 0 A_supp], B̂=[B_frz; B_supp], Ĉ=[C_frz C_supp],
        shared input projection and shared Δ (Eq. 5–10)     → SS2D
        Implemented by concatenating along the state axis and running ONE
        selective-scan kernel, which is exactly the block-diagonal system.
  §3.5  Neighbourhood Initialisation: A_supp^(j) = A_frz^(K) + ε (Eq. 11),
        B_supp Xavier, C_supp = 0 → network equals the frozen backbone at
        initialisation (Eq. 13)                             → SS2D.reset_supplementary
  §3.6  Symmetric VSS-block decoder with Supplementary Scan  → DecoderLayer
  §3.6.1 SCAM: bidirectional GRU over ordered slices/frames,
        f̃_t = →h_t + ←h_t (Eq. 12)                          → SCAM
  §1/§4.2 Skip Attention Gates following Oktay et al. [31]  → SkipAttentionGate
  §4.2  Deep supervision at three decoder resolutions        → aux heads
  Table 6 baselines: frozen encoder (K′=0), LoRA, full fine-tuning;
        Mona / LoRand via peft_baselines.py (blocks are named VSSBlock /
  """

The Code coming soon...
        
