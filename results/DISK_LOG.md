# Disk usage log — delete-each-model-after-testing (target: / < 50%)

Policy inherited from [`../ventunoq-llamacpp/results/DISK_LOG.md`](../ventunoq-llamacpp/results/DISK_LOG.md)
— that file remains the full family audit trail. This repo adds its own entries below as
the L2 dtype matrix (Q6_K/Q8_0/F16, 1.5–3.4 GB each) and L5 sweeps need models.

Baseline at repo creation (2026-10-30):
```
/dev/mmcblk0p71   55G   22G   32G  41% /
```

### 2026-10-30 13:58 UTC — DOWNLOAD (LFM3-17B-QAD): `/tmp/opencode/dl/LFM3-1.7B-QAD-Q4_0.gguf`
```
43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452  /tmp/opencode/dl/LFM3-1.7B-QAD-Q4_0.gguf
```
```
/dev/mmcblk0p71   55G   24G   29G  45% /
```
