# Disk usage log — delete-each-model-after-testing (target: / < 50%)

Policy inherited from [`../ventunoq-llamacpp/results/DISK_LOG.md`](../ventunoq-llamacpp/results/DISK_LOG.md)
— that file remains the full family audit trail; entries below were made during the Round-2
survey (2026-10-30) and are mirrored here because they belong to this repo's subject.

### 2026-10-30 07:1x UTC — DOWNLOAD (Round-2 cross-check, `/tmp/opencode/`): `Qwen3.5-0.8B-Q4_K_M.gguf` (unsloth, 533 MB), MNN pre-converted weights `taobao-mnn/Qwen3.5-0.8B-MNN` (0.55 GB) + `Qwen3.5-2B-MNN` (1.19 GB); source clone `MNN/` (406 MB, kept — rebuild recipe in scripts/build_mnn.sh)
```
bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517  mr/Qwen3.5-0.8B-Q4_K_M.gguf
4e25ef4cbfb49a33013d17300c2f68cd9c9ce8f2e3f7b947fd2c574ec297ca5e  mnn/Qwen3.5-0.8B-MNN/llm.mnn.weight
c93f71a2dbecf9328782bd38861656d8faa82e95e7f99607350074768a482054  mnn/Qwen3.5-2B-MNN/llm.mnn.weight
```

### 2026-10-30 07:2x UTC — DELETE (Round-2 finished, numbers in BENCHMARKS.md §1–2): the GGUF + both MNN model dirs above
```
/dev/mmcblk0p71   55G   24G   29G  45% /   (before)
/dev/mmcblk0p71   55G   22G   32G  41% /   (after)
```

### 2026-10-31 20:38 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-0.8B-MNN (462M)
```
4e25ef4cbfb49a33013d17300c2f68cd9c9ce8f2e3f7b947fd2c574ec297ca5e  models/Qwen3.5-0.8B-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   21G   32G  40% /
```

### 2026-10-31 20:44 UTC — DELETE (quality suite done): `models/Qwen3.5-0.8B-MNN`
```
/dev/mmcblk0p71   55G   21G   32G  40% /
```

### 2026-10-31 20:44 UTC — DELETE (quality suite done): `models/Qwen3.5-0.8B-MNN-text`
```
/dev/mmcblk0p71   55G   21G   32G  40% /
```

### 2026-10-31 20:59 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-0.8B-MNN (462M)
```
4e25ef4cbfb49a33013d17300c2f68cd9c9ce8f2e3f7b947fd2c574ec297ca5e  models/Qwen3.5-0.8B-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   21G   32G  40% /
```

### 2026-10-31 22:07 UTC — DELETE (quality suite done): `models/Qwen3.5-0.8B-MNN`
```
/dev/mmcblk0p71   55G   21G   32G  40% /
```

### 2026-10-31 22:07 UTC — DELETE (quality suite done): `models/Qwen3.5-0.8B-MNN-text`
```
/dev/mmcblk0p71   55G   21G   32G  40% /
```
