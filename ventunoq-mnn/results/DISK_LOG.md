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

### 2026-11-01 21:20 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-2B-MNN (1.2G)
```
c93f71a2dbecf9328782bd38861656d8faa82e95e7f99607350074768a482054  models/Qwen3.5-2B-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   22G   32G  41% /
```

### 2026-11-01 23:36 UTC — DELETE (quality suite done): `models/Qwen3.5-2B-MNN`
```
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-01 23:36 UTC — DELETE (quality suite done): `models/Qwen3.5-2B-MNN-text`
```
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-01 23:37 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-0.8B-Base-MNN (462M)
```
25f43e453651ff3f87ec256cbb97f9d5e3a88e0c2f42e9eca927751a3e753b4a  models/Qwen3.5-0.8B-Base-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   21G   32G  40% /
```

### 2026-11-02 01:01 UTC — DELETE (quality suite done): `models/Qwen3.5-0.8B-Base-MNN`
```
/dev/mmcblk0p71   55G   21G   32G  40% /
```

### 2026-11-02 01:01 UTC — DELETE (quality suite done): `models/Qwen3.5-0.8B-Base-MNN-text`
```
/dev/mmcblk0p71   55G   21G   32G  40% /
```

### 2026-11-02 01:03 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-2B-Base-MNN (1.2G)
```
bb0591a3dfeb49d2ffab46008eb65e82b86d85c95172a8cd91426a19a851c303  models/Qwen3.5-2B-Base-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   22G   32G  41% /
```

### 2026-11-02 04:03 UTC — DELETE (quality suite done): `models/Qwen3.5-2B-Base-MNN`
```
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-02 04:03 UTC — DELETE (quality suite done): `models/Qwen3.5-2B-Base-MNN-text`
```
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-02 04:05 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-4B-MNN (2.5G)
```
a31d50e5668ce6b22655ff4750cfd48b972790ae883f2c154ba34b9173b38de1  models/Qwen3.5-4B-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-02 09:11 UTC — DELETE (quality suite done): `models/Qwen3.5-4B-MNN`
```
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-02 09:11 UTC — DELETE (quality suite done): `models/Qwen3.5-4B-MNN-text`
```
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-02 09:12 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled-MNN (459M)
```
55aad7cb6c019c8db59163f99f0a1bbbfb170bdf40da1b8a4ceac5b87f64b4f4  models/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   21G   32G  40% /
```

### 2026-11-02 10:37 UTC — DELETE (quality suite done): `models/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled-MNN`
```
/dev/mmcblk0p71   55G   21G   32G  40% /
```

### 2026-11-02 10:37 UTC — DELETE (quality suite done): `models/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text`
```
/dev/mmcblk0p71   55G   21G   32G  40% /
```

### 2026-11-02 10:38 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN (1.2G)
```
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   22G   32G  41% /
```

### 2026-11-02 13:31 UTC — DELETE (quality suite done): `models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN`
```
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-02 13:31 UTC — DELETE (quality suite done): `models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text`
```
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-02 13:32 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN (2.5G)
```
42d970bb752e9d87d40b1f16ac2a349d69530870c5fcc86de8ce0036c3d3595d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-02 20:17 UTC — DELETE (quality suite done): `models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN`
```
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-02 20:17 UTC — DELETE (quality suite done): `models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text`
```
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-02 20:34 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-9B-MNN (4.7G)
```
896bf5a7b7e164186f1c0d065cb6d92306c9aa9c3cb62047531e24fa08c033af  models/Qwen3.5-9B-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   26G   28G  48% /
```

### 2026-11-03 09:20 UTC — DELETE (quality suite done): `models/Qwen3.5-9B-MNN`
```
/dev/mmcblk0p71   55G   26G   28G  48% /
```

### 2026-11-03 09:21 UTC — DELETE (quality suite done): `models/Qwen3.5-9B-MNN-text`
```
/dev/mmcblk0p71   55G   26G   28G  48% /
```

### 2026-11-03 09:21 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-MNN (28K)
```
f36668ddf22403a332f978057d527cf285b01468bc3431b04094a7bafa6aba59  models/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   21G   33G  39% /
```

### 2026-11-03 11:41 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN (1.2G)
```
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-03 11:43 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN (2.5G)
```
42d970bb752e9d87d40b1f16ac2a349d69530870c5fcc86de8ce0036c3d3595d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   24G   29G  46% /
```

### 2026-11-03 12:29 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN (1.2G)
```
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   24G   29G  46% /
```

### 2026-11-03 12:29 UTC — DELETE (quality suite done): `models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN`
```
/dev/mmcblk0p71   55G   24G   29G  46% /
```

### 2026-11-03 12:29 UTC — DELETE (quality suite done): `models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text`
```
/dev/mmcblk0p71   55G   24G   29G  46% /
```

### 2026-11-03 12:29 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN (2.5G)
```
42d970bb752e9d87d40b1f16ac2a349d69530870c5fcc86de8ce0036c3d3595d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-03 12:30 UTC — DELETE (quality suite done): `models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN`
```
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-03 12:30 UTC — DELETE (quality suite done): `models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text`
```
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-03 12:45 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN (1.2G)
```
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-03 12:45 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN (1.2G)
```
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-03 12:48 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN (1.2G)
```
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-03 20:17 UTC — DELETE (quality suite done): `models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN`
```
92853033efe602f95efca3e1c05cd8b108f973c8beed417843a9671f8147ed8d  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/config.json
2454c9d07451b9debc0a7c732846a63a5493db18f514bb0aa4f3eeb28543ef46  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/export_args.json
ba0b3ec3d7eb600f4cfbac62d765f0e4427bef79c15b9fda9e45649a3b16c167  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm_config.json
aac90290bde6d7746b0df08f06582a6004747c0c3ab5760ce652a6eaad6d2d61  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn
7a26766882e7ab40570a1af7eb6090408e21b4cdd59fbe448cef47d3b6a28059  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.json
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
1.2G	models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-03 20:17 UTC — DELETE (quality suite done): `models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text`
```
1c1fac0e7490302ca10f7328dd797ae839cf191cd45331b909b37874855a8ade  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/config.json
2454c9d07451b9debc0a7c732846a63a5493db18f514bb0aa4f3eeb28543ef46  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/export_args.json
721196a1f3cfc88dcbe30e09d1f127021bf8f7fdc5d278cb4bdfb318cf1c5f07  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm_config.json
aac90290bde6d7746b0df08f06582a6004747c0c3ab5760ce652a6eaad6d2d61  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn
7a26766882e7ab40570a1af7eb6090408e21b4cdd59fbe448cef47d3b6a28059  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn.json
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn.weight
1.2G	models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-03 20:28 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN (2.5G)
```
42d970bb752e9d87d40b1f16ac2a349d69530870c5fcc86de8ce0036c3d3595d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-04 10:08 UTC — DELETE (quality suite done): `models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN`
```
92853033efe602f95efca3e1c05cd8b108f973c8beed417843a9671f8147ed8d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/config.json
e911b4dec9c9d4b50efd91b6a9097bc10d84f8952feab3be4091f23c0a256095  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/export_args.json
c638aaea7eecc3f84093488882f98628fd04b676856bb539ebc018a1be710ac6  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm_config.json
3f938a204480266ca48be0a2073226cf639b1aa04a56fce032f913c95196a46e  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn
7dde7f9c0a68756b4ac96cc91991710ca5206baf79b2a73face7d2cfc9e73e08  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.json
42d970bb752e9d87d40b1f16ac2a349d69530870c5fcc86de8ce0036c3d3595d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
2.5G	models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-04 10:09 UTC — DELETE (quality suite done): `models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text`
```
1c1fac0e7490302ca10f7328dd797ae839cf191cd45331b909b37874855a8ade  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/config.json
e911b4dec9c9d4b50efd91b6a9097bc10d84f8952feab3be4091f23c0a256095  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/export_args.json
9d45053fd2ddd12e25dac5b9186b5d3f5dc25c54475f2b26d3f9f89e969f07fe  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm_config.json
3f938a204480266ca48be0a2073226cf639b1aa04a56fce032f913c95196a46e  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn
7dde7f9c0a68756b4ac96cc91991710ca5206baf79b2a73face7d2cfc9e73e08  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn.json
42d970bb752e9d87d40b1f16ac2a349d69530870c5fcc86de8ce0036c3d3595d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn.weight
2.5G	models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-04 11:41 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN (1.2G)
```
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-04 14:43 UTC — DELETE (quality suite done): `models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN`
```
92853033efe602f95efca3e1c05cd8b108f973c8beed417843a9671f8147ed8d  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/config.json
2454c9d07451b9debc0a7c732846a63a5493db18f514bb0aa4f3eeb28543ef46  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/export_args.json
ba0b3ec3d7eb600f4cfbac62d765f0e4427bef79c15b9fda9e45649a3b16c167  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm_config.json
aac90290bde6d7746b0df08f06582a6004747c0c3ab5760ce652a6eaad6d2d61  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn
7a26766882e7ab40570a1af7eb6090408e21b4cdd59fbe448cef47d3b6a28059  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.json
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
1.2G	models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-04 14:43 UTC — DELETE (quality suite done): `models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text`
```
1c1fac0e7490302ca10f7328dd797ae839cf191cd45331b909b37874855a8ade  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/config.json
2454c9d07451b9debc0a7c732846a63a5493db18f514bb0aa4f3eeb28543ef46  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/export_args.json
721196a1f3cfc88dcbe30e09d1f127021bf8f7fdc5d278cb4bdfb318cf1c5f07  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm_config.json
aac90290bde6d7746b0df08f06582a6004747c0c3ab5760ce652a6eaad6d2d61  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn
7a26766882e7ab40570a1af7eb6090408e21b4cdd59fbe448cef47d3b6a28059  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn.json
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn.weight
1.2G	models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-04 14:45 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN (2.5G)
```
42d970bb752e9d87d40b1f16ac2a349d69530870c5fcc86de8ce0036c3d3595d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-04 16:19 UTC — DELETE (quality suite done): `models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN`
```
92853033efe602f95efca3e1c05cd8b108f973c8beed417843a9671f8147ed8d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/config.json
e911b4dec9c9d4b50efd91b6a9097bc10d84f8952feab3be4091f23c0a256095  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/export_args.json
c638aaea7eecc3f84093488882f98628fd04b676856bb539ebc018a1be710ac6  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm_config.json
3f938a204480266ca48be0a2073226cf639b1aa04a56fce032f913c95196a46e  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn
7dde7f9c0a68756b4ac96cc91991710ca5206baf79b2a73face7d2cfc9e73e08  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.json
42d970bb752e9d87d40b1f16ac2a349d69530870c5fcc86de8ce0036c3d3595d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
2.5G	models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-04 16:19 UTC — DELETE (quality suite done): `models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text`
```
1c1fac0e7490302ca10f7328dd797ae839cf191cd45331b909b37874855a8ade  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/config.json
e911b4dec9c9d4b50efd91b6a9097bc10d84f8952feab3be4091f23c0a256095  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/export_args.json
9d45053fd2ddd12e25dac5b9186b5d3f5dc25c54475f2b26d3f9f89e969f07fe  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm_config.json
3f938a204480266ca48be0a2073226cf639b1aa04a56fce032f913c95196a46e  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn
7dde7f9c0a68756b4ac96cc91991710ca5206baf79b2a73face7d2cfc9e73e08  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn.json
42d970bb752e9d87d40b1f16ac2a349d69530870c5fcc86de8ce0036c3d3595d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn.weight
2.5G	models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-05 01:31 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN (1.2G)
```
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-05 01:39 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN (1.2G)
```
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-05 02:11 UTC — DELETE (quality suite done): `models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN`
```
92853033efe602f95efca3e1c05cd8b108f973c8beed417843a9671f8147ed8d  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/config.json
2454c9d07451b9debc0a7c732846a63a5493db18f514bb0aa4f3eeb28543ef46  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/export_args.json
ba0b3ec3d7eb600f4cfbac62d765f0e4427bef79c15b9fda9e45649a3b16c167  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm_config.json
aac90290bde6d7746b0df08f06582a6004747c0c3ab5760ce652a6eaad6d2d61  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn
7a26766882e7ab40570a1af7eb6090408e21b4cdd59fbe448cef47d3b6a28059  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.json
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
1.2G	models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-05 02:11 UTC — DELETE (quality suite done): `models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text`
```
1c1fac0e7490302ca10f7328dd797ae839cf191cd45331b909b37874855a8ade  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/config.json
2454c9d07451b9debc0a7c732846a63a5493db18f514bb0aa4f3eeb28543ef46  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/export_args.json
721196a1f3cfc88dcbe30e09d1f127021bf8f7fdc5d278cb4bdfb318cf1c5f07  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm_config.json
aac90290bde6d7746b0df08f06582a6004747c0c3ab5760ce652a6eaad6d2d61  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn
7a26766882e7ab40570a1af7eb6090408e21b4cdd59fbe448cef47d3b6a28059  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn.json
e01b6e410837fee107466a228717b0e06e8d0b62398c3540bfdd4994977896ba  models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn.weight
1.2G	models/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text
/dev/mmcblk0p71   55G   22G   31G  41% /
```

### 2026-11-05 02:13 UTC — DOWNLOAD: taobao-mnn/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN (2.5G)
```
42d970bb752e9d87d40b1f16ac2a349d69530870c5fcc86de8ce0036c3d3595d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-05 03:30 UTC — DELETE (quality suite done): `models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN`
```
92853033efe602f95efca3e1c05cd8b108f973c8beed417843a9671f8147ed8d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/config.json
e911b4dec9c9d4b50efd91b6a9097bc10d84f8952feab3be4091f23c0a256095  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/export_args.json
c638aaea7eecc3f84093488882f98628fd04b676856bb539ebc018a1be710ac6  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm_config.json
3f938a204480266ca48be0a2073226cf639b1aa04a56fce032f913c95196a46e  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn
7dde7f9c0a68756b4ac96cc91991710ca5206baf79b2a73face7d2cfc9e73e08  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.json
42d970bb752e9d87d40b1f16ac2a349d69530870c5fcc86de8ce0036c3d3595d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN/llm.mnn.weight
2.5G	models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-11-05 03:30 UTC — DELETE (quality suite done): `models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text`
```
1c1fac0e7490302ca10f7328dd797ae839cf191cd45331b909b37874855a8ade  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/config.json
e911b4dec9c9d4b50efd91b6a9097bc10d84f8952feab3be4091f23c0a256095  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/export_args.json
9d45053fd2ddd12e25dac5b9186b5d3f5dc25c54475f2b26d3f9f89e969f07fe  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm_config.json
3f938a204480266ca48be0a2073226cf639b1aa04a56fce032f913c95196a46e  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn
7dde7f9c0a68756b4ac96cc91991710ca5206baf79b2a73face7d2cfc9e73e08  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn.json
42d970bb752e9d87d40b1f16ac2a349d69530870c5fcc86de8ce0036c3d3595d  models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text/llm.mnn.weight
2.5G	models/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN-text
/dev/mmcblk0p71   55G   23G   30G  44% /
```
