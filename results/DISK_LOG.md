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

### 2026-10-30 14:15 UTC — REUSE (LFM3-17B-QAD): `/tmp/opencode/dl/LFM3-1.7B-QAD-Q4_0.gguf`
```
43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452  /tmp/opencode/dl/LFM3-1.7B-QAD-Q4_0.gguf
```
```
/dev/mmcblk0p71   55G   24G   29G  45% /
```

### 2026-10-30 14:22 UTC — DELETE (tests finished): `/tmp/opencode/dl/LFM3-1.7B-QAD-Q4_0.gguf`
```
43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452  /tmp/opencode/dl/LFM3-1.7B-QAD-Q4_0.gguf
```
```
/dev/mmcblk0p71   55G   24G   29G  45% /
```

### 2026-10-30 14:23 UTC — DOWNLOAD (LFM3-17B-Q6K): `/tmp/opencode/dl/LFM3-1.7B-Q6_K.gguf`
```
54e1bd947c77167823d45bf495f1c3e2d226ca7fb242f83cfa141ee3a6ab55c6  /tmp/opencode/dl/LFM3-1.7B-Q6_K.gguf
```
```
/dev/mmcblk0p71   55G   24G   29G  46% /
```

### 2026-10-30 14:34 UTC — DELETE (tests finished): `/tmp/opencode/dl/LFM3-1.7B-Q6_K.gguf`
```
54e1bd947c77167823d45bf495f1c3e2d226ca7fb242f83cfa141ee3a6ab55c6  /tmp/opencode/dl/LFM3-1.7B-Q6_K.gguf
```
```
/dev/mmcblk0p71   55G   24G   29G  46% /
```

### 2026-10-30 14:35 UTC — DOWNLOAD (LFM3-17B-Q80): `/tmp/opencode/dl/LFM3-1.7B-Q8_0.gguf`
```
8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6  /tmp/opencode/dl/LFM3-1.7B-Q8_0.gguf
```
```
/dev/mmcblk0p71   55G   25G   29G  47% /
```

### 2026-10-30 14:44 UTC — DELETE (tests finished): `/tmp/opencode/dl/LFM3-1.7B-Q8_0.gguf`
```
8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6  /tmp/opencode/dl/LFM3-1.7B-Q8_0.gguf
```
```
/dev/mmcblk0p71   55G   25G   29G  47% /
```

### 2026-10-30 14:46 UTC — DOWNLOAD (LFM3-17B-F16): `/tmp/opencode/dl/LFM3-1.7B-F16.gguf`
```
b7ab8b69a85e4dfbf62cdfd0e02d7d051ed09f2c5b84c701771dddb073b85970  /tmp/opencode/dl/LFM3-1.7B-F16.gguf
```
```
/dev/mmcblk0p71   55G   26G   27G  50% /
```

### 2026-10-30 15:04 UTC — DELETE (tests finished): `/tmp/opencode/dl/LFM3-1.7B-F16.gguf`
```
b7ab8b69a85e4dfbf62cdfd0e02d7d051ed09f2c5b84c701771dddb073b85970  /tmp/opencode/dl/LFM3-1.7B-F16.gguf
```
```
/dev/mmcblk0p71   55G   26G   27G  50% /
```

### 2026-10-30 15:29 UTC — DELETE (tests finished): `/tmp/opencode/dl/LFM3-1.7B-Q4_K_M.gguf`
```
3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4  /tmp/opencode/dl/LFM3-1.7B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   23G   30G  43% /
```

### 2026-10-30 15:30 UTC — DOWNLOAD (LFM25-26B-Q4KM): `/tmp/opencode/dl/LFM2.5-2.6B-Q4_K_M.gguf`
```
02a8b7e17487d326e46d68ce0ba24211e1b80a14c4cd0597fa73c1cd697f52ed  /tmp/opencode/dl/LFM2.5-2.6B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   24G   30G  44% /
```

### 2026-10-30 15:44 UTC — DELETE (tests finished): `/tmp/opencode/dl/LFM2.5-2.6B-Q4_K_M.gguf`
```
02a8b7e17487d326e46d68ce0ba24211e1b80a14c4cd0597fa73c1cd697f52ed  /tmp/opencode/dl/LFM2.5-2.6B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   24G   30G  44% /
```

### 2026-10-30 15:45 UTC — DOWNLOAD (Qwen35-08B-Q4KM): `/tmp/opencode/dl/Qwen3.5-0.8B-Q4_K_M.gguf`
```
bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517  /tmp/opencode/dl/Qwen3.5-0.8B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   23G   31G  42% /
```

### 2026-10-30 15:51 UTC — DELETE (tests finished): `/tmp/opencode/dl/Qwen3.5-0.8B-Q4_K_M.gguf`
```
bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517  /tmp/opencode/dl/Qwen3.5-0.8B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   23G   31G  42% /
```

### 2026-10-30 15:51 UTC — DOWNLOAD (Qwen35-2B-Q4KM): `/tmp/opencode/dl/Qwen3.5-2B-Q4_K_M.gguf`
```
aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223  /tmp/opencode/dl/Qwen3.5-2B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-10-30 16:02 UTC — DELETE (tests finished): `/tmp/opencode/dl/Qwen3.5-2B-Q4_K_M.gguf`
```
aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223  /tmp/opencode/dl/Qwen3.5-2B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   23G   30G  44% /
```

### 2026-10-30 16:03 UTC — DOWNLOAD (Qwen35-4B-Q4KM): `/tmp/opencode/dl/Qwen3.5-4B-Q4_K_M.gguf`
```
00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4  /tmp/opencode/dl/Qwen3.5-4B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   25G   29G  46% /
```

### 2026-10-30 16:41 UTC — DELETE (tests finished): `/tmp/opencode/dl/Qwen3.5-4B-Q4_K_M.gguf`
```
00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4  /tmp/opencode/dl/Qwen3.5-4B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   25G   29G  46% /
```

### 2026-10-30 16:43 UTC — DOWNLOAD (Qwen35-4B-Q40): `/tmp/opencode/dl/Qwen3.5-4B-Q4_0.gguf`
```
298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb  /tmp/opencode/dl/Qwen3.5-4B-Q4_0.gguf
```
```
/dev/mmcblk0p71   55G   24G   29G  46% /
```

### 2026-10-30 17:39 UTC — DELETE (tests finished): `/tmp/opencode/dl/Qwen3.5-4B-Q4_0.gguf`
```
298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb  /tmp/opencode/dl/Qwen3.5-4B-Q4_0.gguf
```
```
/dev/mmcblk0p71   55G   24G   29G  46% /
```

### 2026-10-30 17:42 UTC — DOWNLOAD (Qwen35-9B-Q4KM): `/tmp/opencode/dl/Qwen3.5-9B-Q4_K_M.gguf`
```
03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8  /tmp/opencode/dl/Qwen3.5-9B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   27G   26G  52% /
```

### 2026-10-30 18:26 UTC — DELETE (tests finished): `/tmp/opencode/dl/Qwen3.5-9B-Q4_K_M.gguf`
```
03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8  /tmp/opencode/dl/Qwen3.5-9B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   27G   26G  52% /
```

### 2026-10-30 18:29 UTC — DOWNLOAD (Qwen35-9B-Q40): `/tmp/opencode/dl/Qwen3.5-9B-Q4_0.gguf`
```
17670346b4260ddcb0173965145155885024f3c9a4a24389a3370751edbcde24  /tmp/opencode/dl/Qwen3.5-9B-Q4_0.gguf
```
```
/dev/mmcblk0p71   55G   27G   26G  51% /
```

### 2026-10-30 19:01 UTC — DELETE (tests finished): `/tmp/opencode/dl/Qwen3.5-9B-Q4_0.gguf`
```
17670346b4260ddcb0173965145155885024f3c9a4a24389a3370751edbcde24  /tmp/opencode/dl/Qwen3.5-9B-Q4_0.gguf
```
```
/dev/mmcblk0p71   55G   27G   26G  51% /
```

### 2026-10-30 19:03 UTC — DOWNLOAD (K2H-4B-Q4KM): `/tmp/opencode/dl/K2-Horizon-4B-Q4_K_M.gguf`
```
07773aed93890d4f5d08b421430000d5c8d73cefc74fbdc32e67d820f23207b2  /tmp/opencode/dl/K2-Horizon-4B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   25G   29G  47% /
```

### 2026-10-30 20:17 UTC — DELETE (tests finished): `/tmp/opencode/dl/K2-Horizon-4B-Q4_K_M.gguf`
```
07773aed93890d4f5d08b421430000d5c8d73cefc74fbdc32e67d820f23207b2  /tmp/opencode/dl/K2-Horizon-4B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   25G   29G  47% /
```

### 2026-10-30 20:19 UTC — DOWNLOAD (K2H-4B-Q40): `/tmp/opencode/dl/K2-Horizon-3.7B-Q4_0.gguf`
```
6edf6a91684f08a654693d71bfb921e4856b10e5bd48583cba8cfe33e5c3b3a0  /tmp/opencode/dl/K2-Horizon-3.7B-Q4_0.gguf
```
```
/dev/mmcblk0p71   55G   25G   29G  47% /
```

### 2026-10-30 20:56 UTC — DELETE (tests finished): `/tmp/opencode/dl/K2-Horizon-3.7B-Q4_0.gguf`
```
6edf6a91684f08a654693d71bfb921e4856b10e5bd48583cba8cfe33e5c3b3a0  /tmp/opencode/dl/K2-Horizon-3.7B-Q4_0.gguf
```
```
/dev/mmcblk0p71   55G   25G   29G  47% /
```

### 2026-10-30 20:59 UTC — DOWNLOAD (K2H-7B-Q4KM): `/tmp/opencode/dl/K2-Horizon-7B-Q4_K_M.gguf`
```
eb89c15a0ae9712be2ee462bf43802de14200f20f93b73da6eb68c2ebdd28e4e  /tmp/opencode/dl/K2-Horizon-7B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   27G   26G  51% /
```

### 2026-10-30 21:52 UTC — DELETE (tests finished): `/tmp/opencode/dl/K2-Horizon-7B-Q4_K_M.gguf`
```
eb89c15a0ae9712be2ee462bf43802de14200f20f93b73da6eb68c2ebdd28e4e  /tmp/opencode/dl/K2-Horizon-7B-Q4_K_M.gguf
```
```
/dev/mmcblk0p71   55G   27G   26G  51% /
```
