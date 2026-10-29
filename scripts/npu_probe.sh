#!/bin/bash
# NPU / accelerator hardware probe for the Arduino UNO Q (QCS8275/QCS8300 "Monza").
# Reproducible evidence for REPORT.md §7 (NPU feasibility). No root needed.
echo "== SoC =="
cat /sys/devices/soc0/machine /sys/devices/soc0/soc_id 2>/dev/null
tr '\0' ' ' </sys/firmware/devicetree/base/compatible; echo
echo "== remoteproc (DSPs) =="
for r in /sys/class/remoteproc/remoteproc*; do
  echo "  $(basename $r): $(cat $r/name) state=$(cat $r/state)"
done
echo "== FastRPC nodes (NPU transport) =="
ls -lL /dev/fastrpc-* 2>/dev/null
python3 - <<'EOF'
import os
for d in ["/dev/fastrpc-cdsp", "/dev/fastrpc-gdsp0", "/dev/dri/renderD128"]:
    try:
        os.close(os.open(d, os.O_RDWR)); print(f"  {d}: OPEN-OK")
    except OSError as e:
        print(f"  {d}: {e}")
EOF
echo "== GPU =="
ls /sys/bus/platform/devices | grep -E 'gpu|kgsl'
echo "  renderD128 driver: $(readlink -f /sys/class/drm/renderD128/device/driver 2>/dev/null)  (msm_dpu => display only, Turnip/Vulkan cannot bind KGSL)"
echo "== software stack =="
echo "  llama.cpp $(ls ~/Code/ventunoq-llamacpp/llama.cpp/libggml-cpu-*.so | wc -l) CPU backends, no HTP backend (none exists upstream)"
python3 -m pip show qnnruntime 2>/dev/null | grep -E '^(Name|Version)'
python3 -m pip show qairt-dev 2>/dev/null | grep -E '^(Name|Version)'
ls ~/Code/ventunoq-llamacpp/npu/qairt-sdk 2>/dev/null | head -5
