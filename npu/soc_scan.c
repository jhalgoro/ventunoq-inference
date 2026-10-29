// Probe: create a QNN HTP backend + device via the official interface table.
// Run under libsocspoof.so with FAKE_SOC_ID to find a soc_id whose
// SnapdragonModel passes the HTP startup whitelist (QCS8275 = model 82, blocked).
#include <dlfcn.h>
#include <stdio.h>
#include <QNN/QnnInterface.h>

int main(void) {
  void *h = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL);
  if (!h) { printf("dlopen fail: %s\n", dlerror()); return 2; }
  Qnn_ErrorHandle_t (*getProviders)(const QnnInterface_t ***, uint32_t *) =
      (Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *))
          dlsym(h, "QnnInterface_getProviders");
  if (!getProviders) { printf("no getProviders\n"); return 2; }
  const QnnInterface_t **prov = NULL;
  uint32_t n = 0;
  if (getProviders(&prov, &n) != QNN_SUCCESS || !n) { printf("getProviders FAIL\n"); return 2; }
  QNN_INTERFACE_VER_TYPE api = prov[0]->QNN_INTERFACE_VER_NAME;
  if (!api.backendCreate) { printf("no backendCreate\n"); return 2; }
  Qnn_BackendHandle_t be = NULL;
  Qnn_ErrorHandle_t rc = api.backendCreate(NULL, NULL, &be);
  if (rc) { printf("backend FAIL 0x%lx\n", (unsigned long)rc); return 1; }
  if (api.deviceCreate) {
    Qnn_DeviceHandle_t dev = NULL;
    rc = api.deviceCreate(NULL, NULL, &dev);
    if (rc) { printf("device FAIL 0x%lx\n", (unsigned long)rc); return 1; }
    printf("OK-DEVICE\n");
    return 0;
  }
  printf("OK-BACKEND-ONLY\n");
  return 0;
}
