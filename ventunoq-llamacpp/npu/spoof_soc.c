// LD_PRELOAD shim: QNN HTP gates on a SoC-model whitelist read from
// /sys/devices/soc0/soc_id. QCS8275 (soc_id 675 -> SnapdragonModel 82) is
// functionally Hexagon V75 but not in QAIRT 2.51's list, so present the
// soc_id of a whitelisted V75 part (Snapdragon 8 Gen 3, soc_id 579) to the
// HTP backend's host-side startup check. Benchmark use only.
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <dlfcn.h>

static const char *kTargets[] = {
    "/sys/devices/soc0/soc_id",
    "/sys/devices/soc0/machine",
    "/sys/devices/soc0/raw_version",
    "/sys/devices/soc0/hw_platform",
    "/sys/devices/soc0/platform_subtype",
    "/sys/devices/soc0/jtag_id",
    "/sys/devices/soc0/feature_vector",
    "/sys/class/misc/socinfo_reg_dump",
};
static const char *kFake[] = {
    "579\n",                 // soc_id: SD 8 Gen 3 (Hexagon V75, whitelisted)
    "Snapdragon Pruneo\n",   // machine
    "1\n",
    "QRD\n",
    "0\n",
    "0\n",
    "0\n",
    "0\n",
};

static int is_target(const char *path, int *idx) {
    if (!path) return 0;
    for (unsigned i = 0; i < sizeof(kTargets)/sizeof(*kTargets); i++)
        if (!strcmp(path, kTargets[i])) { *idx = (int)i; return 1; }
    return 0;
}

static int fake_fd(int idx) {
    static const char *soc = NULL;
    int fd = (int)syscall(SYS_memfd_create, "socinfo", 2 /*MFD_CLOEXEC*/);
    if (fd < 0) return -1;
    if (idx == 0) {
        if (!soc) soc = getenv("FAKE_SOC_ID");
        char buf[32];
        snprintf(buf, sizeof buf, "%s\n", soc ? soc : "579");
        if (write(fd, buf, strlen(buf)) < 0) { }
    } else if (write(fd, kFake[idx], strlen(kFake[idx])) < 0) { }
    lseek(fd, 0, SEEK_SET);
    return fd;
}

#define REAL(name, ...) ((typeof(name)*)dlsym(RTLD_NEXT, #name))
static int (*real_openat)(int, const char*, int, ...);

// If the loader reads our memfd through the raw read() path, still hand out
// the fake content (strace showed the real file was NOT opened under preload,
// yet the printed model changed -> some readers use read()/pread64 on fds we
// never see; keep this hook so targeted probes can force the value).
__attribute__((unused)) static ssize_t (*real_read)(int, void *, size_t);
ssize_t read(int fd, void *buf, size_t n) {
  if (!real_read) real_read = (ssize_t (*)(int, void *, size_t))dlsym(RTLD_NEXT, "read");
  char link[64];
  if (snprintf(link, sizeof link, "/proc/self/fd/%d", fd) < 64) {
    char tgt[128];
    ssize_t m = readlink(link, tgt, sizeof tgt - 1);
    if (m > 0) {
      tgt[m] = 0;
      if (!strcmp(tgt, "/socinfo"))  // our memfd label
        return real_read(fd, buf, n);  // already fake
    }
  }
  return real_read(fd, buf, n);
}

int openat(int dirfd, const char *path, int flags, ...) {
    if (!real_openat) real_openat = dlsym(RTLD_NEXT, "openat");
    int idx;
    if (is_target(path, &idx)) return fake_fd(idx);
    mode_t mode = 0;
    if (flags & O_CREAT) { va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap); }
    return real_openat(dirfd, path, flags, mode);
}

int open(const char *path, int flags, ...) {
    int idx;
    if (is_target(path, &idx)) return fake_fd(idx);
    int (*r)(const char*, int, ...) = REAL(open);
    mode_t mode = 0;
    if (flags & O_CREAT) { va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap); }
    return r(path, flags, mode);
}

// fopen-based readers too
FILE *fopen(const char *path, const char *mode) {
    int idx;
    if (is_target(path, &idx)) {
        char p[64]; static int seq;
        snprintf(p, sizeof p, "/proc/self/fd/%d", fake_fd(idx));
        FILE *(*r)(const char*, const char*) = REAL(fopen);
        return r(p, "r");
    }
    FILE *(*r)(const char*, const char*) = REAL(fopen);
    return r(path, mode);
}
