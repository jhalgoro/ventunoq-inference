/* pin_probe.c — replicates the LEGACY fastrpc_mmap path used by llama.cpp's
 * Hexagon backend (ggml-hex: fastrpc_mmap failed): pins anonymous user pages
 * via FASTRPC_IOCTL_MMAP until the kernel refuses, to test whether the board's
 * RLIMIT_MEMLOCK (~1.9 GiB) is what killed the K2-Horizon-4B NPU load.
 * The kernel's fastrpc driver pins user pages (pin_user_pages) and enforces
 * RLIMIT_MEMLOCK, returning -EPERM when exceeded.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <linux/types.h>
#include <stdint.h>

#define FASTRPC_IOCTL_INIT_ATTACH _IO('R', 4)
struct fastrpc_ioctl_req_mmap   { __s32 fd; __u32 flags; __u64 vaddrin; __u64 size; __u64 vaddrout; };
struct fastrpc_ioctl_req_munmap { __u64 vaddrout; __u64 size; };
#define FASTRPC_IOCTL_MMAP   _IOWR('R', 6, struct fastrpc_ioctl_req_mmap)
#define FASTRPC_IOCTL_MUNMAP _IOWR('R', 7, struct fastrpc_ioctl_req_munmap)

int main(void) {
    struct rlimit rl;
    getrlimit(RLIMIT_MEMLOCK, &rl);
    printf("RLIMIT_MEMLOCK soft=%llu hard=%llu (KiB)\n",
           rl.rlim_cur / 1024, rl.rlim_max / 1024);

    int cfd = open("/dev/fastrpc-cdsp", O_RDWR);
    if (cfd < 0) { perror("open /dev/fastrpc-cdsp"); return 1; }
    if (ioctl(cfd, FASTRPC_IOCTL_INIT_ATTACH) < 0) perror("INIT_ATTACH");

    const size_t chunk = 256UL << 20; /* 256 MiB */
    size_t total = 0;
    for (int i = 0; i < 40; i++) {
        char *p = mmap(NULL, chunk, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { printf("host mmap failed at chunk %d\n", i); break; }
        memset(p, 1, chunk); /* fault pages in — pinning needs present pages */

        struct fastrpc_ioctl_req_mmap m = {0};
        m.fd = -1; /* legacy userspace-pointer path */
        m.flags = 0;
        m.vaddrin = (uint64_t)(uintptr_t)p;
        m.size = chunk;
        int rc = ioctl(cfd, FASTRPC_IOCTL_MMAP, &m);
        if (rc) {
            printf("chunk %2d (256 MiB, cumulative %zu MiB) -> rc=%d errno=%d (%s)\n",
                   i, total >> 20, rc, errno, strerror(errno));
            break;
        }
        total += chunk;
        printf("chunk %2d ok, DSP VA 0x%llx, cumulative %zu MiB\n",
               i, (unsigned long long)m.vaddrout, total >> 20);
    }
    printf("TOTAL pinned: %zu MiB\n", total >> 20);
    return 0;
}
