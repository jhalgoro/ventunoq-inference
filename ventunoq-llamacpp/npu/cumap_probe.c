/* cumap_probe.c — replicates llama.cpp Hexagon backend buffer registration:
 * repeated (FASTRPC_IOCTL_ALLOC_DMA_BUFF + MEM_MAP) of 256 MiB dma-bufs in one
 * CDSP session, held open, until failure. Finds the real per-session ceiling:
 * DSP IOMMU aperture (32-bit, "DMA64 disabled") vs RLIMIT_MEMLOCK (1.9 GiB).
 * This is the path that failed for K2-Horizon-4B at fd 19 / size 561401856.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <stdint.h>

#define FASTRPC_IOCTL_ALLOC_DMA_BUFF _IOWR('R', 1, struct fastrpc_ioctl_alloc_dma_buf)
#define FASTRPC_IOCTL_INIT_ATTACH    _IO('R', 4)
#define FASTRPC_IOCTL_MEM_MAP        _IOWR('R', 10, struct fastrpc_ioctl_mem_map)
#define FASTRPC_IOCTL_MMAP           _IOWR('R', 6, struct fastrpc_ioctl_req_mmap)
struct fastrpc_ioctl_alloc_dma_buf { __s32 fd; __u32 flags; __u64 size; };
struct fastrpc_ioctl_mem_map {
    __s32 version; __s32 fd; __s32 offset; __u32 flags;
    __u64 vaddrin; __u64 length; __u64 vaddrout;
    __s32 attrs; __s32 reserved[4];
};
struct fastrpc_ioctl_req_mmap { __s32 fd; __u32 flags; __u64 vaddrin; __u64 size; __u64 vaddrout; };
#define REMOTE_MAP_UNCACHED (1 << 0)

int main(int argc, char **argv) {
    size_t chunk = (argc > 1 ? (size_t)atol(argv[1]) : 256UL) << 20;
    int maxn = argc > 2 ? atoi(argv[2]) : 24;
    int cfd = open("/dev/fastrpc-cdsp", O_RDWR);
    if (cfd < 0) { perror("open /dev/fastrpc-cdsp"); return 1; }
    if (ioctl(cfd, FASTRPC_IOCTL_INIT_ATTACH) < 0) perror("INIT_ATTACH");

    size_t total = 0;
    int i;
    for (i = 0; i < 24; i++) {
        struct fastrpc_ioctl_alloc_dma_buf a;
        memset(&a, 0, sizeof(a));
        a.size = chunk; a.flags = O_RDWR;
        if (ioctl(cfd, FASTRPC_IOCTL_ALLOC_DMA_BUFF, &a) < 0) {
            printf("chunk %2d ALLOC_DMA_BUFF failed @cumulative %zu MiB: errno=%d (%s)\n",
                   i, total >> 20, errno, strerror(errno));
            break;
        }
        struct fastrpc_ioctl_req_mmap m; /* legacy MMAP(6) — the call ggml-hex uses */
        memset(&m, 0, sizeof(m));
        m.fd = a.fd; m.flags = 0; m.vaddrin = 0; m.size = chunk;
        int rc = ioctl(cfd, FASTRPC_IOCTL_MMAP, &m);
        if (rc < 0) {
            printf("chunk %2d MMAP failed @cumulative %zu MiB: rc=%d (0x%x) errno=%d (%s)\n",
                   i, total >> 20, rc, (unsigned)rc, errno, strerror(errno));
            break;
        }
        total += chunk;
        printf("chunk %2d ok  DSP VA 0x%08llx  cumulative %4zu MiB\n",
               i, (unsigned long long)m.vaddrout, total >> 20);
    }
    printf("TOTAL mapped: %zu MiB (%d chunks)\n", total >> 20, i);
    return 0;
}
