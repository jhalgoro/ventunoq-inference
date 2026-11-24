/* mmap_probe.c — direct fastrpc ioctl probe for /dev/fastrpc-cdsp
 * Goal: see what the kernel returns for MEM_MAP of an external dma-buf
 * (the exact path that fails during QNN HTP graph execution).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/types.h>
#include <stdint.h>

#define FASTRPC_IOCTL_ALLOC_DMA_BUFF _IOWR('R', 1, struct fastrpc_ioctl_alloc_dma_buf)
#define FASTRPC_IOCTL_INIT_ATTACH    _IO('R', 4)
#define FASTRPC_IOCTL_MEM_MAP        _IOWR('R', 10, struct fastrpc_ioctl_mem_map)
#define FASTRPC_IOCTL_MMAP           _IOWR('R', 6, struct fastrpc_ioctl_req_mmap)
#define FASTRPC_IOCTL_MUNMAP         _IOWR('R', 7, struct fastrpc_ioctl_req_munmap)
#define FASTRPC_IOCTL_MEM_UNMAP      _IOWR('R', 11, struct fastrpc_ioctl_mem_unmap)
#define FASTRPC_IOCTL_GET_DSP_INFO   _IOWR('R', 13, struct fastrpc_ioctl_capability)

struct fastrpc_ioctl_alloc_dma_buf { __s32 fd; __u32 flags; __u64 size; };
struct fastrpc_ioctl_mem_map {
    __s32 version; __s32 fd; __s32 offset; __u32 flags;
    __u64 vaddrin; __u64 length; __u64 vaddrout;
    __s32 attrs; __s32 reserved[4];
};
struct fastrpc_ioctl_mem_unmap { __s32 version; __s32 fd; __u64 vaddr; __u64 length; __s32 reserved[5]; };
struct fastrpc_ioctl_req_mmap { __s32 fd; __u32 flags; __u64 vaddrin; __u64 size; __u64 vaddrout; };
struct fastrpc_ioctl_req_munmap { __u64 vaddrout; __u64 size; };
struct fastrpc_ioctl_capability { __u32 domain; __u32 attribute_id; __u32 capability; __u32 reserved[4]; };

#define REMOTE_MAP_UNCACHED (1 << 0)
#define REMOTE_MAP_CACHED   (1 << 1)

static int open_cdsp(void) {
    int fd = open("/dev/fastrpc-cdsp", O_RDWR);
    if (fd < 0) { perror("open /dev/fastrpc-cdsp"); exit(1); }
    if (ioctl(fd, FASTRPC_IOCTL_INIT_ATTACH) < 0)
        perror("INIT_ATTACH");
    return fd;
}

static void try_mem_map(int cfd, int buf_fd, size_t len, const char *tag, int heapfd_alloced) {
    struct fastrpc_ioctl_mem_map m;
    memset(&m, 0, sizeof(m));
    m.version = 1;
    m.fd = buf_fd;
    m.offset = 0;
    m.flags = REMOTE_MAP_UNCACHED;
    m.vaddrin = 0;
    m.length = len;
    m.attrs = 0;
    int rc = ioctl(cfd, FASTRPC_IOCTL_MEM_MAP, &m);
    printf("[%s] len=%zu ioctl_rc=%d errno=%d(%s) vaddrout=0x%llx\n",
           tag, len, rc, errno, strerror(errno), (unsigned long long)m.vaddrout);
    if (rc == 0) {
        struct fastrpc_ioctl_mem_unmap u = { .version = 1, .fd = buf_fd, .vaddr = m.vaddrout, .length = len };
        ioctl(cfd, FASTRPC_IOCTL_MEM_UNMAP, &u);
    }
    if (heapfd_alloced) close(buf_fd);
}

int main(void) {
    int cfd = open_cdsp();

    /* capability probe for sanity */
    struct fastrpc_ioctl_capability cap;
    memset(&cap, 0, sizeof(cap));
    cap.domain = 3; cap.attribute_id = 1; /* DSP SMMU state */
    int rc = ioctl(cfd, FASTRPC_IOCTL_GET_DSP_INFO, &cap);
    printf("GET_DSP_INFO(domain=3,attr=1) rc=%d errno=%d cap=0x%x\n", rc, errno, cap.capability);

    /* ---- Variant A: kernel-allocated dma-buf via ALLOC_DMA_BUF ---- */
    size_t len = 602368;
    struct fastrpc_ioctl_alloc_dma_buf a;
    memset(&a, 0, sizeof(a));
    a.size = len;
    a.flags = O_RDWR;
    rc = ioctl(cfd, FASTRPC_IOCTL_ALLOC_DMA_BUFF, &a);
    printf("ALLOC_DMA_BUF(%zu) rc=%d errno=%d(%s) fd=%d\n", len, rc, errno, strerror(errno), a.fd);
    if (rc == 0 && a.fd >= 0)
        try_mem_map(cfd, a.fd, len, "A kernel-alloc dma-buf", 1);

    /* ---- Variant B: dma-buf imported from /dev/dma_heap/<heap> ---- */
    struct dma_heap_allocation_data { __u64 len; __u32 fd; __u32 fd_flags; __u64 heap_flags; };
    const int DMA_HEAP_IOCTL_ALLOC = _IOWR('H', 0, struct dma_heap_allocation_data);
    const char *heaps[] = {"system", "qcom,system", "qcom,secure-non-pixel"};
    for (unsigned i = 0; i < 3; i++) {
        char path[128];
        snprintf(path, sizeof(path), "/dev/dma_heap/%s", heaps[i]);
        int hfd = open(path, O_RDWR);
        if (hfd < 0) { printf("[B %s] open failed: %s\n", heaps[i], strerror(errno)); continue; }
        struct dma_heap_allocation_data hd;
        memset(&hd, 0, sizeof(hd));
        hd.len = len;
        hd.fd_flags = O_RDWR | O_CLOEXEC;
        if (ioctl(hfd, DMA_HEAP_IOCTL_ALLOC, &hd) == 0) {
            try_mem_map(cfd, (int)hd.fd, len, heaps[i], 1);
        } else {
            printf("[B %s] heap alloc failed errno=%d(%s)\n", heaps[i], errno, strerror(errno));
        }
        close(hfd);
    }

    /* ---- Variant D: legacy MMAP ioctl (6) — remote_mmap path ---- */
    {
        struct fastrpc_ioctl_alloc_dma_buf b;
        memset(&b, 0, sizeof(b));
        b.size = len; b.flags = O_RDWR;
        if (ioctl(cfd, FASTRPC_IOCTL_ALLOC_DMA_BUFF, &b) == 0) {
            struct fastrpc_ioctl_req_mmap m;
            memset(&m, 0, sizeof(m));
            m.fd = b.fd; m.flags = 0; m.vaddrin = 0; m.size = len;
            rc = ioctl(cfd, FASTRPC_IOCTL_MMAP, &m);
            printf("[D legacy MMAP(6)] rc=%d errno=%d(%s) vaddrout=0x%llx\n",
                   rc, errno, strerror(errno), (unsigned long long)m.vaddrout);
            if (rc == 0) {
                struct fastrpc_ioctl_req_munmap u = { .vaddrout = m.vaddrout, .size = len };
                ioctl(cfd, FASTRPC_IOCTL_MUNMAP, &u);
            }
            close(b.fd);
        }
    }

    /* ---- Variant E: MEM_MAP with vaddrin=host VA (dspqueue-style) ---- */
    {
        struct fastrpc_ioctl_alloc_dma_buf b;
        memset(&b, 0, sizeof(b));
        b.size = len; b.flags = O_RDWR;
        if (ioctl(cfd, FASTRPC_IOCTL_ALLOC_DMA_BUFF, &b) == 0) {
            void *hv = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, b.fd, 0);
            struct fastrpc_ioctl_mem_map m;
            memset(&m, 0, sizeof(m));
            m.version = 1; m.fd = b.fd; m.flags = 2;
            m.vaddrin = (uint64_t)(uintptr_t)hv; m.length = len;
            rc = ioctl(cfd, FASTRPC_IOCTL_MEM_MAP, &m);
            printf("[E MEM_MAP vaddrin=hostVA] rc=%d errno=%d(%s) vaddrout=0x%llx\n",
                   rc, errno, strerror(errno), (unsigned long long)m.vaddrout);
            if (rc == 0) {
                struct fastrpc_ioctl_mem_unmap u = { .version = 1, .fd = b.fd, .vaddr = m.vaddrout, .length = len };
                ioctl(cfd, FASTRPC_IOCTL_MEM_UNMAP, &u);
            }
            if (hv != MAP_FAILED) munmap(hv, len);
            close(b.fd);
        }
    }

    /* ---- Variant F: MEM_MAP from a SECOND session fd (shim simulation) ---- */
    {
        int cfd2 = open("/dev/fastrpc-cdsp", O_RDWR);
        if (cfd2 >= 0) ioctl(cfd2, FASTRPC_IOCTL_INIT_ATTACH);
        struct fastrpc_ioctl_alloc_dma_buf b;
        memset(&b, 0, sizeof(b));
        b.size = len; b.flags = O_RDWR;
        if (cfd2 >= 0 && ioctl(cfd, FASTRPC_IOCTL_ALLOC_DMA_BUFF, &b) == 0) {
            void *hv = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, b.fd, 0);
            struct fastrpc_ioctl_mem_map m;
            memset(&m, 0, sizeof(m));
            m.version = 1; m.fd = b.fd; m.flags = 2;
            m.vaddrin = (uint64_t)(uintptr_t)hv; m.length = len;
            rc = ioctl(cfd2, FASTRPC_IOCTL_MEM_MAP, &m);
            printf("[F MEM_MAP second-session] rc=%d errno=%d(%s) vaddrout=0x%llx\n",
                   rc, errno, strerror(errno), (unsigned long long)m.vaddrout);
            if (rc == 0) {
                struct fastrpc_ioctl_mem_unmap u = { .version = 1, .fd = b.fd, .vaddr = m.vaddrout, .length = len };
                ioctl(cfd2, FASTRPC_IOCTL_MEM_UNMAP, &u);
            }
            if (hv != MAP_FAILED) munmap(hv, len);
            close(b.fd);
        }
        if (cfd2 >= 0) close(cfd2);
    }

    /* ---- Variant C: sizes sweep on kernel-allocated buffers ---- */
    size_t sizes[] = {4096, 65536, 262144, 602368, 1048576, 4194304};
    for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        struct fastrpc_ioctl_alloc_dma_buf b;
        memset(&b, 0, sizeof(b));
        b.size = sizes[i];
        b.flags = O_RDWR;
        rc = ioctl(cfd, FASTRPC_IOCTL_ALLOC_DMA_BUFF, &b);
        if (rc == 0 && b.fd >= 0) {
            try_mem_map(cfd, b.fd, sizes[i], "C sweep", 1);
        } else {
            printf("[C sweep] alloc %zu failed rc=%d errno=%d(%s)\n", sizes[i], rc, errno, strerror(errno));
        }
    }
    close(cfd);
    return 0;
}
