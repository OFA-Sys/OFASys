#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <vector>

inline void assert_with_message(bool ok, const char* msg, ...) {
    if (!ok) {
        va_list args;
        va_start(args, msg);
        vfprintf(stderr, msg, args);
        va_end(args);
        exit(-1);
    }
}

inline void assert_with_errno(bool ok, const char* prefix) {
    if (!ok) {
        fprintf(stderr, "%s: %s\n", prefix, strerror(errno));
        exit(-1);
    }
}

#define MAX_MEMORY_USAGE (1<<30)
#define MIN(a, b) (a < b ? a : b)

std::vector<off_t> line_locater(const char* file_path) {
    int fd = open(file_path, O_RDONLY, 0);
    assert_with_errno(fd >= 0, "open error");

    off_t size = lseek(fd, 0, SEEK_END);
    assert_with_errno(size >= 0, "lseek error");

    if (size == 0) {
        int ret = close(fd);
        assert_with_errno(ret == 0, "close error");
        return std::vector<off_t>();
    }

    size_t cnt = 0;
    std::vector<off_t> pos;
    pos.push_back(0);

    off_t epochs = (size + MAX_MEMORY_USAGE - 1) / MAX_MEMORY_USAGE;
    off_t offset = 0;
    for (off_t e = 0; e < epochs; e++) {
        off_t local_size = MIN(size - offset, MAX_MEMORY_USAGE);
        assert_with_message(local_size <= MAX_MEMORY_USAGE,
                            "local size error %d\n", local_size);

        char* data = (char*) mmap(
                NULL, local_size, PROT_READ, MAP_PRIVATE, fd, offset);
        assert_with_errno(data != NULL, "mmap error");

        for (off_t i = 0; i < local_size; i++)
            if (data[i] == '\n') {
                cnt++;
                pos.push_back(offset + i + 1);
            }

        int ret = munmap(data, local_size);
        assert_with_errno(ret == 0, "munmap error");

        offset += MAX_MEMORY_USAGE;
    }
    // if (pos[pos.size() - 1] >= size)
    pos.pop_back();
    assert_with_message(pos.size() == cnt, "line count error %lu vs %lu\n", pos.size(), cnt);

    int ret = close(fd);
    assert_with_errno(ret == 0, "close error");

    return pos;
}

int main(int argc, char* argv[]) {
    assert_with_message(argc == 2, "Usage: %s [file_name]\n", argv[0]);
    char* file_path = argv[1];
    std::vector<off_t> pos = line_locater(file_path);

    size_t cnt = pos.size();
    printf("%zu\n", pos.size());
    if (cnt == 0)
        return 0;

    printf("%jd", (intmax_t)pos[0]);
    for (size_t i = 1; i < cnt; i++)
        printf(" %jd", (intmax_t)pos[i]);
    printf("\n");
}
