# malloc-realloc-free
How to reimplement malloc realloc and free to use managed memory

## Usage

In code using malloc/realloc/free functions, `#include <cuda_managed_mem.h>`, and replace with

 * `malloc` with `cuda_managed_mem_malloc`
 * `free` with `cuda_managed_mem_free`
 * `realloc` with `cuda_managed_mem_realloc`

Then, compile `cuda_managed_mem.cu` and link it with main binary.


When compiling with `-DUSEREALLOC=1`, regular system calls are used. When compiling with `-DUSERALLOC=0` or undefined, then cuda managed memory is used.

Logging troubleshooting information can be done defining `-DLOGGING=1` when compiling `cuda_managed_mem.cu`.