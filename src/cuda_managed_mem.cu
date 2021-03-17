/*

The MIT License (MIT)

Copyright (c) 2021 NVIDIA CORPORATION

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/


/*----------------------------------------------------------------------------
 * Standard C/C++ library headers
 *----------------------------------------------------------------------------*/
#include <assert.h>
#include <stdio.h>

#include <algorithm>
#include <list>

#include <cstdio>

/*----------------------------------------------------------------------------
 *  Local headers
 *----------------------------------------------------------------------------*/
#include "cuda_managed_mem.h"

#ifndef LOGGING
#define LOGGING 0
#endif

struct kv
{
    void* ptr;
    size_t size;
};

size_t sg_malloc_entry_count = 0 ;
kv* sg_malloc_entries = 0 ;

FILE* sg_log = 0 ;

int sg_malloc_has(void* ptr)
{
    for (size_t k = 0 ; k < sg_malloc_entry_count ; ++k)
    {
        if (sg_malloc_entries[k].ptr == ptr)
        {
            return 1;
        }
    }
    return 0;
}

void sg_malloc_put(void* ptr, size_t sz)
{
    #if LOGGING
    if (sg_log == 0) sg_log = fopen("nvidia-malloc.log", "w");
    fprintf(sg_log, "sg_malloc_put %p = %zu\n", ptr, sz);
    fclose(sg_log);
    #endif
    for (size_t k = 0 ; k < sg_malloc_entry_count ; ++k)
    {
        if (sg_malloc_entries[k].ptr == ptr)
        {
            sg_malloc_entries[k].size = sz;
            return;
        }
    }
    sg_malloc_entries = (kv*)realloc(sg_malloc_entries, (sg_malloc_entry_count+1)*sizeof(kv));
    sg_malloc_entries[sg_malloc_entry_count].ptr = ptr;
    sg_malloc_entries[sg_malloc_entry_count].size = sz;
    ++sg_malloc_entry_count;
}

size_t sg_malloc_get(void* ptr)
{
    #if LOGGING
    if (sg_log == 0) sg_log = fopen("nvidia-malloc.log", "w");
    fprintf(sg_log, "sg_malloc_get %p \n", ptr);
    fclose(sg_log);
    #endif
    for (size_t k = 0 ; k < sg_malloc_entry_count ; ++k)
    {
        if (sg_malloc_entries[k].ptr == ptr)
        {
            fprintf(sg_log, "\treturning sg_malloc_get %p = %zu \n", ptr, sg_malloc_entries[k].size);
            #if LOGGING
            fclose(sg_log);
            #endif
            return sg_malloc_entries[k].size;
        }
    }
    #if LOGGING
    fprintf(sg_log, "sg_malloc_get %p == NOT FOUND ==\n", ptr);
    fclose(sg_log);
    #endif
    return (size_t)0;
}

void sg_malloc_erase(void* ptr)
{
    #if LOGGING
    if (sg_log == 0) sg_log = fopen("nvidia-malloc.log", "w");
    fprintf(sg_log, "sg_malloc_erase %p \n", ptr);
    fclose(sg_log);
    #endif
    for (size_t k = 0 ; k < sg_malloc_entry_count ; ++k)
    {
        if (sg_malloc_entries[k].ptr == ptr)
        {
            // found !
            for (size_t j = k ; j < sg_malloc_entry_count-1 ; ++j)
            {
                sg_malloc_entries[j].ptr = sg_malloc_entries[j+1].ptr;
                sg_malloc_entries[j].size = sg_malloc_entries[j+1].size;
            }
            sg_malloc_entries = (kv*)realloc(sg_malloc_entries, (sg_malloc_entry_count-1)*sizeof(kv));
            --sg_malloc_entry_count;
            return;
        }
    }
}

#ifndef USEREALLOC
#define USEREALLOC 0
#endif

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

#define CUDA_CHECK_LINE(a,file,line) {                                          \
    cudaError_t __cuer = a;                                                     \
    if (cudaSuccess != __cuer) {                                                \
        ::fprintf (stderr, "[CUDA-ERRROR] @ %s:%d -- %d : %s -- running %s\n",  \
                file,line, __cuer, ::cudaGetErrorString(__cuer),#a) ;           \
        ::exit(__cuer) ;                                                        \
    }                                                                           \
}

#define CUDA_CHECK(a) CUDA_CHECK_LINE(a,__FILE__,__LINE__)

void cuda_managed_mem_malloc(void **pointer, size_t size)
{
    #if USEREALLOC
    *pointer = malloc(size);
    #else
    CUDA_CHECK(cudaMallocManaged (pointer, size, cudaMemAttachGlobal));
    // START ON HOST (seems that getcwd fails otherwise - System error: Bad address)
    CUDA_CHECK(cudaMemPrefetchAsync (*pointer, size, cudaCpuDeviceId, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    sg_malloc_put(*pointer, size);
    #endif
} 

void cuda_managed_mem_free(void *pointer){
    #if USEREALLOC
    free(pointer);
    #else
    CUDA_CHECK(cudaFree(pointer));
    CUDA_CHECK(cudaDeviceSynchronize());
    sg_malloc_erase(pointer);
    #endif
}

void* cuda_managed_mem_realloc(void *pointer, size_t size)
{
    #if USEREALLOC
    return realloc(pointer, size);
    #else
    void* res;
    // http://www.cplusplus.com/reference/cstdlib/realloc/
    if (pointer != 0)
    {
        size_t prevsz ;
        // has it been previously freed ?
        if (sg_malloc_has(pointer) == 0)
        {
            // pointer has been freed, yet used again...
            #if LOGGING
            if (sg_log == 0) sg_log = fopen("nvidia-malloc.log", "w");
            fprintf(sg_log, "WARNING !! realloc with an unknown (non null) pointer = %p - no memcpy will be performed !\n", pointer);
            fclose(sg_log);
            #endif
            prevsz = 0;
        } else 
            prevsz = sg_malloc_get(pointer);
        if (prevsz != 0)
        {
            CUDA_CHECK(cudaMemPrefetchAsync (pointer, prevsz, cudaCpuDeviceId, 0));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        cs_cuda_mem_malloc(&res, size);
        size_t minsz = prevsz < size ? prevsz : size;
        memcpy(res, pointer, minsz);
        // CUDA_CHECK(cudaMemcpy(res, pointer, minsz, cudaMemcpyDefault));
        CUDA_CHECK(cudaDeviceSynchronize());
        cs_cuda_mem_free(pointer);
    } else 
        cs_cuda_mem_malloc(&res, size);
    return res;
    #endif
}

}
