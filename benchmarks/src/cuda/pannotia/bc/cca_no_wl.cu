/*
ECL-CC code: ECL-CC is a connected components algorithm. The CUDA
implementation thereof is very fast. It operates on graphs stored in
binary CSR format.

Copyright (c) 2017, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted for academic, research, experimental, or personal use provided
that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice,
     this list of conditions, and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions, and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry
Relations at Texas State University <http://www.txstate.edu/ocir/>.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Jayadharini Jaiganesh and Martin Burtscher
*/

#include <cuda.h>
#include <graph_parser/libinputs_externs.h>
#include <graph_parser/libinputs_mappings.h>
#include <stdlib.h>
#include <stdio.h>
#include <set>
#include <deque>
#include <assert.h>
//#include "../graph_parser/parse.h"
//#include "../graph_parser/util.h"

int BLOCK_SIZE = 64;
int g_num_nodes;              // total number of nodes in input graph
int g_num_edges;              // total number of edges in input graph
int g_num_cpu_threads;        // number of CPU threads that do work
//int g_num_running_pthreads;   // number of pthreads launched (needs to be power of 2 for treebar
int g_num_data_blocks;        // total amount of work in graph
int g_num_cpu_blocks;         // total amount of work allocated to cpu threads
int g_num_gpu_threadblocks;   // number of GPU threadblocks launched
int g_num_gpu_blocks;         // total amount of work allocated to GPU
int max_iters = 0;
int num_warmup_iters = 0;
int verts_per_thread = 0;
int gpu_percentage = 100;
bool DEBUG_LOCAL = false;

//static const int ThreadsPerBlock = 256;
//static const int warpsize = 32;

/*
 Repurposed atomics
 Atomic Read - atomicAdd(&var,0)
 Atomic Store - atomicOr(&var, val)
 */

/** Initialize with first smaller neighbor ID
 * @param   nodes   number of vertices
 * @param   nidx    neighbor index list
 * @param   nlist   neighbor (edge) list
 * @param   nstat   vertex label list
 */
static __global__ 
void init(int nodes,
          const int* const __restrict__ nidx,
          const int* const __restrict__ nlist,
          int* const __restrict__ nstat)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) {
        const int beg = nidx[v];
        const int end = nidx[v + 1];
        int m = v;
        int i = beg;
        while ((m == v) && (i < end)) {
            m = min(m, nlist[i]);
            i++;
        }
        #ifdef SYNC
        atomicExch(&nstat[v], m);
        #else
        nstat[v] = m;
        #endif
    }
}

/** Intermediate pointer jumping 
 * @param   idx     Index of vertex to IPJ
 * @param   nstat   vertex label list
 */
static inline __device__ int representative(const int idx, int* const __restrict__ nstat)
{
    #ifdef SYNC
    int curr = atomicAdd(&nstat[idx], 0);
    #else
    int curr = nstat[idx];
    #endif
    if (curr != idx) {
        int next, prev = idx;
        #ifdef SYNC
        while (curr > (next = atomicAdd(&nstat[curr], 0))) {
        #else
        while (curr > (next = nstat[curr])) {
        #endif    
            #ifdef SYNC
            atomicExch(&nstat[prev], next); //atomicExch is an atomicStore
            #else
            nstat[prev] =  next;
            #endif
            prev = curr;
            curr = next;
        }
    }
    return curr;
}

/** Process low-degree vertices at thread granularity and fill workist
 * @param   nodes   number of vertices
 * @param   nidx    neighbor index list
 * @param   nlist   neighbor (edge) list
 * @param   nstat   vertex label list
 * @param   wl      worklist
 */
static __global__
void compute1(int nodes,
              const int* const __restrict__ nidx,
              const int* const __restrict__ nlist,
              int* const __restrict__ nstat,
              int* const __restrict__ wl)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) {
        #ifdef SYNC
        const int vstat = atomicAdd(&nstat[v], 0);
        #else
        const int vstat = nstat[v];
        #endif
        if (v != vstat) {
            const int beg = nidx[v];
            const int end = nidx[v + 1];
            int vstat = representative(v, nstat);
            for (int i = beg; i < end; i++) {
                const int nli = nlist[i];
                if (v > nli) {
                    int ostat = representative(nli, nstat);
                    bool repeat;
                    do {
                        repeat = false;
                        if (vstat != ostat) {
                            int ret;
                            if (vstat < ostat) {
                                #ifdef SYNC
                                if ((ret = atomicMin(&nstat[ostat], vstat)) != ostat) {
                                #else
                                if ((ret = ((nstat[ostat] < vstat) ? nstat[ostat] : vstat)) != ostat)
                                #endif    
                                    ostat = ret;
                                    repeat = true;
                                }
                            } else {
                                #ifdef SYNC
                                if ((ret = atomicMin(&nstat[vstat], ostat)) != vstat) {
                                #else
                                if ((ret = ((nstat[vstat] < ostat) ? nstat[vstat] : ostat)) != vstat)
                                #endif    
                                    vstat = ret;
                                    repeat = true;
                                }
                            }
                        }
                    } while (repeat);
                }
            }
        }
    }
}
 
/** Link all vertices to sink
 * @param   nodes   number of vertices
 * @param   nidx    neighbor index list
 * @param   nlist   neighbor (edge) list
 * @param   nstat   vertex label list
 */
static __global__
void flatten(int nodes,
             const int* const __restrict__ nidx,
             const int* const __restrict__ nlist,
             int* const __restrict__ nstat)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) {
        int next, vstat;
        #ifdef SYNC
        vstat = atomicAdd(&nstat[v], 0);
        #else
        vstat = nstat[v];
        #endif
        const int old = vstat;
        #ifdef SYNC
        while (vstat > (next = atomicAdd(&nstat[vstat], 0))) {
        #else
        while (vstat > (next = nstat[vstat])) {
        #endif    
            vstat = next;
        }
        #ifdef SYNC
        if (old != vstat) { atomicExch(&nstat[v], vstat); }
        #else
        if (old != vstat) { nstat[v] = vstat; }
        #endif
    }
}


static void computeCC(int nodes,
                      int edges,
                      int* nidx,
                      int* nlist,
                      int* nstat,
                      int* wl,
                      int blocks,
                      int ThreadsPerBlock)
{
    if (DEBUG_LOCAL) {
        fprintf(stdout, "# TB: %d, # threads/TB: %d, total threads: %d\n",
                blocks, ThreadsPerBlock, blocks * ThreadsPerBlock);
    }
    init<<<blocks, ThreadsPerBlock>>>(nodes, nidx, nlist, nstat);
    cudaDeviceSynchronize();
    if (DEBUG_LOCAL) {
      fprintf(stdout, "init kernel completed.\n");
    }
    
    compute1<<<blocks, ThreadsPerBlock>>>(nodes, nidx, nlist, nstat, wl);
    cudaDeviceSynchronize();
    if (DEBUG_LOCAL) {
      fprintf(stdout, "compute1 kernel completed.\n");
    }

    flatten<<<blocks, ThreadsPerBlock>>>(nodes, nidx, nlist, nstat);
    cudaDeviceSynchronize();
    if (DEBUG_LOCAL) {
      fprintf(stdout, "flatten kernel completed.\n");
    }
}

static void verify(const int v, const int id, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat)
{
    std::deque<int> stack;
    stack.push_back(v);
    while(!stack.empty()) {
        const int v_new = stack.front();
        stack.pop_front();
        if (nstat[v_new] >= 0) {
            if (nstat[v_new] != id) {fprintf(stderr, "ERROR: found incorrect ID value\n\n");  exit(-1);}
            nstat[v_new] = -1;
            for (int i = nidx[v_new]; i < nidx[v_new + 1]; i++) {
                stack.push_back(nlist[i]);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    printf("Main started\n");
    fflush(stdout);

    int graph_file_id = -1;
    int graph_file_variant_int = (int)GRAPH_INPUTS::NORMAL;

    /******************************PARSE INPUT***********************************/
    if (argc >= 2) {
        if (argc >= 3) {
            DEBUG_LOCAL = (bool)atoi(argv[2]);
        } else {
            DEBUG_LOCAL = false;
        }

        char * const input = argv[1];

        /*
          Overall format is:
          
          - graph_file_id comes from libinput_mappings.h in graph_parser.  For example, f11 is rajat
          - v0 is for "use graph as-is, don't sort, etc.".
          - i15 is 15 iterations (ignored in this app)
          - w0 is 0 warmup iterations
          - c1 is 1 CPU (thread, ignored in this app)
          - g15 is 15 SMs (seems to be ignored in this app)
          - b256 is TB size
          - n1 is 1 vertex per thread (verts_per_thread -- we want it to be 1)
          - p100 is how much of the program should be run on the GPU (we want this to be 100%).

          More details:
          - if we use "n0" instead, then the 'g' parameter will be used directly and a persistent thread model will be used (n1 --> standard 1V1T with as many thread blocks as required by the input). 
         */
        sscanf(input, "f%dv%di%dw%dc%dg%db%dn%dp%d",
               &graph_file_id, &graph_file_variant_int, &max_iters,
               &num_warmup_iters,
               &g_num_cpu_threads, &g_num_gpu_threadblocks, &BLOCK_SIZE,
               &verts_per_thread,
               &gpu_percentage);
        //fprintf(stdout, "# warmup iters: %d, max iters: %d, vertices per thread: %d, GPU percentage: %d\n", num_warmup_iters, max_iters, verts_per_thread, gpu_percentage);

        // Not applicable here because we don't use warmup iterations but kept
        assert(num_warmup_iters < max_iters &&
               "Must have at least one simulated iteration");
    } else {
        fprintf(stderr, "You did something wrong! format is:\n \
                         cca_no_wl f<graph_file_index>c<num_cpu_threads> \
                         g<num_gpu_blocks>p<gpu_percentage>\n");
        exit(1);
    }

    if (DEBUG_LOCAL) {
        fprintf(stdout, "Successfully parsed input\n");
    }
    /******************************READ IN GRAPH*********************************/
    int *nidx;
    int *nlist;
    int* wl;
    int* nstat;
    int *col_d_in, *row_d_in;
    int *data_d_in, *col_cnt_d_in;
    int num_random_verts;
    int *random_verts;

    const std::string graph_file_str = GRAPH_INPUTS::get_graph_data(graph_file_id,
                                                                    graph_file_variant_int,
                                                                    g_num_nodes,
                                                                    g_num_edges,
                                                                    num_random_verts,
                                                                    row_d_in,
                                                                    col_d_in,
                                                                    data_d_in,
                                                                    col_cnt_d_in,
                                                                    random_verts);

    if (DEBUG_LOCAL) {
        fprintf(stdout, "Successfully called get_graph_data\n");
    }

	nidx  = (int *)malloc((g_num_nodes + 1) * sizeof(int));
    nlist = (int *)malloc(g_num_edges * sizeof(int));
    nstat = (int *)malloc(g_num_nodes * sizeof(int));
    wl    = (int *)malloc(g_num_nodes * sizeof(int));

    int * nidx_d = NULL;
    int * nlist_d = NULL;
    int * wl_d = NULL;
    int * nstat_d = NULL;

    // since GPGPU-Sim doesn't seem to support CUDA UVM, use discrete copies
    cudaError_t err = cudaMalloc(&nidx_d, (g_num_nodes + 1) * sizeof(int));
    if (err != cudaSuccess) {
      fprintf(stderr, "ERROR: cudaMalloc nidx_d (size: %d) => %s\n",  g_num_nodes + 1, cudaGetErrorString(err));
      return -1;
    }

    err = cudaMalloc(&nlist_d, g_num_edges * sizeof(int));
    if (err != cudaSuccess) {
      fprintf(stderr, "ERROR: cudaMalloc nlist_d (size: %d) => %s\n",  g_num_edges, cudaGetErrorString(err));
      return -1;
    }

    err = cudaMalloc(&nstat_d, g_num_nodes * sizeof(int));
    if (err != cudaSuccess) {
      fprintf(stderr, "ERROR: cudaMalloc nstat_d (size: %d) => %s\n",  g_num_nodes, cudaGetErrorString(err));
      return -1;
    }

    err = cudaMalloc(&wl_d, g_num_nodes * sizeof(int));
    if (err != cudaSuccess) {
      fprintf(stderr, "ERROR: cudaMalloc wl_d (size: %d) => %s\n",  g_num_nodes, cudaGetErrorString(err));
      return -1;
    }

    if (DEBUG_LOCAL) {
        fprintf(stdout, "Successfully malloced\n");
    }

    //Copy local data to unified mem
    for(int i=0; i < g_num_nodes + 1; i++) {
        //nidx[i] = csr->row_array[i];
        nidx[i] = row_d_in[i];
    } 
    for(int i=0; i < g_num_edges; i++) {
        //nlist[i] = csr->col_array[i];
        nlist[i] = col_d_in[i];
    }
    for(int i=0; i < g_num_nodes; i++) {
        nstat[i] = 0;
        wl[i] = 0;
    }

	// copy host arrays to device arrays
	err = cudaMemcpy(nidx_d, nidx, (g_num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	  fprintf(stderr, "ERROR: cudaMemcpy nidx_d (size:%d) => %s\n", g_num_nodes + 1, cudaGetErrorString(err));
	  return -1;
	}

	err = cudaMemcpy(nlist_d, nlist, g_num_edges * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	  fprintf(stderr, "ERROR: cudaMemcpy nlist_d (size:%d) => %s\n", g_num_edges, cudaGetErrorString(err));
	  return -1;
	}

	err = cudaMemcpy(nstat_d, nstat, g_num_nodes * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	  fprintf(stderr, "ERROR: cudaMemcpy nstat_d (size:%d) => %s\n", g_num_nodes, cudaGetErrorString(err));
	  return -1;
	}

	err = cudaMemcpy(wl_d, wl, g_num_nodes * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	  fprintf(stderr, "ERROR: cudaMemcpy wl_d (size:%d) => %s\n", g_num_nodes, cudaGetErrorString(err));
	  return -1;
	}

    if (DEBUG_LOCAL) {
      fprintf(stdout, "Successfully copied/initialized data\n");
    }

    if (DEBUG_LOCAL) {
      fprintf(stdout, "input graph: %d nodes and %d edges\n", g_num_nodes, g_num_edges);
      fprintf(stdout, "average degree: %.2f edges per node\n", 1.0 * g_num_edges / g_num_nodes);
    }

    /******************************COMPUTATION***********************************/
    g_num_data_blocks = (g_num_nodes % BLOCK_SIZE == 0) ?
                        (g_num_nodes / BLOCK_SIZE) :
                        (g_num_nodes / BLOCK_SIZE + 1);
    g_num_gpu_blocks = (gpu_percentage * g_num_data_blocks) / 100;
    g_num_cpu_blocks = g_num_data_blocks - g_num_gpu_blocks;

    if (verts_per_thread > 0) {
        g_num_gpu_threadblocks = (int)ceil(g_num_gpu_blocks / (double) verts_per_thread);
    } else if (g_num_gpu_threadblocks > g_num_gpu_blocks) {
        g_num_gpu_threadblocks = g_num_gpu_blocks;
    }

    computeCC(g_num_nodes, g_num_edges, nidx_d, nlist_d, nstat_d, wl_d, g_num_gpu_threadblocks, BLOCK_SIZE);

	// copy back arrays we need to access on host now
	err = cudaMemcpy(nidx, nidx_d, (g_num_nodes + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
	  fprintf(stderr, "ERROR: read nidx_d variable (%s)\n", cudaGetErrorString(err));
	  return -1;
	}

	err = cudaMemcpy(nlist, nlist_d, g_num_edges * sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
	  fprintf(stderr, "ERROR: read nlist_d variable (%s)\n", cudaGetErrorString(err));
	  return -1;
	}

	err = cudaMemcpy(nstat, nstat_d, g_num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
	  fprintf(stderr, "ERROR: read nstat_d variable (%s)\n", cudaGetErrorString(err));
	  return -1;
	}

    /******************************VERIFICATION**********************************/
    /* verification code (may need extra runtime stack space due to deep recursion) */
    if (DEBUG_LOCAL) {
        std::set<int> s1;
        for (int v = 0; v < g_num_nodes; v++) {
            s1.insert(nstat[v]);
        }
        if (DEBUG_LOCAL) {
            fprintf(stdout, "number of connected components: %ld\n", s1.size());
        }

        for (int v = 0; v < g_num_nodes; v++) {
            for (int i = nidx[v]; i < nidx[v + 1]; i++) {
                if (nstat[nlist[i]] != nstat[v]) {
                    fprintf(stderr, "ERROR: found adjacent nodes in different components\n\n");
                    exit(-1);
                }
            }
        }

        for (int v = 0; v < g_num_nodes; v++) {
            if (nstat[v] < 0) {
                fprintf(stderr, "ERROR: found negative component number\n\n");
                exit(-1);
            }
        }

        std::set<int> s2;
        int count = 0;
        for (int v = 0; v < g_num_nodes; v++) {
            if (nstat[v] >= 0) {
                count++;
                s2.insert(nstat[v]);
                verify(v, nstat[v], nidx, nlist, nstat);
            }
        }
        if (s1.size() != s2.size()) {
            fprintf(stderr, "ERROR: number of components do not match\n\n");
            exit(-1);
        }
        if (s1.size() != count) {
            fprintf(stderr, "ERROR: component IDs are not unique\n\n");
            exit(-1);
        }
    }

    cudaFree(nidx);
    cudaFree(nlist);
    cudaFree(nstat);
    cudaFree(wl);

    return 0;
}
