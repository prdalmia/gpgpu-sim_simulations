/************************************************************************************\
 *                                                                                  *
 * Copyright � 2014 Advanced Micro Devices, Inc.                                    *
 * All rights reserved.                                                             *
 *                                                                                  *
 * Redistribution and use in source and binary forms, with or without               *
 * modification, are permitted provided that the following are met:                 *
 *                                                                                  *
 * You must reproduce the above copyright notice.                                   *
 *                                                                                  *
 * Neither the name of the copyright holder nor the names of its contributors       *
 * may be used to endorse or promote products derived from this software            *
 * without specific, prior, written permission from at least the copyright holder.  *
 *                                                                                  *
 * You must include the following terms in your license and/or other materials      *
 * provided with the software.                                                      *
 *                                                                                  *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"      *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE        *
 * IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, AND FITNESS FOR A       *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER        *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,         *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT  *
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN          *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING  *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY   *
 * OF SUCH DAMAGE.                                                                  *
 *                                                                                  *
 * Without limiting the foregoing, the software may implement third party           *
 * technologies for which you must obtain licenses from parties other than AMD.     *
 * You agree that AMD has not obtained or conveyed to you, and that you shall       *
 * be responsible for obtaining the rights to use and/or distribute the applicable  *
 * underlying intellectual property rights related to the third party technologies. *
 * These third party technologies are not licensed hereunder.                       *
 *                                                                                  *
 * If you use the software (in whole or in part), you shall adhere to all           *
 * applicable U.S., European, and other export laws, including but not limited to   *
 * the U.S. Export Administration Regulations ("EAR") (15 C.F.R Sections 730-774),  *
 * and E.U. Council Regulation (EC) No 428/2009 of 5 May 2009.  Further, pursuant   *
 * to Section 740.6 of the EAR, you hereby certify that, except pursuant to a       *
 * license granted by the United States Department of Commerce Bureau of Industry   *
 * and Security or as otherwise permitted pursuant to a License Exception under     *
 * the U.S. Export Administration Regulations ("EAR"), you will not (1) export,     *
 * re-export or release to a national of a country in Country Groups D:1, E:1 or    *
 * E:2 any restricted technology, software, or source code you receive hereunder,   *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such       *
 * technology or software, if such foreign produced direct product is subject to    *
 * national security controls as identified on the Commerce Control List (currently *
 * found in Supplement 1 to Part 774 of EAR).  For the most current Country Group   *
 * listings, or for additional information about the EAR or your obligations under  *
 * those regulations, please refer to the U.S. Bureau of Industry and Security's    *
 * website at http://www.bis.doc.gov/.                                              *
 *                                                                                  *
\************************************************************************************/

#define BIG_NUM 99999999
//#include "denovo_util.h"
//#include "gpuKernels_util.cu"
//#include "util.h"
/**
 * @brief   min.+
 * @param   num_nodes  Number of vertices
 * @param   row        CSR pointer array
 * @param   col        CSR column array
 * @param   data       Weight array
 * @param   x          Input vector
 * @param   y          Output vector
 */
__global__ void
spmv_min_dot_plus_kernel(int num_nodes,
                         int *row,
                         int *col,
                         int *data,
                         int *x,
                         int *y,
                         int *stop)
                         {
                            // Get my workitem id
                            int tid = blockDim.x * blockIdx.x + threadIdx.x;
                            int edge = 0;
                        /*
                            if (threadIdx.x == 0) {
                              __denovo_setAcquireRegion(SPECIAL_REGION);
                              __denovo_addAcquireRegion(READ_ONLY_REGION);
                              __denovo_addAcquireRegion(default_reg);
                              __denovo_addAcquireRegion(rel_reg);
                            }
                            */
                            __syncthreads();
                        
                            for (; tid < num_nodes; tid += blockDim.x * gridDim.x) {
                              //const int should_stop = (int)atomicXor(&stop[tid], 0);
                              const int should_stop = stop[tid];
                        
                              if (!should_stop) {
                                // Get the start and end pointers
                                const int row_start = row[tid];
                                const int row_end = row[tid + 1];
                        
                                const int min = x[tid];
                        
                                for (int i = row_start; i < row_end; i++) {
                                  const int col_i = col[i];
                                 const int data_i = data[i];
                        
                                  int * const y_col_addr = &y[col_i];
                                  const int new_val = data_i + min;
                        
                                  atomicMin(y_col_addr, new_val);
                                }
                        /*
                                asm volatile
                                (
                                  // Temp Register
                                  ".reg .u64 m99;\n\t"            // Temp reg
                                  ".reg .s32 m100;\n\t"           // Temp reg
                        
                                  "mov.u64 m99, %0;\n\t"          // m99 = y
                                  "mov.s32 m100, %1;\n\t"         // m100 = min
                        
                                  :                               // No outputs
                                  : "l"(y), "r"(min)              // Inputs
                                );
                        
                                for (edge = row_start; edge <= (row_end - 8); edge += 8) {
                                  int * const col_base_addr = &col[edge];
                                  int * const data_base_addr = &data[edge];
                        
                                  asm volatile
                                  (
                                    ".reg .s32 m1;\n\t"                                  // Register for nid loaded from col
                                    ".reg .s32 m2;\n\t"                                  // Register for nid loaded from col
                                    ".reg .s32 m3;\n\t"                                  // Register for nid loaded from col
                                    ".reg .s32 m4;\n\t"                                  // Register for nid loaded from col
                                    ".reg .s32 m5;\n\t"                                  // Register for nid loaded from col
                                    ".reg .s32 m6;\n\t"                                  // Register for nid loaded from col
                                    ".reg .s32 m7;\n\t"                                  // Register for nid loaded from col
                                    ".reg .s32 m8;\n\t"                                  // Register for nid loaded from col
                        
                                    ".reg .s32 m9;\n\t"                                  // Register for data
                                    ".reg .s32 m10;\n\t"                                 // Register for data
                                    ".reg .s32 m11;\n\t"                                 // Register for data
                                    ".reg .s32 m12;\n\t"                                 // Register for data
                                    ".reg .s32 m13;\n\t"                                 // Register for data
                                    ".reg .s32 m14;\n\t"                                 // Register for data
                                    ".reg .s32 m15;\n\t"                                 // Register for data
                                    ".reg .s32 m16;\n\t"                                 // Register for data
                        
                                    ".reg .u64 m17;\n\t"                                 // Register for multiplied nid value as address
                                    ".reg .u64 m18;\n\t"                                 // Register for multiplied nid value as address
                                    ".reg .u64 m19;\n\t"                                 // Register for multiplied nid value as address
                                    ".reg .u64 m20;\n\t"                                 // Register for multiplied nid value as address
                                    ".reg .u64 m21;\n\t"                                 // Register for multiplied nid value as address
                                    ".reg .u64 m22;\n\t"                                 // Register for multiplied nid value as address
                                    ".reg .u64 m23;\n\t"                                 // Register for multiplied nid value as address
                                    ".reg .u64 m24;\n\t"                                 // Register for multiplied nid value as address
                        
                                    ".reg .u64 m25;\n\t"                                 // Register for final address to load from x
                                    ".reg .u64 m26;\n\t"                                 // Register for final address to load from x
                                    ".reg .u64 m27;\n\t"                                 // Register for final address to load from x
                                    ".reg .u64 m28;\n\t"                                 // Register for final address to load from x
                                    ".reg .u64 m29;\n\t"                                 // Register for final address to load from x
                                    ".reg .u64 m30;\n\t"                                 // Register for final address to load from x
                                    ".reg .u64 m31;\n\t"                                 // Register for final address to load from x
                                    ".reg .u64 m32;\n\t"                                 // Register for final address to load from x
                        
                                    ".reg .s32 m33;\n\t"                                 // Register for x
                                    ".reg .s32 m34;\n\t"                                 // Register for x
                                    ".reg .s32 m35;\n\t"                                 // Register for x
                                    ".reg .s32 m36;\n\t"                                 // Register for x
                                    ".reg .s32 m37;\n\t"                                 // Register for x
                                    ".reg .s32 m38;\n\t"                                 // Register for x
                                    ".reg .s32 m39;\n\t"                                 // Register for x
                                    ".reg .s32 m40;\n\t"                                 // Register for x
                        
                                    ".reg .s32 m41;\n\t"                                 // (Unused) Register
                                    ".reg .s32 m42;\n\t"                                 // (Unused) Register
                                    ".reg .s32 m43;\n\t"                                 // (Unused) Register
                                    ".reg .s32 m44;\n\t"                                 // (Unused) Register
                                    ".reg .s32 m45;\n\t"                                 // (Unused) Register
                                    ".reg .s32 m46;\n\t"                                 // (Unused) Register
                                    ".reg .s32 m47;\n\t"                                 // (Unused) Register
                                    ".reg .s32 m48;\n\t"                                 // (Unused) Register
                        
                                    "ld.s32 m1, [%0+0];\n\t"                             // Load nid
                                    "ld.s32 m2, [%0+4];\n\t"                             // Load nid
                                    "ld.s32 m3, [%0+8];\n\t"                             // Load nid
                                    "ld.s32 m4, [%0+12];\n\t"                            // Load nid
                                    "ld.s32 m5, [%0+16];\n\t"                            // Load nid
                                    "ld.s32 m6, [%0+20];\n\t"                            // Load nid
                                    "ld.s32 m7, [%0+24];\n\t"                            // Load nid
                                    "ld.s32 m8, [%0+28];\n\t"                            // Load nid
                        
                                    "ld.s32 m9, [%1+0];\n\t"                             // Load data
                                    "ld.s32 m10, [%1+4];\n\t"                            // Load data
                                    "ld.s32 m11, [%1+8];\n\t"                            // Load data
                                    "ld.s32 m12, [%1+12];\n\t"                           // Load data
                                    "ld.s32 m13, [%1+16];\n\t"                           // Load data
                                    "ld.s32 m14, [%1+20];\n\t"                           // Load data
                                    "ld.s32 m15, [%1+24];\n\t"                           // Load data
                                    "ld.s32 m16, [%1+28];\n\t"                           // Load data
                        
                                    "mul.wide.s32 m17, m1, 4;\n\t"                       // Multiply nid for y address calculation
                                    "mul.wide.s32 m18, m2, 4;\n\t"                       // Multiply nid for y address calculation
                                    "mul.wide.s32 m19, m3, 4;\n\t"                       // Multiply nid for y address calculation
                                    "mul.wide.s32 m20, m4, 4;\n\t"                       // Multiply nid for y address calculation
                                    "mul.wide.s32 m21, m5, 4;\n\t"                       // Multiply nid for y address calculation
                                    "mul.wide.s32 m22, m6, 4;\n\t"                       // Multiply nid for y address calculation
                                    "mul.wide.s32 m23, m7, 4;\n\t"                       // Multiply nid for y address calculation
                                    "mul.wide.s32 m24, m8, 4;\n\t"                       // Multiply nid for y address calculation
                        
                                    "add.u64 m25, m99, m17;\n\t"                         // Final address calculation for y
                                    "add.u64 m26, m99, m18;\n\t"                         // Final address calculation for y
                                    "add.u64 m27, m99, m19;\n\t"                         // Final address calculation for y
                                    "add.u64 m28, m99, m20;\n\t"                         // Final address calculation for y
                                    "add.u64 m29, m99, m21;\n\t"                         // Final address calculation for y
                                    "add.u64 m30, m99, m22;\n\t"                         // Final address calculation for y
                                    "add.u64 m31, m99, m23;\n\t"                         // Final address calculation for y
                                    "add.u64 m32, m99, m24;\n\t"                         // Final address calculation for y
                        
                                    "add.s32 m33, m9, m100;\n\t"                         // Add data + min
                                    "add.s32 m34, m10, m100;\n\t"                        // Add data + min
                                    "add.s32 m35, m11, m100;\n\t"                        // Add data + min
                                    "add.s32 m36, m12, m100;\n\t"                        // Add data + min
                                    "add.s32 m37, m13, m100;\n\t"                        // Add data + min
                                    "add.s32 m38, m14, m100;\n\t"                        // Add data + min
                                    "add.s32 m39, m15, m100;\n\t"                        // Add data + min
                                    "add.s32 m40, m16, m100;\n\t"                        // Add data + min
                        
                                    "atom.min.s32 m41, [m25], m33;\n\t"                  // Do min for y and data + min
                                    "atom.min.s32 m42, [m26], m34;\n\t"                  // Do min for y and data + min
                                    "atom.min.s32 m43, [m27], m35;\n\t"                  // Do min for y and data + min
                                    "atom.min.s32 m44, [m28], m36;\n\t"                  // Do min for y and data + min
                                    "atom.min.s32 m45, [m29], m37;\n\t"                  // Do min for y and data + min
                                    "atom.min.s32 m46, [m30], m38;\n\t"                  // Do min for y and data + min
                                    "atom.min.s32 m47, [m31], m39;\n\t"                  // Do min for y and data + min
                                    "atom.min.s32 m48, [m32], m40;\n\t"                  // Do min for y and data + min
                        
                                    :                                                    // Outputs
                                    : "l"(col_base_addr), "l"(data_base_addr)            // Inputs
                                  );
                                }
                        
                                for (; edge <= (row_end - 4); edge += 4) {
                                  int * const col_base_addr = &col[edge];
                                  int * const data_base_addr = &data[edge];
                        
                                  asm volatile
                                  (
                                    ".reg .s32 q1;\n\t"                                  // Register for nid loaded from col
                                    ".reg .s32 q2;\n\t"                                  // Register for nid loaded from col
                                    ".reg .s32 q3;\n\t"                                  // Register for nid loaded from col
                                    ".reg .s32 q4;\n\t"                                  // Register for nid loaded from col
                        
                                    ".reg .s32 q9;\n\t"                                  // Register for data
                                    ".reg .s32 q10;\n\t"                                 // Register for data
                                    ".reg .s32 q11;\n\t"                                 // Register for data
                                    ".reg .s32 q12;\n\t"                                 // Register for data
                        
                                    ".reg .u64 q17;\n\t"                                 // Register for multiplied nid value as address
                                    ".reg .u64 q18;\n\t"                                 // Register for multiplied nid value as address
                                    ".reg .u64 q19;\n\t"                                 // Register for multiplied nid value as address
                                    ".reg .u64 q20;\n\t"                                 // Register for multiplied nid value as address
                        
                                    ".reg .u64 q25;\n\t"                                 // Register for final address to load from x
                                    ".reg .u64 q26;\n\t"                                 // Register for final address to load from x
                                    ".reg .u64 q27;\n\t"                                 // Register for final address to load from x
                                    ".reg .u64 q28;\n\t"                                 // Register for final address to load from x
                        
                                    ".reg .s32 q33;\n\t"                                 // Register for x
                                    ".reg .s32 q34;\n\t"                                 // Register for x
                                    ".reg .s32 q35;\n\t"                                 // Register for x
                                    ".reg .s32 q36;\n\t"                                 // Register for x
                        
                                    ".reg .s32 q41;\n\t"                                 // (Unused) Register
                                    ".reg .s32 q42;\n\t"                                 // (Unused) Register
                                    ".reg .s32 q43;\n\t"                                 // (Unused) Register
                                    ".reg .s32 q44;\n\t"                                 // (Unused) Register
                        
                                    "ld.s32 q1, [%0+0];\n\t"                             // Load nid
                                    "ld.s32 q2, [%0+4];\n\t"                             // Load nid
                                    "ld.s32 q3, [%0+8];\n\t"                             // Load nid
                                    "ld.s32 q4, [%0+12];\n\t"                            // Load nid
                        
                                    "ld.s32 q9, [%1+0];\n\t"                             // Load data
                                    "ld.s32 q10, [%1+4];\n\t"                            // Load data
                                    "ld.s32 q11, [%1+8];\n\t"                            // Load data
                                    "ld.s32 q12, [%1+12];\n\t"                           // Load data

                                    "mul.wide.s32 q17, q1, 4;\n\t"                       // Multiply nid for y address calculation
                                    "mul.wide.s32 q18, q2, 4;\n\t"                       // Multiply nid for y address calculation
                                    "mul.wide.s32 q19, q3, 4;\n\t"                       // Multiply nid for y address calculation
                                    "mul.wide.s32 q20, q4, 4;\n\t"                       // Multiply nid for y address calculation
                        
                                    "add.u64 q25, m99, q17;\n\t"                         // Final address calculation for y
                                    "add.u64 q26, m99, q18;\n\t"                         // Final address calculation for y
                                    "add.u64 q27, m99, q19;\n\t"                         // Final address calculation for y
                                    "add.u64 q28, m99, q20;\n\t"                         // Final address calculation for y
                        
                                    "add.s32 q33, q9, m100;\n\t"                         // Add data + min
                                    "add.s32 q34, q10, m100;\n\t"                        // Add data + min
                                    "add.s32 q35, q11, m100;\n\t"                        // Add data + min
                                    "add.s32 q36, q12, m100;\n\t"                        // Add data + min
                        
                                    "atom.min.s32 q41, [q25], q33;\n\t"                  // Do min for y and data + min
                                    "atom.min.s32 q42, [q26], q34;\n\t"                  // Do min for y and data + min
                                    "atom.min.s32 q43, [q27], q35;\n\t"                  // Do min for y and data + min
                                    "atom.min.s32 q44, [q28], q36;\n\t"                  // Do min for y and data + min
                        
                                    :                                                    // Outputs
                                    : "l"(col_base_addr), "l"(data_base_addr)            // Inputs
                                  );
                                }
                        
                                for (; edge <= (row_end - 2); edge += 2) {
                                  int * const col_base_addr = &col[edge];
                                  int * const data_base_addr = &data[edge];
                        
                                  asm volatile
                                  (
                                    ".reg .s32 t1;\n\t"                                  // Register for nid loaded from col
                                    ".reg .s32 t2;\n\t"                                  // Register for nid loaded from col
                        
                                    ".reg .s32 t9;\n\t"                                  // Register for data
                                    ".reg .s32 t10;\n\t"                                 // Register for data
                        
                                    ".reg .u64 t17;\n\t"                                 // Register for multiplied nid value as address
                                    ".reg .u64 t18;\n\t"                                 // Register for multiplied nid value as address
                        
                                    ".reg .u64 t25;\n\t"                                 // Register for final address to load from x
                                    ".reg .u64 t26;\n\t"                                 // Register for final address to load from x
                        
                                    ".reg .s32 t33;\n\t"                                 // Register for x
                                    ".reg .s32 t34;\n\t"                                 // Register for x
                        
                                    ".reg .s32 t41;\n\t"                                 // (Unused) Register
                                    ".reg .s32 t42;\n\t"                                 // (Unused) Register
                        
                                    "ld.s32 t1, [%0+0];\n\t"                             // Load nid
                                    "ld.s32 t2, [%0+4];\n\t"                             // Load nid
                        
                                    "ld.s32 t9, [%1+0];\n\t"                             // Load data
                                    "ld.s32 t10, [%1+4];\n\t"                            // Load data
                        
                                    "mul.wide.s32 t17, t1, 4;\n\t"                       // Multiply nid for y address calculation
                                    "mul.wide.s32 t18, t2, 4;\n\t"                       // Multiply nid for y address calculation
                        
                                    "add.u64 t25, m99, t17;\n\t"                         // Final address calculation for y
                                    "add.u64 t26, m99, t18;\n\t"                         // Final address calculation for y
                        
                                    "add.s32 t33, t9, m100;\n\t"                         // Add data + min
                                    "add.s32 t34, t10, m100;\n\t"                        // Add data + min
                        
                                    "atom.min.s32 t41, [t25], t33;\n\t"                  // Do min for y and data + min
                                    "atom.min.s32 t42, [t26], t34;\n\t"                  // Do min for y and data + min
                        
                                    :                                                    // Outputs
                                    : "l"(col_base_addr), "l"(data_base_addr)            // Inputs
                                  );
                                }

                                for (; edge < row_end; edge++) {
                                  int * const col_base_addr = &col[edge];
                                  int * const data_base_addr = &data[edge];
                        
                                  asm volatile
                                  (
                                    "{\n\t"
                                    ".reg .s32 a1;\n\t"                                  // Register for nid loaded from col
                                    ".reg .s32 a9;\n\t"                                  // Register for data
                                    ".reg .u64 a17;\n\t"                                 // Register for multiplied nid value as address
                                    ".reg .u64 a25;\n\t"                                 // Register for final address to load from x
                                    ".reg .s32 a33;\n\t"                                 // Register for x
                                    ".reg .s32 a41;\n\t"                                 // (Unused) Register
                        
                                    "ld.s32 a1, [%0+0];\n\t"                             // Load nid
                                    "ld.s32 a9, [%1+0];\n\t"                             // Load data
                                    "mul.wide.s32 a17, a1, 4;\n\t"                       // Multiply nid for y address calculation
                                    "add.u64 a25, m99, a17;\n\t"                         // Final address calculation for y
                                    "add.s32 a33, a9, m100;\n\t"                         // Add data + min
                                    "atom.min.s32 a41, [a25], a33;\n\t"                  // Do min for y and data + min
                                    "}"
                                    :                                                    // Outputs
                                    : "l"(col_base_addr), "l"(data_base_addr)            // Inputs
                                  );
                                }
                                */
                              }
                              
                              }
                        /*
                            if (threadIdx.x == 0) {
                              __denovo_gpuEpilogue(SPECIAL_REGION);
                              __denovo_gpuEpilogue(READ_ONLY_REGION);
                              __denovo_gpuEpilogue(default_reg);
                              __denovo_gpuEpilogue(rel_reg);
                            }
                            */
                        }
                        
                        /**
                         * @brief   vector_assign
                         * @param   vector1      vector1
                         * @param   vector2      vector2
                         * @param   num_nodes    number of vertices
                         */
                          __global__ void
                        vector_assign(int *x,
                                      int *y,
                                      int *stop,
                                      int num_nodes)
                        
                        {
                          int tid = blockDim.x * blockIdx.x + threadIdx.x;
                        /*
                          if (threadIdx.x == 0) {
                            __denovo_setAcquireRegion(SPECIAL_REGION);
                            __denovo_addAcquireRegion(READ_ONLY_REGION);
                            __denovo_addAcquireRegion(default_reg);
                            __denovo_addAcquireRegion(rel_reg);
                          }
                         */
                          __syncthreads();
                        //AskMatt
                          for (; tid < num_nodes; tid += blockDim.x * gridDim.x) {
                            const int x_val = x[tid];
                            const int y_val = (int)atomicAdd(&y[tid], 0);
                            //const int y_val = (int)atomicXor(&y[tid], 0);
                            const bool changed = x_val != y_val;
                        
                            if (changed) {
                              x[tid] = y_val;
                            }
                        
                            //x[tid] = y_val;
                            //atomicOr((int *)&(y[tid]), y_val);
                        
                            stop[tid] = !changed;
                            //atomicExch(&stop[tid], !changed);
                          }
                        /*
                           if (threadIdx.x == 0) {
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
    __denovo_gpuEpilogue(default_reg);
    __denovo_gpuEpilogue(rel_reg);
  }
*/
}

/**
 * @brief   min.+
 * @param   num_nodes  number of vertices
 * @param   height     the height of the adjacency matrix (col-major)
 * @param   col        the col array
 * @param   data       the data array
 * @param   x          the input vector
 * @param   y          the output vector
 */

 /**
 * @brief   vector_diff
 * @param   vector1      vector1
 * @param   vector2      vector2
 * @param   stop         termination variable
 * @param   num_nodes    number of vertices
 */
__global__ void
vector_diff(int *vector1, int *vector2, int *stop, const int num_nodes)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_nodes) {
        if (vector2[tid] != vector1[tid]) {
            *stop[tid] = 1;
        }
    }
}

__global__ void
vector_init(int *vector1, int *vector2, const int i, const int num_nodes)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_nodes) {
        if (tid == i) {
            // If it is the source vertex
            vector1[tid] = BIG_NUM;
            vector2[tid] = 0;
        } else {
            // If it a non-source vertex
            vector1[tid] = BIG_NUM;
            vector2[tid] = BIG_NUM;
        }
    }
}

                        
                                               