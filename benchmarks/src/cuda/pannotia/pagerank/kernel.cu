/************************************************************************************\
 *                                                                                  *
 * Copyright 2014 Advanced Micro Devices, Inc.                                      *
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

/**
 * @brief   pagerank1
 * @param   row         csr pointer array
 * @param   col         csr column array
 * @param   data        weight array
 * @param   page_rank1  pagerank array 1
 * @param   page_rank2  pagerank array 2
 * @param   num_nodes   number of vertices
 * @param   num_edges   number of edges
 * @param   rowColReg   csr pointer array and column array region
 * @param   pgRk1Reg    pagerank array 1 region
 * @param   pgRk2Reg    pagerank array 2 region
 */
__global__ void
pagerank1(int * row, int * col, int * data, float * page_rank1,
          float * page_rank2, const int num_nodes, const int num_edges)
{
  // Get my workitem id
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  // for inlining
  float * addr0 = NULL, * addr1 = NULL, * addr2 = NULL,
        * addr3 = NULL, * addr4 = NULL, * addr5 = NULL,
        * addr6 = NULL, * addr7 = NULL;
  int nid0 = 0, nid1 = 0, nid2 = 0, nid3 = 0, nid4 = 0,
      nid5 = 0, nid6 = 0, nid7 = 0;
/*
  if (threadIdx.x == 0) {
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
    __denovo_addAcquireRegion(pgRk1Reg); // read-only in this kernel, could have its region switched between kernels ...
    __denovo_addAcquireRegion(pgRk2Reg);
    __denovo_addAcquireRegion(rowColReg); // read-only region
  }
  __syncthreads();
*/
  for (; tid < num_nodes; tid += blockDim.x * gridDim.x) {
    // Get the starting and ending pointers of the neighborlist
    int start = row[tid];
    int end = row[tid + 1];

    /*
      NOTE: In the original code, page_rank1 is also read in the critical
      section but this is completely unnecessary because the value is
      constant for all of the loop iterations.
    */
    int numEdges = (end - start);
    float addVal = page_rank1[tid] / (float)(numEdges);

    // Navigate the neighbor list
    // Only move addVal into a register once
    // Only move page_rank into a register once
    asm volatile
    (
      // Temp Register
      ".reg .f32 m41;\n\t"            // Temp reg
      ".reg .u64 m42;\n\t"            // Temp reg

      "mov.f32 m41, %0;\n\t"          // m41 = addVal
      "mov.u64 m42, %1;\n\t"          // m42 = page_rank2

      :                               // No outputs
      : "f"(addVal), "l"(page_rank2)  // Inputs
    );

    /*
      Use loop peeling to overlap as many of the edges as possible -- overlap
      8 atomics as many times as possible, then 4 atomics, then 2, finally 1
    */
    int edge = 0; // need to keep track of edge peeled loop ends on
    for (edge = start; edge <= (end - 8); edge += 8) {
      int * const col_base_addr = &col[edge];

      asm volatile
      (
        ".reg .s32 m1;\n\t"                    // Register for nid loaded from col
        ".reg .s32 m2;\n\t"                    // Register for nid loaded from col
        ".reg .s32 m3;\n\t"                    // Register for nid loaded from col
        ".reg .s32 m4;\n\t"                    // Register for nid loaded from col
        ".reg .s32 m5;\n\t"                    // Register for nid loaded from col
        ".reg .s32 m6;\n\t"                    // Register for nid loaded from col
        ".reg .s32 m7;\n\t"                    // Register for nid loaded from col
        ".reg .s32 m8;\n\t"                    // Register for nid loaded from col

        ".reg .s64 m9;\n\t"                    // Register for casted nid value
        ".reg .s64 m10;\n\t"                   // Register for casted nid value
        ".reg .s64 m11;\n\t"                   // Register for casted nid value
        ".reg .s64 m12;\n\t"                   // Register for casted nid value
        ".reg .s64 m13;\n\t"                   // Register for casted nid value
        ".reg .s64 m14;\n\t"                   // Register for casted nid value
        ".reg .s64 m15;\n\t"                   // Register for casted nid value
        ".reg .s64 m16;\n\t"                   // Register for casted nid value

        ".reg .u64 m17;\n\t"                   // Register for multiplied nid value as address
        ".reg .u64 m18;\n\t"                   // Register for multiplied nid value as address
        ".reg .u64 m19;\n\t"                   // Register for multiplied nid value as address
        ".reg .u64 m20;\n\t"                   // Register for multiplied nid value as address
        ".reg .u64 m21;\n\t"                   // Register for multiplied nid value as address
        ".reg .u64 m22;\n\t"                   // Register for multiplied nid value as address
        ".reg .u64 m23;\n\t"                   // Register for multiplied nid value as address
        ".reg .u64 m24;\n\t"                   // Register for multiplied nid value as address

        ".reg .u64 m25;\n\t"                   // Register for final address to load from page_rank2
        ".reg .u64 m26;\n\t"                   // Register for final address to load from page_rank2
        ".reg .u64 m27;\n\t"                   // Register for final address to load from page_rank2
        ".reg .u64 m28;\n\t"                   // Register for final address to load from page_rank2
        ".reg .u64 m29;\n\t"                   // Register for final address to load from page_rank2
        ".reg .u64 m30;\n\t"                   // Register for final address to load from page_rank2
        ".reg .u64 m31;\n\t"                   // Register for final address to load from page_rank2
        ".reg .u64 m32;\n\t"                   // Register for final address to load from page_rank2

        ".reg .f32 m33;\n\t"                   // (Unused) Register for atomicAdd
        ".reg .f32 m34;\n\t"                   // (Unused) Register for atomicAdd
        ".reg .f32 m35;\n\t"                   // (Unused) Register for atomicAdd
        ".reg .f32 m36;\n\t"                   // (Unused) Register for atomicAdd
        ".reg .f32 m37;\n\t"                   // (Unused) Register for atomicAdd
        ".reg .f32 m38;\n\t"                   // (Unused) Register for atomicAdd
        ".reg .f32 m39;\n\t"                   // (Unused) Register for atomicAdd
        ".reg .f32 m40;\n\t"                   // (Unused) Register for atomicAdd

        "ld.s32 m1, [%0+0];\n\t"               // Load nid
        "ld.s32 m2, [%0+4];\n\t"               // Load nid
        "ld.s32 m3, [%0+8];\n\t"               // Load nid
        "ld.s32 m4, [%0+12];\n\t"              // Load nid
        "ld.s32 m5, [%0+16];\n\t"              // Load nid
        "ld.s32 m6, [%0+20];\n\t"              // Load nid
        "ld.s32 m7, [%0+24];\n\t"              // Load nid
        "ld.s32 m8, [%0+28];\n\t"              // Load nid

        //"cvt.s64.s32 m9, m1;\n\t"              // Cast nid value
        //"cvt.s64.s32 m10, m2;\n\t"             // Cast nid value
        //"cvt.s64.s32 m11, m3;\n\t"             // Cast nid value
        //"cvt.s64.s32 m12, m4;\n\t"             // Cast nid value
        //"cvt.s64.s32 m13, m5;\n\t"             // Cast nid value
        //"cvt.s64.s32 m14, m6;\n\t"             // Cast nid value
        //"cvt.s64.s32 m15, m7;\n\t"             // Cast nid value
        //"cvt.s64.s32 m16, m8;\n\t"             // Cast nid value

        "mul.wide.s32 m17, m1, 4;\n\t"         // Multiply nid for page_rank2 address calculation
        "mul.wide.s32 m18, m2, 4;\n\t"         // Multiply nid for page_rank2 address calculation
        "mul.wide.s32 m19, m3, 4;\n\t"         // Multiply nid for page_rank2 address calculation
        "mul.wide.s32 m20, m4, 4;\n\t"         // Multiply nid for page_rank2 address calculation
        "mul.wide.s32 m21, m5, 4;\n\t"         // Multiply nid for page_rank2 address calculation
        "mul.wide.s32 m22, m6, 4;\n\t"         // Multiply nid for page_rank2 address calculation
        "mul.wide.s32 m23, m7, 4;\n\t"         // Multiply nid for page_rank2 address calculation
        "mul.wide.s32 m24, m8, 4;\n\t"         // Multiply nid for page_rank2 address calculation

        "add.u64 m25, m42, m17;\n\t"           // Final address calculation for page_rank2
        "add.u64 m26, m42, m18;\n\t"           // Final address calculation for page_rank2
        "add.u64 m27, m42, m19;\n\t"           // Final address calculation for page_rank2
        "add.u64 m28, m42, m20;\n\t"           // Final address calculation for page_rank2
        "add.u64 m29, m42, m21;\n\t"           // Final address calculation for page_rank2
        "add.u64 m30, m42, m22;\n\t"           // Final address calculation for page_rank2
        "add.u64 m31, m42, m23;\n\t"           // Final address calculation for page_rank2
        "add.u64 m32, m42, m24;\n\t"           // Final address calculation for page_rank2

        "atom.add.f32 m33, [m25], m41;\n\t"    // atomicAdd for addr0
        "atom.add.f32 m34, [m26], m41;\n\t"    // atomicAdd for addr1
        "atom.add.f32 m35, [m27], m41;\n\t"    // atomicAdd for addr2
        "atom.add.f32 m36, [m28], m41;\n\t"    // atomicAdd for addr3
        "atom.add.f32 m37, [m29], m41;\n\t"    // atomicAdd for addr4
        "atom.add.f32 m38, [m30], m41;\n\t"    // atomicAdd for addr5
        "atom.add.f32 m39, [m31], m41;\n\t"    // atomicAdd for addr6
        "atom.add.f32 m40, [m32], m41;\n\t"    // atomicAdd for addr7

        :                                      // Outputs
        : "l"(col_base_addr)                   // Inputs
      );
      
      //nid0 = col[edge];
      //nid1 = col[edge + 1];
      //nid2 = col[edge + 2];
      //nid3 = col[edge + 3];
      //nid4 = col[edge + 4];
      //nid5 = col[edge + 5];
      //nid6 = col[edge + 6];
      //nid7 = col[edge + 7];

      //addr0 = &page_rank2[nid0];
      //addr1 = &page_rank2[nid1];
      //addr2 = &page_rank2[nid2];
      //addr3 = &page_rank2[nid3];
      //addr4 = &page_rank2[nid4];
      //addr5 = &page_rank2[nid5];
      //addr6 = &page_rank2[nid6];
      //addr7 = &page_rank2[nid7];

      // Transfer the PageRank value to neighbors
      // all unrolled atomics -- can be overlapped and reordered
      // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
      // same temp reg names
      //asm volatile(// Temp Registers
      //             // m1 - m8 not used but needed for correct PTX
      //             ".reg .f32 m1;\n\t"    // temp reg m1 (atomicAdd(addr0) result)
      //             ".reg .f32 m2;\n\t"    // temp reg m2 (atomicAdd(addr1) result)
      //             ".reg .f32 m3;\n\t"    // temp reg m3 (atomicAdd(addr2) result)
      //             ".reg .f32 m4;\n\t"    // temp reg m4 (atomicAdd(addr3) result)
      //             ".reg .f32 m5;\n\t"    // temp reg m5 (atomicAdd(addr4) result)
      //             ".reg .f32 m6;\n\t"    // temp reg m6 (atomicAdd(addr5) result)
      //             ".reg .f32 m7;\n\t"    // temp reg m7 (atomicAdd(addr6) result)
      //             ".reg .f32 m8;\n\t"    // temp reg m8 (atomicAdd(addr7) result)
      //             // PTX Instructions
      //             "atom.add.f32 m1, [%0], m0;\n\t" // atomicAdd for addr0
      //             "atom.add.f32 m2, [%1], m0;\n\t" // atomicAdd for addr1
      //             "atom.add.f32 m3, [%2], m0;\n\t" // atomicAdd for addr2
      //             "atom.add.f32 m4, [%3], m0;\n\t" // atomicAdd for addr3
      //             "atom.add.f32 m5, [%4], m0;\n\t" // atomicAdd for addr4
      //             "atom.add.f32 m6, [%5], m0;\n\t" // atomicAdd for addr5
      //             "atom.add.f32 m7, [%6], m0;\n\t" // atomicAdd for addr6
      //             "atom.add.f32 m8, [%7], m0;"     // atomicAdd for addr7
      //             // no outputs
      //             // inputs
      //             :: "l"(addr0), "l"(addr1), "l"(addr2), "l"(addr3),
      //                "l"(addr4), "l"(addr5), "l"(addr6), "l"(addr7)
      //             );
    }

    for (; edge <= (end - 4); edge += 4) {
      int * const col_base_addr = &col[edge];

      asm volatile
      (
        ".reg .s32 q1;\n\t"                    // Register for nid loaded from col
        ".reg .s32 q2;\n\t"                    // Register for nid loaded from col
        ".reg .s32 q3;\n\t"                    // Register for nid loaded from col
        ".reg .s32 q4;\n\t"                    // Register for nid loaded from col

        ".reg .s64 q9;\n\t"                    // Register for casted nid value
        ".reg .s64 q10;\n\t"                   // Register for casted nid value
        ".reg .s64 q11;\n\t"                   // Register for casted nid value
        ".reg .s64 q12;\n\t"                   // Register for casted nid value

        ".reg .u64 q17;\n\t"                   // Register for multiplied nid value as address
        ".reg .u64 q18;\n\t"                   // Register for multiplied nid value as address
        ".reg .u64 q19;\n\t"                   // Register for multiplied nid value as address
        ".reg .u64 q20;\n\t"                   // Register for multiplied nid value as address

        ".reg .u64 q25;\n\t"                   // Register for final address to load from page_rank2
        ".reg .u64 q26;\n\t"                   // Register for final address to load from page_rank2
        ".reg .u64 q27;\n\t"                   // Register for final address to load from page_rank2
        ".reg .u64 q28;\n\t"                   // Register for final address to load from page_rank2

        ".reg .f32 q33;\n\t"                   // (Unused) Register for atomicAdd
        ".reg .f32 q34;\n\t"                   // (Unused) Register for atomicAdd
        ".reg .f32 q35;\n\t"                   // (Unused) Register for atomicAdd
        ".reg .f32 q36;\n\t"                   // (Unused) Register for atomicAdd

        "ld.s32 q1, [%0+0];\n\t"               // Load nid
        "ld.s32 q2, [%0+4];\n\t"               // Load nid
        "ld.s32 q3, [%0+8];\n\t"               // Load nid
        "ld.s32 q4, [%0+12];\n\t"              // Load nid

        //"cvt.s64.s32 q9, q1;\n\t"              // Cast nid value
        //"cvt.s64.s32 q10, q2;\n\t"             // Cast nid value
        //"cvt.s64.s32 q11, q3;\n\t"             // Cast nid value
        //"cvt.s64.s32 q12, q4;\n\t"             // Cast nid value

        "mul.wide.s32 q17, q1, 4;\n\t"         // Multiply nid for page_rank2 address calculation
        "mul.wide.s32 q18, q2, 4;\n\t"         // Multiply nid for page_rank2 address calculation
        "mul.wide.s32 q19, q3, 4;\n\t"         // Multiply nid for page_rank2 address calculation
        "mul.wide.s32 q20, q4, 4;\n\t"         // Multiply nid for page_rank2 address calculation

        "add.u64 q25, m42, q17;\n\t"           // Final address calculation for page_rank2
        "add.u64 q26, m42, q18;\n\t"           // Final address calculation for page_rank2
        "add.u64 q27, m42, q19;\n\t"           // Final address calculation for page_rank2
        "add.u64 q28, m42, q20;\n\t"           // Final address calculation for page_rank2

        "atom.add.f32 q33, [q25], m41;\n\t"    // atomicAdd for addr0
        "atom.add.f32 q34, [q26], m41;\n\t"    // atomicAdd for addr1
        "atom.add.f32 q35, [q27], m41;\n\t"    // atomicAdd for addr2
        "atom.add.f32 q36, [q28], m41;\n\t"    // atomicAdd for addr3

        :                                      // Outputs
        : "l"(col_base_addr)                   // Inputs
      );
    }

    for (; edge <= (end - 2); edge += 2) {
      int * const col_base_addr = &col[edge];

      asm volatile
      (
        ".reg .s32 t1;\n\t"                    // Register for nid loaded from col
        ".reg .s32 t2;\n\t"                    // Register for nid loaded from col

        ".reg .s64 t9;\n\t"                    // Register for casted nid value
        ".reg .s64 t10;\n\t"                   // Register for casted nid value

        ".reg .u64 t17;\n\t"                   // Register for multiplied nid value as address
        ".reg .u64 t18;\n\t"                   // Register for multiplied nid value as address

        ".reg .u64 t25;\n\t"                   // Register for final address to load from page_rank2
        ".reg .u64 t26;\n\t"                   // Register for final address to load from page_rank2

        ".reg .f32 t33;\n\t"                   // (Unused) Register for atomicAdd
        ".reg .f32 t34;\n\t"                   // (Unused) Register for atomicAdd

        "ld.s32 t1, [%0+0];\n\t"               // Load nid
        "ld.s32 t2, [%0+4];\n\t"               // Load nid

        //"cvt.s64.s32 t9, t1;\n\t"              // Cast nid value
        //"cvt.s64.s32 t10, t2;\n\t"             // Cast nid value

        "mul.wide.s32 t17, t1, 4;\n\t"         // Multiply nid for page_rank2 address calculation
        "mul.wide.s32 t18, t2, 4;\n\t"         // Multiply nid for page_rank2 address calculation

        "add.u64 t25, m42, t17;\n\t"           // Final address calculation for page_rank2
        "add.u64 t26, m42, t18;\n\t"           // Final address calculation for page_rank2

        "atom.add.f32 t33, [t25], m41;\n\t"    // atomicAdd for addr0
        "atom.add.f32 t34, [t26], m41;\n\t"    // atomicAdd for addr1

        :                                      // Outputs
        : "l"(col_base_addr)                   // Inputs
      );
    }

    for (; edge < end; ++edge) {
      int * const col_base_addr = &col[edge];

      asm volatile
      (
        ".reg .s32 a1;\n\t"                    // Register for nid loaded from col
        ".reg .s64 a9;\n\t"                    // Register for casted nid value
        ".reg .u64 a17;\n\t"                   // Register for multiplied nid value as address
        ".reg .u64 a25;\n\t"                   // Register for final address to load from page_rank2
        ".reg .f32 a33;\n\t"                   // (Unused) Register for atomicAdd

        "ld.s32 a1, [%0+0];\n\t"               // Load nid
        //"cvt.s64.s32 a9, a1;\n\t"              // Cast nid value
        "mul.wide.s32 a17, a1, 4;\n\t"         // Multiply nid for page_rank2 address calculation
        "add.u64 a25, m42, a17;\n\t"           // Final address calculation for page_rank2
        "atom.add.f32 a33, [a25], m41;\n\t"    // atomicAdd for addr0

        :                                      // Outputs
        : "l"(col_base_addr)                   // Inputs
      );
    }
  }
/*
  if (threadIdx.x == 0) {
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(READ_ONLY_REGION);
    __denovo_gpuEpilogue(pgRk2Reg); // written with atomics
    __denovo_gpuEpilogue(pgRk1Reg);
    __denovo_gpuEpilogue(rowColReg);
  }
  */
}

/**
 * @brief   pagerank2
 * @param   row         csr pointer array
 * @param   col         csr column array
 * @param   data        weight array
 * @param   page_rank1  pagerank array 1
 * @param   page_rank2  pagerank array 2
 * @param   num_nodes   number of vertices
 * @param   num_edges   number of edges
 * @param   pgRk1Reg    pagerank array 1 region
 * @param   pgRk2Reg    pagerank array 2 region
 */
__global__ void
pagerank2(int * row, int * col, int * data, float * page_rank1,
          float * page_rank2, const int num_nodes, const int num_edges)
{
  // Get my workitem id
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
/*
  if (threadIdx.x == 0) {
    __denovo_setAcquireRegion(SPECIAL_REGION);
    __denovo_addAcquireRegion(READ_ONLY_REGION);
    __denovo_addAcquireRegion(pgRk1Reg);
    __denovo_addAcquireRegion(pgRk2Reg);
    //__denovo_addAcquireRegion(READ_ONLY_REGION); // include to be safe, not used in this kernel
  }
  __syncthreads();
*/
  // Update pagerank value with the damping factor
  for (; tid < num_nodes; tid += blockDim.x * gridDim.x) {
    /*
      To conform to the C++ rules for atomics, all accesses to page_rank2 must
      be atomic.  We want to do an unpaired atomic load and then an unpaired
      atomic store, so use atomicAdd + 0 to approximate an unpaired atomic load
      and reprogrammed atomicOr for the unpaired atomic store -- no release
      semantics needed here.
     */
    page_rank1[tid]	= 0.15 / (float)num_nodes + 0.85 * atomicAdd(&(page_rank2[tid]), 0);
    atomicOr((int *)&(page_rank2[tid]), 0); //page_rank2[tid] = 0.0f;
  }
/*
  if (threadIdx.x == 0) {
    __denovo_gpuEpilogue(SPECIAL_REGION);
    __denovo_gpuEpilogue(pgRk1Reg); // written with data stores
    __denovo_gpuEpilogue(pgRk2Reg); // written with atomics
    __denovo_gpuEpilogue(READ_ONLY_REGION); // include to be safe, not used in this kernel
  }
  */
}

/**
 * @brief   inibuffer
 * @param   row         csr pointer array
 * @param   page_rank1  pagerank array 1
 * @param   page_rank2  pagerank array 2
 * @param   num_nodes   number of vertices
 */
 __global__ void
 inibuffer(int *row, float *page_rank1, float *page_rank2, const int num_nodes,
           const int num_edges)
 {
     // Get my thread id
     int tid = blockDim.x * blockIdx.x + threadIdx.x;
 
     if (tid < num_nodes) {
         page_rank1[tid] = 1 / (float)num_nodes;
         page_rank2[tid] = 0.0f;
     }
 }

///**
// * @brief   inibuffer
// * @param   row         csr pointer array
// * @param   page_rank1  pagerank array 1
// * @param   page_rank2  pagerank array 2
// * @param   num_nodes   number of vertices
// * @param   pgRk1Reg    pagerank array 1 region
// * @param   pgRk2Reg    pagerank array 2 region
// */
//__global__ void
//inibuffer(int * row, float * page_rank1, float * page_rank2,
//          const int num_nodes, const int num_edges, const region_t pgRk1Reg,
//          const region_t pgRk2Reg)
//{
//  // Get my thread id
//  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
//
//  /*
//    On acquires, need to invalidate all of the regions that are read in
//    the critical section.  For this kernel, no arrays are read, so
//    just invalidate special region.
//  */
//  if (threadIdx.x == 0) {
//    __denovo_setAcquireRegion(SPECIAL_REGION);
//    __denovo_addAcquireRegion(pgRk1Reg);
//    __denovo_addAcquireRegion(pgRk2Reg);
//    __denovo_addAcquireRegion(READ_ONLY_REGION); // include to be safe, not used in this kernel
//  }
//  __syncthreads();
//
//  if (tid < num_nodes) {
//    page_rank1[tid] = 1 / (float)num_nodes;
//    /*
//      To conform to the C++ rules for atomics, all accesses to page_rank2 must
//      be atomic.  We want to do an unpaired atomic store, so use reprogrammed
//      atomicOr for this -- no release semantics needed here.
//    */
//    atomicOr((int *)&(page_rank2[tid]), 0); //page_rank2[tid] = 0.0f;
//  }
//
//  if (threadIdx.x == 0) {
//    __denovo_gpuEpilogue(SPECIAL_REGION);
//    __denovo_gpuEpilogue(pgRk1Reg); // written with data stores
//    __denovo_gpuEpilogue(pgRk2Reg); // written with atomics
//    __denovo_gpuEpilogue(READ_ONLY_REGION); // include to be safe, not used in this kernel
//  }
//}
