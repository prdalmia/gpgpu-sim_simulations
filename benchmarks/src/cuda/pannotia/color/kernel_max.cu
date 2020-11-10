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
 * the U.S. Export Administration Regulations ("EAR"�) (15 C.F.R Sections 730-774),  *
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


//include "denovo_util.h"
//#include "gpuKernels_util.cu"

__global__ void color1(int *row, int *col, int *node_value, int *color_array,
    int *stop, int *max_d, const int color,
    const int num_nodes, const int num_edges)
{
    // Get my thread workitem id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cont_tid = false;
    int edge = 0;
/*
    if (threadIdx.x == 0) {
        __denovo_setAcquireRegion(SPECIAL_REGION);
        __denovo_addAcquireRegion(READ_ONLY_REGION);
        __denovo_addAcquireRegion(default_reg);
        __denovo_addAcquireRegion(max_reg);
    }
    __syncthreads();
*/

    for (; tid < num_nodes; tid += blockDim.x * gridDim.x) {
        // If the vertex is still not colored
        if (color_array[tid] == -1) {
            // Get the start and end pointer of the neighbor list
            const int row_start = row[tid];
            const int row_end = row[tid + 1];
            const int out_deg = row_end - row_start;

            int * const max_tid_addr = &max_d[tid];
            int maximum = -1;

            asm volatile
            (
              // Temp Register
              ".reg .u64 m99;\n\t"                             // Temp reg
              ".reg .s32 m100;\n\t"                            // Temp reg
              ".reg .s32 m101;\n\t"                            // Temp reg
              ".reg .u64 m102;\n\t"                            // Temp reg
              ".reg .u64 m103;\n\t"                            // Temp reg

              "mov.u64 m99, %0;\n\t"                           
              "mov.s32 m100, %1;\n\t"                         
              "mov.s32 m101, %2;\n\t"                        
              "mov.u64 m102, %3;\n\t"                        
              "mov.u64 m103, %4;\n\t"                        

              :                                                 // No outputs
              : "l"(max_tid_addr), "r"(cont_tid), "r"(maximum), // Inputs
                "l"(color_array), "l"(node_value)
            );

            if (out_deg <= 1) {
                goto COLOR1_PULL_MAX_BODY;
            }

            // Navigate the neighbor list
            // for (int edge = start; edge < end; edge++) {
            //     // Determine if the vertex value is the maximum in the neighborhood
            //     const int nid = col[edge];

            //     if (color_array[nid] == -1 && out_deg > 1) {
            //         cont_tid = true;
            //         maximum = max(maximum, node_value[nid]);
            //     }
            // }

            // // Assign maximum the max array
            // max_d[tid] = maximum;

            for (edge = row_start; edge <= (row_end - 8); edge += 8) {
                int * const col_base_addr = &col[edge];

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

                    ".reg .u64 m9;\n\t"                                  // Register for multiplied nid value as address
                    ".reg .u64 m10;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m11;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m12;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m13;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m14;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m15;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m16;\n\t"                                 // Register for multiplied nid value as address

                    ".reg .u64 m17;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 m18;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 m19;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 m20;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 m21;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 m22;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 m23;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 m24;\n\t"                                 // Register for final address to load from color

                    ".reg .u64 m25;\n\t"                                 // Register for final address to load from node_value
                    ".reg .u64 m26;\n\t"                                 // Register for final address to load from node_value
                    ".reg .u64 m27;\n\t"                                 // Register for final address to load from node_value
                    ".reg .u64 m28;\n\t"                                 // Register for final address to load from node_value
                    ".reg .u64 m29;\n\t"                                 // Register for final address to load from node_value
                    ".reg .u64 m30;\n\t"                                 // Register for final address to load from node_value
                    ".reg .u64 m31;\n\t"                                 // Register for final address to load from node_value
                    ".reg .u64 m32;\n\t"                                 // Register for final address to load from node_value

                    ".reg .s32 m69;\n\t"                                 // Register for color
                    ".reg .s32 m70;\n\t"                                 // Register for color
                    ".reg .s32 m71;\n\t"                                 // Register for color
                    ".reg .s32 m72;\n\t"                                 // Register for color
                    ".reg .s32 m73;\n\t"                                 // Register for color
                    ".reg .s32 m74;\n\t"                                 // Register for color
                    ".reg .s32 m75;\n\t"                                 // Register for color
                    ".reg .s32 m76;\n\t"                                 // Register for color

                    ".reg .s32 m79;\n\t"                                 // Register for node_value
                    ".reg .s32 m80;\n\t"                                 // Register for node_value
                    ".reg .s32 m81;\n\t"                                 // Register for node_value
                    ".reg .s32 m82;\n\t"                                 // Register for node_value
                    ".reg .s32 m83;\n\t"                                 // Register for node_value
                    ".reg .s32 m84;\n\t"                                 // Register for node_value
                    ".reg .s32 m85;\n\t"                                 // Register for node_value
                    ".reg .s32 m86;\n\t"                                 // Register for node_value

                    ".reg .pred m41;\n\t"                                // Register for predicate
                    ".reg .pred m42;\n\t"                                // Register for predicate
                    ".reg .pred m43;\n\t"                                // Register for predicate
                    ".reg .pred m44;\n\t"                                // Register for predicate
                    ".reg .pred m45;\n\t"                                // Register for predicate
                    ".reg .pred m46;\n\t"                                // Register for predicate
                    ".reg .pred m47;\n\t"                                // Register for predicate
                    ".reg .pred m48;\n\t"                                // Register for predicate

                    "ld.s32 m1, [%0+0];\n\t"                             // Load nid
                    "ld.s32 m2, [%0+4];\n\t"                             // Load nid
                    "ld.s32 m3, [%0+8];\n\t"                             // Load nid
                    "ld.s32 m4, [%0+12];\n\t"                            // Load nid
                    "ld.s32 m5, [%0+16];\n\t"                            // Load nid
                    "ld.s32 m6, [%0+20];\n\t"                            // Load nid
                    "ld.s32 m7, [%0+24];\n\t"                            // Load nid
                    "ld.s32 m8, [%0+28];\n\t"                            // Load nid

                    "mul.wide.s32 m9, m1, 4;\n\t"                        // Multiply nid for x address calculation
                    "mul.wide.s32 m10, m2, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 m11, m3, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 m12, m4, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 m13, m5, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 m14, m6, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 m15, m7, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 m16, m8, 4;\n\t"                       // Multiply nid for x address calculation

                    "add.u64 m17, m102, m9;\n\t"                         // Final address calculation for color
                    "add.u64 m18, m102, m10;\n\t"                        // Final address calculation for color
                    "add.u64 m19, m102, m11;\n\t"                        // Final address calculation for color
                    "add.u64 m20, m102, m12;\n\t"                        // Final address calculation for color
                    "add.u64 m21, m102, m13;\n\t"                        // Final address calculation for color
                    "add.u64 m22, m102, m14;\n\t"                        // Final address calculation for color
                    "add.u64 m23, m102, m15;\n\t"                        // Final address calculation for color
                    "add.u64 m24, m102, m16;\n\t"                        // Final address calculation for color

                    "add.u64 m25, m103, m9;\n\t"                         // Final address calculation for node_value
                    "add.u64 m26, m103, m10;\n\t"                        // Final address calculation for node_value
                    "add.u64 m27, m103, m11;\n\t"                        // Final address calculation for node_value
                    "add.u64 m28, m103, m12;\n\t"                        // Final address calculation for node_value
                    "add.u64 m29, m103, m13;\n\t"                        // Final address calculation for node_value
                    "add.u64 m30, m103, m14;\n\t"                        // Final address calculation for node_value
                    "add.u64 m31, m103, m15;\n\t"                        // Final address calculation for node_value
                    "add.u64 m32, m103, m16;\n\t"                        // Final address calculation for node_value

                    "ld.s32 m69, [m17];\n\t"                             // Load color
                    "ld.s32 m70, [m18];\n\t"                             // Load color
                    "ld.s32 m71, [m19];\n\t"                             // Load color
                    "ld.s32 m72, [m20];\n\t"                             // Load color
                    "ld.s32 m73, [m21];\n\t"                             // Load color
                    "ld.s32 m74, [m22];\n\t"                             // Load color
                    "ld.s32 m75, [m23];\n\t"                             // Load color
                    "ld.s32 m76, [m24];\n\t"                             // Load color

                    "ld.s32 m79, [m25];\n\t"                             // Load node_value
                    "ld.s32 m80, [m26];\n\t"                             // Load node_value
                    "ld.s32 m81, [m27];\n\t"                             // Load node_value
                    "ld.s32 m82, [m28];\n\t"                             // Load node_value
                    "ld.s32 m83, [m29];\n\t"                             // Load node_value
                    "ld.s32 m84, [m30];\n\t"                             // Load node_value
                    "ld.s32 m85, [m31];\n\t"                             // Load node_value
                    "ld.s32 m86, [m32];\n\t"                             // Load node_value

                    "setp.eq.s32 m41, m69, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 m42, m70, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 m43, m71, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 m44, m72, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 m45, m73, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 m46, m74, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 m47, m75, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 m48, m76, -1;\n\t"                      // Do predicate

                    "COLOR1_PULL_PRED_BODY_8_1:\n\t"
                    "@!m41 bra COLOR1_PULL_PRED_BODY_8_2;\n\t"           // Check if node is processed
                    "max.s32 m101, m101, m79;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PULL_PRED_BODY_8_2:\n\t"
                    "@!m42 bra COLOR1_PULL_PRED_BODY_8_3;\n\t"           // Check if node is processed
                    "max.s32 m101, m101, m80;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PULL_PRED_BODY_8_3:\n\t"
                    "@!m43 bra COLOR1_PULL_PRED_BODY_8_4;\n\t"           // Check if node is processed
                    "max.s32 m101, m101, m81;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PULL_PRED_BODY_8_4:\n\t"
                    "@!m44 bra COLOR1_PULL_PRED_BODY_8_5;\n\t"           // Check if node is processed
                    "max.s32 m101, m101, m82;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PULL_PRED_BODY_8_5:\n\t"
                    "@!m45 bra COLOR1_PULL_PRED_BODY_8_6;\n\t"           // Check if node is processed
                    "max.s32 m101, m101, m83;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PULL_PRED_BODY_8_6:\n\t"
                    "@!m46 bra COLOR1_PULL_PRED_BODY_8_7;\n\t"           // Check if node is processed
                    "max.s32 m101, m101, m84;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PULL_PRED_BODY_8_7:\n\t"
                    "@!m47 bra COLOR1_PULL_PRED_BODY_8_8;\n\t"           // Check if node is processed
                    "max.s32 m101, m101, m85;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PULL_PRED_BODY_8_8:\n\t"
                    "@!m48 bra COLOR1_PULL_NEIGH_END_8;\n\t"             // Check if node is processed
                    "max.s32 m101, m101, m86;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PULL_NEIGH_END_8:\n\t"

                    :                                                    // Outputs
                    : "l"(col_base_addr)                                 // Inputs
                );
            }

            for (; edge <= (row_end - 4); edge += 4) {
                int * const col_base_addr = &col[edge];

                asm volatile
                (
                    ".reg .s32 q1;\n\t"                                  // Register for nid loaded from col
                    ".reg .s32 q2;\n\t"                                  // Register for nid loaded from col
                    ".reg .s32 q3;\n\t"                                  // Register for nid loaded from col
                    ".reg .s32 q4;\n\t"                                  // Register for nid loaded from col

                    ".reg .u64 q9;\n\t"                                  // Register for multiplied nid value as address
                    ".reg .u64 q10;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 q11;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 q12;\n\t"                                 // Register for multiplied nid value as address

                    ".reg .u64 q17;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 q18;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 q19;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 q20;\n\t"                                 // Register for final address to load from color

                    ".reg .u64 q25;\n\t"                                 // Register for final address to load from node_value
                    ".reg .u64 q26;\n\t"                                 // Register for final address to load from node_value
                    ".reg .u64 q27;\n\t"                                 // Register for final address to load from node_value
                    ".reg .u64 q28;\n\t"                                 // Register for final address to load from node_value

                    ".reg .s32 q69;\n\t"                                 // Register for color
                    ".reg .s32 q70;\n\t"                                 // Register for color
                    ".reg .s32 q71;\n\t"                                 // Register for color
                    ".reg .s32 q72;\n\t"                                 // Register for color

                    ".reg .s32 q79;\n\t"                                 // Register for node_value
                    ".reg .s32 q80;\n\t"                                 // Register for node_value
                    ".reg .s32 q81;\n\t"                                 // Register for node_value
                    ".reg .s32 q82;\n\t"                                 // Register for node_value

                    ".reg .pred q41;\n\t"                                // Register for predicate
                    ".reg .pred q42;\n\t"                                // Register for predicate
                    ".reg .pred q43;\n\t"                                // Register for predicate
                    ".reg .pred q44;\n\t"                                // Register for predicate

                    "ld.s32 q1, [%0+0];\n\t"                             // Load nid
                    "ld.s32 q2, [%0+4];\n\t"                             // Load nid
                    "ld.s32 q3, [%0+8];\n\t"                             // Load nid
                    "ld.s32 q4, [%0+12];\n\t"                            // Load nid

                    "mul.wide.s32 q9, q1, 4;\n\t"                        // Multiply nid for x address calculation
                    "mul.wide.s32 q10, q2, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 q11, q3, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 q12, q4, 4;\n\t"                       // Multiply nid for x address calculation

                    "add.u64 q17, m102, q9;\n\t"                         // Final address calculation for color
                    "add.u64 q18, m102, q10;\n\t"                        // Final address calculation for color
                    "add.u64 q19, m102, q11;\n\t"                        // Final address calculation for color
                    "add.u64 q20, m102, q12;\n\t"                        // Final address calculation for color

                    "add.u64 q25, m103, q9;\n\t"                         // Final address calculation for node_value
                    "add.u64 q26, m103, q10;\n\t"                        // Final address calculation for node_value
                    "add.u64 q27, m103, q11;\n\t"                        // Final address calculation for node_value
                    "add.u64 q28, m103, q12;\n\t"                        // Final address calculation for node_value

                    "ld.s32 q69, [q17];\n\t"                             // Load color
                    "ld.s32 q70, [q18];\n\t"                             // Load color
                    "ld.s32 q71, [q19];\n\t"                             // Load color
                    "ld.s32 q72, [q20];\n\t"                             // Load color

                    "ld.s32 q79, [q25];\n\t"                             // Load node_value
                    "ld.s32 q80, [q26];\n\t"                             // Load node_value
                    "ld.s32 q81, [q27];\n\t"                             // Load node_value
                    "ld.s32 q82, [q28];\n\t"                             // Load node_value

                    "setp.eq.s32 q41, q69, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 q42, q70, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 q43, q71, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 q44, q72, -1;\n\t"                      // Do predicate

                    "COLOR1_PULL_PRED_BODY_4_1:\n\t"
                    "@!q41 bra COLOR1_PULL_PRED_BODY_4_2;\n\t"           // Check if node is processed
                    "max.s32 m101, m101, q79;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PULL_PRED_BODY_4_2:\n\t"
                    "@!q42 bra COLOR1_PULL_PRED_BODY_4_3;\n\t"           // Check if node is processed
                    "max.s32 m101, m101, q80;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PULL_PRED_BODY_4_3:\n\t"
                    "@!q43 bra COLOR1_PULL_PRED_BODY_4_4;\n\t"           // Check if node is processed
                    "max.s32 m101, m101, q81;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PULL_PRED_BODY_4_4:\n\t"
                    "@!q44 bra COLOR1_PULL_NEIGH_END_4;\n\t"             // Check if node is processed
                    "max.s32 m101, m101, q82;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PULL_NEIGH_END_4:\n\t"

                    :                                                    // Outputs
                    : "l"(col_base_addr)                                 // Inputs
                );
            }

            for (; edge <= (row_end - 2); edge += 2) {
                int * const col_base_addr = &col[edge];

                asm volatile
                (
                    ".reg .s32 t1;\n\t"                                  // Register for nid loaded from col
                    ".reg .s32 t2;\n\t"                                  // Register for nid loaded from col

                    ".reg .u64 t9;\n\t"                                  // Register for multiplied nid value as address
                    ".reg .u64 t10;\n\t"                                 // Register for multiplied nid value as address

                    ".reg .u64 t17;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 t18;\n\t"                                 // Register for final address to load from color

                    ".reg .u64 t25;\n\t"                                 // Register for final address to load from node_value
                    ".reg .u64 t26;\n\t"                                 // Register for final address to load from node_value

                    ".reg .s32 t69;\n\t"                                 // Register for color
                    ".reg .s32 t70;\n\t"                                 // Register for color

                    ".reg .s32 t79;\n\t"                                 // Register for node_value
                    ".reg .s32 t80;\n\t"                                 // Register for node_value

                    ".reg .pred t41;\n\t"                                // Register for predicate
                    ".reg .pred t42;\n\t"                                // Register for predicate

                    "ld.s32 t1, [%0+0];\n\t"                             // Load nid
                    "ld.s32 t2, [%0+4];\n\t"                             // Load nid

                    "mul.wide.s32 t9, t1, 4;\n\t"                        // Multiply nid for x address calculation
                    "mul.wide.s32 t10, t2, 4;\n\t"                       // Multiply nid for x address calculation

                    "add.u64 t17, m102, t9;\n\t"                         // Final address calculation for color
                    "add.u64 t18, m102, t10;\n\t"                        // Final address calculation for color

                    "add.u64 t25, m103, t9;\n\t"                         // Final address calculation for node_value
                    "add.u64 t26, m103, t10;\n\t"                        // Final address calculation for node_value

                    "ld.s32 t69, [t17];\n\t"                             // Load color
                    "ld.s32 t70, [t18];\n\t"                             // Load color

                    "ld.s32 t79, [t25];\n\t"                             // Load node_value
                    "ld.s32 t80, [t26];\n\t"                             // Load node_value

                    "setp.eq.s32 t41, t69, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 t42, t70, -1;\n\t"                      // Do predicate

                    "COLOR1_PULL_PRED_BODY_2_1:\n\t"
                    "@!t41 bra COLOR1_PULL_PRED_BODY_2_2;\n\t"           // Check if node is processed
                    "max.s32 m101, m101, t79;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PULL_PRED_BODY_2_2:\n\t"
                    "@!t42 bra COLOR1_PULL_NEIGH_END_2;\n\t"             // Check if node is processed
                    "max.s32 m101, m101, t80;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PULL_NEIGH_END_2:\n\t"

                    :                                                    // Outputs
                    : "l"(col_base_addr)                                 // Inputs
                );
            }

            for (; edge < row_end; edge++) {
                int * const col_base_addr = &col[edge];

                asm volatile
                (
                    "{\n\t"
                    ".reg .s32 a1;\n\t"                                  // Register for nid loaded from col
                    ".reg .u64 a9;\n\t"                                  // Register for multiplied nid value as address
                    ".reg .u64 a17;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 a25;\n\t"                                 // Register for final address to load from node_value
                    ".reg .s32 a69;\n\t"                                 // Register for color
                    ".reg .s32 a79;\n\t"                                 // Register for node_value
                    ".reg .pred a41;\n\t"                                // Register for predicate

                    "ld.s32 a1, [%0+0];\n\t"                             // Load nid
                    "mul.wide.s32 a9, a1, 4;\n\t"                        // Multiply nid for x address calculation
                    "add.u64 a17, m102, a9;\n\t"                         // Final address calculation for color
                    "add.u64 a25, m103, a9;\n\t"                         // Final address calculation for node_value
                    "ld.s32 a69, [a17];\n\t"                             // Load color
                    "ld.s32 a79, [a25];\n\t"                             // Load node_value
                    "setp.eq.s32 a41, a69, -1;\n\t"                      // Do predicate

                    "COLOR1_PULL_PRED_BODY_1_1:\n\t"
                    "@!a41 bra COLOR1_PULL_NEIGH_END_1;\n\t"             // Check if node is processed
                    "max.s32 m101, m101, a79;\n\t"                       // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont
                    "COLOR1_PULL_NEIGH_END_1:\n\t"
                    "}"
                    :                                                    // Outputs
                    : "l"(col_base_addr)                                 // Inputs
                );
            }

COLOR1_PULL_MAX_BODY:
            asm volatile
            (
              "st.s32 [m99], m101;\n\t"                        // Store maximum in max_d
              "mov.s32 %0, m100;\n\t"                          // Store local_cont

              : "=r"(cont_tid)                                 // No outputs
              :                                                // Inputs
            );
        }
    }

    //cont[blockIdx.x * blockDim.x + threadIdx.x] = cont_tid;
/*
    if (threadIdx.x == 0) {
        __denovo_gpuEpilogue(SPECIAL_REGION);
        __denovo_gpuEpilogue(READ_ONLY_REGION);
        __denovo_gpuEpilogue(default_reg);
        __denovo_gpuEpilogue(max_reg);
    }
*/
}


// __global__ void color1(int *row, int *col, int *node_value,
//                        int *col_cnt, int *color_array,
//                        int *cont, int *max_d, const int color,
//                        const int num_nodes, const int num_edges,
//                        region_t default_reg, region_t max_reg)
// {
//     // Get my thread workitem id
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int cont_tid = false;
// 
//     if (threadIdx.x == 0) {
//         __denovo_setAcquireRegion(SPECIAL_REGION);
//         __denovo_addAcquireRegion(READ_ONLY_REGION);
//         __denovo_addAcquireRegion(default_reg);
//         __denovo_addAcquireRegion(max_reg);
//     }
//     __syncthreads();
// 
//     for (; tid < num_nodes; tid += blockDim.x * gridDim.x) {
//         // If the vertex is still not colored
//         if (color_array[tid] == -1) {
//             // Get the start and end pointer of the neighbor list
//             const int start = row[tid];
//             const int end = row[tid + 1];
//             const int out_deg = end - start;
// 
//             int maximum = -1;
// 
//             // Navigate the neighbor list
//             for (int edge = start; edge < end; edge++) {
//                 // Determine if the vertex value is the maximum in the neighborhood
//                 const int nid = col[edge];
// 
//                 if (color_array[nid] == -1 && out_deg > 1) {
//                     cont_tid = true;
//                     maximum = max(maximum, node_value[nid]);
//                 }
//             }
// 
//             // Assign maximum the max array
//             max_d[tid] = maximum;
//         }
//     }
// 
//     cont[blockIdx.x * blockDim.x + threadIdx.x] = cont_tid;
// 
//     if (threadIdx.x == 0) {
//         __denovo_gpuEpilogue(SPECIAL_REGION);
//         __denovo_gpuEpilogue(READ_ONLY_REGION);
//         __denovo_gpuEpilogue(default_reg);
//         __denovo_gpuEpilogue(max_reg);
//     }
// }


__global__ void color2(int *node_value, int *color_array, int *max_d,
                       const int color, const int num_nodes,
                       const int num_edgesg)
{
    // Get my workitem id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
/*
    if (threadIdx.x == 0) {
        __denovo_setAcquireRegion(SPECIAL_REGION);
        __denovo_addAcquireRegion(READ_ONLY_REGION);
        __denovo_addAcquireRegion(default_reg);
        __denovo_addAcquireRegion(max_reg);
    }
    __syncthreads();
*/
    for (; tid < num_nodes; tid += blockDim.x * gridDim.x) {
        // If the vertex is still not colored
        if (color_array[tid] == -1) {
            // even though not racy here, max_d is racy in color1_push
            // so need to use atomic
            const int max_neighbor = max_d[tid];

            if (node_value[tid] >= max_neighbor) {
                // Assign a color
                color_array[tid] = color;
            }
        }
    }
/*
    if (threadIdx.x == 0) {
        __denovo_gpuEpilogue(SPECIAL_REGION);
        __denovo_gpuEpilogue(READ_ONLY_REGION);
        __denovo_gpuEpilogue(default_reg);
        __denovo_gpuEpilogue(max_reg);
    }
*/
}


// push version of color1- use atomicMax to update every neighbor for assigned node
// with max of current value and this nodes value
__global__ void color1_push(int *row, int *col, int *node_value,
                            int *col_cnt, int *color_array,
                            int *cont, int *max_d, const int color,
                            const int num_nodes, const int num_edges)
{
    // Get my thread workitem id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cont_tid = false;
    int edge = 0;
/*
    if (threadIdx.x == 0) {
        __denovo_setAcquireRegion(SPECIAL_REGION);
        __denovo_addAcquireRegion(READ_ONLY_REGION);
        __denovo_addAcquireRegion(default_reg);
        __denovo_addAcquireRegion(max_reg);
    }
    __syncthreads();
*/
    for (; tid < num_nodes; tid += blockDim.x * gridDim.x) {
        // If the vertex is still not colored
        if (color_array[tid] == -1) {
            // Get the start and end pointer of the neighbor list
            const int row_start = row[tid];
            const int row_end = row[tid + 1];

            const int this_node_val = node_value[tid];

            asm volatile
            (
              // Temp Register
              ".reg .s32 m99;\n\t"                              // Temp reg
              ".reg .s32 m100;\n\t"                             // Temp reg
              ".reg .u64 m101;\n\t"                             // Temp reg
              ".reg .u64 m102;\n\t"                             // Temp reg

              "mov.s32 m99, %0;\n\t"                           
              "mov.s32 m100, %1;\n\t"                         
              "mov.u64 m101, %2;\n\t"                        
              "mov.u64 m102, %3;\n\t"                        

              :                                                  // No outputs
              : "r"(this_node_val), "r"(cont_tid),               // Inputs
                "l"(color_array), "l"(max_d)
            );

            //// Navigate the neighbor list, update max with this value
            //for (int edge = start; edge < end; edge++) {
            //    const int nid = col[edge];
            //    const int neigh_out_deg = col_cnt[edge];

            //    // Determine if the vertex value is the maximum in the neighborhood
            //    if (color_array[nid] == -1 && neigh_out_deg > 1) {
            //        cont_tid = true;
            //        atomicMax(&max_d[nid], this_node_val);
            //    }
            //}

            for (edge = row_start; edge <= (row_end - 8); edge += 8) {
                int * const col_base_addr = &col[edge];
                int * const col_cnt_base_addr = &col_cnt[edge];

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

                    ".reg .s32 m33;\n\t"                                 // Register for out_deg loaded from col_cnt
                    ".reg .s32 m34;\n\t"                                 // Register for out_deg loaded from col_cnt
                    ".reg .s32 m35;\n\t"                                 // Register for out_deg loaded from col_cnt
                    ".reg .s32 m36;\n\t"                                 // Register for out_deg loaded from col_cnt
                    ".reg .s32 m37;\n\t"                                 // Register for out_deg loaded from col_cnt
                    ".reg .s32 m38;\n\t"                                 // Register for out_deg loaded from col_cnt
                    ".reg .s32 m39;\n\t"                                 // Register for out_deg loaded from col_cnt
                    ".reg .s32 m40;\n\t"                                 // Register for out_deg loaded from col_cnt

                    ".reg .u64 m9;\n\t"                                  // Register for multiplied nid value as address
                    ".reg .u64 m10;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m11;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m12;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m13;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m14;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m15;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m16;\n\t"                                 // Register for multiplied nid value as address

                    ".reg .u64 m17;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 m18;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 m19;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 m20;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 m21;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 m22;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 m23;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 m24;\n\t"                                 // Register for final address to load from color

                    ".reg .u64 m25;\n\t"                                 // Register for final address to load from max_d
                    ".reg .u64 m26;\n\t"                                 // Register for final address to load from max_d
                    ".reg .u64 m27;\n\t"                                 // Register for final address to load from max_d
                    ".reg .u64 m28;\n\t"                                 // Register for final address to load from max_d
                    ".reg .u64 m29;\n\t"                                 // Register for final address to load from max_d
                    ".reg .u64 m30;\n\t"                                 // Register for final address to load from max_d
                    ".reg .u64 m31;\n\t"                                 // Register for final address to load from max_d
                    ".reg .u64 m32;\n\t"                                 // Register for final address to load from max_d

                    ".reg .s32 m69;\n\t"                                 // Register for color
                    ".reg .s32 m70;\n\t"                                 // Register for color
                    ".reg .s32 m71;\n\t"                                 // Register for color
                    ".reg .s32 m72;\n\t"                                 // Register for color
                    ".reg .s32 m73;\n\t"                                 // Register for color
                    ".reg .s32 m74;\n\t"                                 // Register for color
                    ".reg .s32 m75;\n\t"                                 // Register for color
                    ".reg .s32 m76;\n\t"                                 // Register for color

                    ".reg .s32 m79;\n\t"                                 // Register for max_d
                    ".reg .s32 m80;\n\t"                                 // Register for max_d
                    ".reg .s32 m81;\n\t"                                 // Register for max_d
                    ".reg .s32 m82;\n\t"                                 // Register for max_d
                    ".reg .s32 m83;\n\t"                                 // Register for max_d
                    ".reg .s32 m84;\n\t"                                 // Register for max_d
                    ".reg .s32 m85;\n\t"                                 // Register for max_d
                    ".reg .s32 m86;\n\t"                                 // Register for max_d

                    ".reg .pred m41;\n\t"                                // Register for predicate
                    ".reg .pred m42;\n\t"                                // Register for predicate
                    ".reg .pred m43;\n\t"                                // Register for predicate
                    ".reg .pred m44;\n\t"                                // Register for predicate
                    ".reg .pred m45;\n\t"                                // Register for predicate
                    ".reg .pred m46;\n\t"                                // Register for predicate
                    ".reg .pred m47;\n\t"                                // Register for predicate
                    ".reg .pred m48;\n\t"                                // Register for predicate

                    ".reg .pred m49;\n\t"                                // Register for predicate
                    ".reg .pred m50;\n\t"                                // Register for predicate
                    ".reg .pred m51;\n\t"                                // Register for predicate
                    ".reg .pred m52;\n\t"                                // Register for predicate
                    ".reg .pred m53;\n\t"                                // Register for predicate
                    ".reg .pred m54;\n\t"                                // Register for predicate
                    ".reg .pred m55;\n\t"                                // Register for predicate
                    ".reg .pred m56;\n\t"                                // Register for predicate

                    "ld.s32 m1, [%0+0];\n\t"                             // Load nid
                    "ld.s32 m2, [%0+4];\n\t"                             // Load nid
                    "ld.s32 m3, [%0+8];\n\t"                             // Load nid
                    "ld.s32 m4, [%0+12];\n\t"                            // Load nid
                    "ld.s32 m5, [%0+16];\n\t"                            // Load nid
                    "ld.s32 m6, [%0+20];\n\t"                            // Load nid
                    "ld.s32 m7, [%0+24];\n\t"                            // Load nid
                    "ld.s32 m8, [%0+28];\n\t"                            // Load nid

                    "ld.s32 m33, [%1+0];\n\t"                            // Load nid
                    "ld.s32 m34, [%1+4];\n\t"                            // Load nid
                    "ld.s32 m35, [%1+8];\n\t"                            // Load nid
                    "ld.s32 m36, [%1+12];\n\t"                           // Load nid
                    "ld.s32 m37, [%1+16];\n\t"                           // Load nid
                    "ld.s32 m38, [%1+20];\n\t"                           // Load nid
                    "ld.s32 m39, [%1+24];\n\t"                           // Load nid
                    "ld.s32 m40, [%1+28];\n\t"                           // Load nid

                    "mul.wide.s32 m9, m1, 4;\n\t"                        // Multiply nid for x address calculation
                    "mul.wide.s32 m10, m2, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 m11, m3, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 m12, m4, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 m13, m5, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 m14, m6, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 m15, m7, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 m16, m8, 4;\n\t"                       // Multiply nid for x address calculation

                    "add.u64 m17, m101, m9;\n\t"                         // Final address calculation for color
                    "add.u64 m18, m101, m10;\n\t"                        // Final address calculation for color
                    "add.u64 m19, m101, m11;\n\t"                        // Final address calculation for color
                    "add.u64 m20, m101, m12;\n\t"                        // Final address calculation for color
                    "add.u64 m21, m101, m13;\n\t"                        // Final address calculation for color
                    "add.u64 m22, m101, m14;\n\t"                        // Final address calculation for color
                    "add.u64 m23, m101, m15;\n\t"                        // Final address calculation for color
                    "add.u64 m24, m101, m16;\n\t"                        // Final address calculation for color

                    "add.u64 m25, m102, m9;\n\t"                         // Final address calculation for max_d
                    "add.u64 m26, m102, m10;\n\t"                        // Final address calculation for max_d
                    "add.u64 m27, m102, m11;\n\t"                        // Final address calculation for max_d
                    "add.u64 m28, m102, m12;\n\t"                        // Final address calculation for max_d
                    "add.u64 m29, m102, m13;\n\t"                        // Final address calculation for max_d
                    "add.u64 m30, m102, m14;\n\t"                        // Final address calculation for max_d
                    "add.u64 m31, m102, m15;\n\t"                        // Final address calculation for max_d
                    "add.u64 m32, m102, m16;\n\t"                        // Final address calculation for max_d

                    "ld.s32 m69, [m17];\n\t"                             // Load color
                    "ld.s32 m70, [m18];\n\t"                             // Load color
                    "ld.s32 m71, [m19];\n\t"                             // Load color
                    "ld.s32 m72, [m20];\n\t"                             // Load color
                    "ld.s32 m73, [m21];\n\t"                             // Load color
                    "ld.s32 m74, [m22];\n\t"                             // Load color
                    "ld.s32 m75, [m23];\n\t"                             // Load color
                    "ld.s32 m76, [m24];\n\t"                             // Load color

                    "setp.eq.s32 m41, m69, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 m42, m70, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 m43, m71, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 m44, m72, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 m45, m73, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 m46, m74, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 m47, m75, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 m48, m76, -1;\n\t"                      // Do predicate

                    "setp.gt.s32 m49, m33, 1;\n\t"                       // Do predicate
                    "setp.gt.s32 m50, m34, 1;\n\t"                       // Do predicate
                    "setp.gt.s32 m51, m35, 1;\n\t"                       // Do predicate
                    "setp.gt.s32 m52, m36, 1;\n\t"                       // Do predicate
                    "setp.gt.s32 m53, m37, 1;\n\t"                       // Do predicate
                    "setp.gt.s32 m54, m38, 1;\n\t"                       // Do predicate
                    "setp.gt.s32 m55, m39, 1;\n\t"                       // Do predicate
                    "setp.gt.s32 m56, m40, 1;\n\t"                       // Do predicate

                    "COLOR1_PUSH_PRED_BODY_8_1:\n\t"
                    "@!m41 bra COLOR1_PUSH_PRED_BODY_8_2;\n\t"           // Check if node is processed
                    "@!m49 bra COLOR1_PUSH_PRED_BODY_8_2;\n\t"           // Check if node is well connected
                    "atom.max.s32 m79, [m25], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_PRED_BODY_8_2:\n\t"
                    "@!m42 bra COLOR1_PUSH_PRED_BODY_8_3;\n\t"           // Check if node is processed
                    "@!m50 bra COLOR1_PUSH_PRED_BODY_8_3;\n\t"           // Check if node is well connected
                    "atom.max.s32 m80, [m26], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_PRED_BODY_8_3:\n\t"
                    "@!m43 bra COLOR1_PUSH_PRED_BODY_8_4;\n\t"           // Check if node is processed
                    "@!m51 bra COLOR1_PUSH_PRED_BODY_8_4;\n\t"           // Check if node is well connected
                    "atom.max.s32 m81, [m27], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_PRED_BODY_8_4:\n\t"
                    "@!m44 bra COLOR1_PUSH_PRED_BODY_8_5;\n\t"           // Check if node is processed
                    "@!m52 bra COLOR1_PUSH_PRED_BODY_8_5;\n\t"           // Check if node is well connected
                    "atom.max.s32 m82, [m28], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_PRED_BODY_8_5:\n\t"
                    "@!m45 bra COLOR1_PUSH_PRED_BODY_8_6;\n\t"           // Check if node is processed
                    "@!m53 bra COLOR1_PUSH_PRED_BODY_8_6;\n\t"           // Check if node is well connected
                    "atom.max.s32 m83, [m29], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_PRED_BODY_8_6:\n\t"
                    "@!m46 bra COLOR1_PUSH_PRED_BODY_8_7;\n\t"           // Check if node is processed
                    "@!m54 bra COLOR1_PUSH_PRED_BODY_8_7;\n\t"           // Check if node is well connected
                    "atom.max.s32 m84, [m30], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_PRED_BODY_8_7:\n\t"
                    "@!m47 bra COLOR1_PUSH_PRED_BODY_8_8;\n\t"           // Check if node is processed
                    "@!m55 bra COLOR1_PUSH_PRED_BODY_8_8;\n\t"           // Check if node is well connected
                    "atom.max.s32 m85, [m31], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_PRED_BODY_8_8:\n\t"
                    "@!m48 bra COLOR1_PUSH_NEIGH_END_8;\n\t"             // Check if node is processed
                    "@!m56 bra COLOR1_PUSH_NEIGH_END_8;\n\t"             // Check if node is well connected
                    "atom.max.s32 m86, [m32], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_NEIGH_END_8:\n\t"

                    :                                                    // Outputs
                    : "l"(col_base_addr), "l"(col_cnt_base_addr)         // Inputs
                );
            }

            for (; edge <= (row_end - 4); edge += 4) {
                int * const col_base_addr = &col[edge];
                int * const col_cnt_base_addr = &col_cnt[edge];

                asm volatile
                (
                    ".reg .s32 q1;\n\t"                                  // Register for nid loaded from col
                    ".reg .s32 q2;\n\t"                                  // Register for nid loaded from col
                    ".reg .s32 q3;\n\t"                                  // Register for nid loaded from col
                    ".reg .s32 q4;\n\t"                                  // Register for nid loaded from col

                    ".reg .s32 q33;\n\t"                                 // Register for out_deg loaded from col_cnt
                    ".reg .s32 q34;\n\t"                                 // Register for out_deg loaded from col_cnt
                    ".reg .s32 q35;\n\t"                                 // Register for out_deg loaded from col_cnt
                    ".reg .s32 q36;\n\t"                                 // Register for out_deg loaded from col_cnt

                    ".reg .u64 q9;\n\t"                                  // Register for multiplied nid value as address
                    ".reg .u64 q10;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 q11;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 q12;\n\t"                                 // Register for multiplied nid value as address

                    ".reg .u64 q17;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 q18;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 q19;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 q20;\n\t"                                 // Register for final address to load from color

                    ".reg .u64 q25;\n\t"                                 // Register for final address to load from max_d
                    ".reg .u64 q26;\n\t"                                 // Register for final address to load from max_d
                    ".reg .u64 q27;\n\t"                                 // Register for final address to load from max_d
                    ".reg .u64 q28;\n\t"                                 // Register for final address to load from max_d

                    ".reg .s32 q69;\n\t"                                 // Register for color
                    ".reg .s32 q70;\n\t"                                 // Register for color
                    ".reg .s32 q71;\n\t"                                 // Register for color
                    ".reg .s32 q72;\n\t"                                 // Register for color

                    ".reg .s32 q79;\n\t"                                 // Register for max_d
                    ".reg .s32 q80;\n\t"                                 // Register for max_d
                    ".reg .s32 q81;\n\t"                                 // Register for max_d
                    ".reg .s32 q82;\n\t"                                 // Register for max_d

                    ".reg .pred q41;\n\t"                                // Register for predicate
                    ".reg .pred q42;\n\t"                                // Register for predicate
                    ".reg .pred q43;\n\t"                                // Register for predicate
                    ".reg .pred q44;\n\t"                                // Register for predicate

                    ".reg .pred q49;\n\t"                                // Register for predicate
                    ".reg .pred q50;\n\t"                                // Register for predicate
                    ".reg .pred q51;\n\t"                                // Register for predicate
                    ".reg .pred q52;\n\t"                                // Register for predicate

                    "ld.s32 q1, [%0+0];\n\t"                             // Load nid
                    "ld.s32 q2, [%0+4];\n\t"                             // Load nid
                    "ld.s32 q3, [%0+8];\n\t"                             // Load nid
                    "ld.s32 q4, [%0+12];\n\t"                            // Load nid

                    "ld.s32 q33, [%1+0];\n\t"                            // Load nid
                    "ld.s32 q34, [%1+4];\n\t"                            // Load nid
                    "ld.s32 q35, [%1+8];\n\t"                            // Load nid
                    "ld.s32 q36, [%1+12];\n\t"                           // Load nid

                    "mul.wide.s32 q9, q1, 4;\n\t"                        // Multiply nid for x address calculation
                    "mul.wide.s32 q10, q2, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 q11, q3, 4;\n\t"                       // Multiply nid for x address calculation
                    "mul.wide.s32 q12, q4, 4;\n\t"                       // Multiply nid for x address calculation

                    "add.u64 q17, m101, q9;\n\t"                         // Final address calculation for color
                    "add.u64 q18, m101, q10;\n\t"                        // Final address calculation for color
                    "add.u64 q19, m101, q11;\n\t"                        // Final address calculation for color
                    "add.u64 q20, m101, q12;\n\t"                        // Final address calculation for color

                    "add.u64 q25, m102, q9;\n\t"                         // Final address calculation for max_d
                    "add.u64 q26, m102, q10;\n\t"                        // Final address calculation for max_d
                    "add.u64 q27, m102, q11;\n\t"                        // Final address calculation for max_d
                    "add.u64 q28, m102, q12;\n\t"                        // Final address calculation for max_d

                    "ld.s32 q69, [q17];\n\t"                             // Load color
                    "ld.s32 q70, [q18];\n\t"                             // Load color
                    "ld.s32 q71, [q19];\n\t"                             // Load color
                    "ld.s32 q72, [q20];\n\t"                             // Load color

                    "setp.eq.s32 q41, q69, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 q42, q70, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 q43, q71, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 q44, q72, -1;\n\t"                      // Do predicate

                    "setp.gt.s32 q49, q33, 1;\n\t"                       // Do predicate
                    "setp.gt.s32 q50, q34, 1;\n\t"                       // Do predicate
                    "setp.gt.s32 q51, q35, 1;\n\t"                       // Do predicate
                    "setp.gt.s32 q52, q36, 1;\n\t"                       // Do predicate

                    "COLOR1_PUSH_PRED_BODY_4_1:\n\t"
                    "@!q41 bra COLOR1_PUSH_PRED_BODY_4_2;\n\t"           // Check if node is processed
                    "@!q49 bra COLOR1_PUSH_PRED_BODY_4_2;\n\t"           // Check if node is well connected
                    "atom.max.s32 q79, [q25], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_PRED_BODY_4_2:\n\t"
                    "@!q42 bra COLOR1_PUSH_PRED_BODY_4_3;\n\t"           // Check if node is processed
                    "@!q50 bra COLOR1_PUSH_PRED_BODY_4_3;\n\t"           // Check if node is well connected
                    "atom.max.s32 q80, [q26], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_PRED_BODY_4_3:\n\t"
                    "@!q43 bra COLOR1_PUSH_PRED_BODY_4_4;\n\t"           // Check if node is processed
                    "@!q51 bra COLOR1_PUSH_PRED_BODY_4_4;\n\t"           // Check if node is well connected
                    "atom.max.s32 q81, [q27], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_PRED_BODY_4_4:\n\t"
                    "@!q44 bra COLOR1_PUSH_NEIGH_END_4;\n\t"             // Check if node is processed
                    "@!q52 bra COLOR1_PUSH_NEIGH_END_4;\n\t"             // Check if node is well connected
                    "atom.max.s32 q82, [q28], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_NEIGH_END_4:\n\t"

                    :                                                    // Outputs
                    : "l"(col_base_addr), "l"(col_cnt_base_addr)         // Inputs
                );
            }

            for (; edge <= (row_end - 2); edge += 2) {
                int * const col_base_addr = &col[edge];
                int * const col_cnt_base_addr = &col_cnt[edge];

                asm volatile
                (
                    ".reg .s32 t1;\n\t"                                  // Register for nid loaded from col
                    ".reg .s32 t2;\n\t"                                  // Register for nid loaded from col

                    ".reg .s32 t33;\n\t"                                 // Register for out_deg loaded from col_cnt
                    ".reg .s32 t34;\n\t"                                 // Register for out_deg loaded from col_cnt

                    ".reg .u64 t9;\n\t"                                  // Register for multiplied nid value as address
                    ".reg .u64 t10;\n\t"                                 // Register for multiplied nid value as address

                    ".reg .u64 t17;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 t18;\n\t"                                 // Register for final address to load from color

                    ".reg .u64 t25;\n\t"                                 // Register for final address to load from max_d
                    ".reg .u64 t26;\n\t"                                 // Register for final address to load from max_d

                    ".reg .s32 t69;\n\t"                                 // Register for color
                    ".reg .s32 t70;\n\t"                                 // Register for color

                    ".reg .s32 t79;\n\t"                                 // Register for max_d
                    ".reg .s32 t80;\n\t"                                 // Register for max_d

                    ".reg .pred t41;\n\t"                                // Register for predicate
                    ".reg .pred t42;\n\t"                                // Register for predicate

                    ".reg .pred t49;\n\t"                                // Register for predicate
                    ".reg .pred t50;\n\t"                                // Register for predicate

                    "ld.s32 t1, [%0+0];\n\t"                             // Load nid
                    "ld.s32 t2, [%0+4];\n\t"                             // Load nid

                    "ld.s32 t33, [%1+0];\n\t"                            // Load nid
                    "ld.s32 t34, [%1+4];\n\t"                            // Load nid

                    "mul.wide.s32 t9, t1, 4;\n\t"                        // Multiply nid for x address calculation
                    "mul.wide.s32 t10, t2, 4;\n\t"                       // Multiply nid for x address calculation

                    "add.u64 t17, m101, t9;\n\t"                         // Final address calculation for color
                    "add.u64 t18, m101, t10;\n\t"                        // Final address calculation for color

                    "add.u64 t25, m102, t9;\n\t"                         // Final address calculation for max_d
                    "add.u64 t26, m102, t10;\n\t"                        // Final address calculation for max_d

                    "ld.s32 t69, [t17];\n\t"                             // Load color
                    "ld.s32 t70, [t18];\n\t"                             // Load color

                    "setp.eq.s32 t41, t69, -1;\n\t"                      // Do predicate
                    "setp.eq.s32 t42, t70, -1;\n\t"                      // Do predicate

                    "setp.gt.s32 t49, t33, 1;\n\t"                       // Do predicate
                    "setp.gt.s32 t50, t34, 1;\n\t"                       // Do predicate

                    "COLOR1_PUSH_PRED_BODY_2_1:\n\t"
                    "@!t41 bra COLOR1_PUSH_PRED_BODY_2_2;\n\t"           // Check if node is processed
                    "@!t49 bra COLOR1_PUSH_PRED_BODY_2_2;\n\t"           // Check if node is well connected
                    "atom.max.s32 t79, [t25], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_PRED_BODY_2_2:\n\t"
                    "@!t42 bra COLOR1_PUSH_NEIGH_END_2;\n\t"             // Check if node is processed
                    "@!t50 bra COLOR1_PUSH_NEIGH_END_2;\n\t"             // Check if node is well connected
                    "atom.max.s32 t80, [t26], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_NEIGH_END_2:\n\t"

                    :                                                    // Outputs
                    : "l"(col_base_addr), "l"(col_cnt_base_addr)         // Inputs
                );
            }

            for (; edge < row_end; edge++) {
                int * const col_base_addr = &col[edge];
                int * const col_cnt_base_addr = &col_cnt[edge];

                asm volatile
                (
                    ".reg .s32 a1;\n\t"                                  // Register for nid loaded from col
                    ".reg .s32 a33;\n\t"                                 // Register for out_deg loaded from col_cnt
                    ".reg .u64 a9;\n\t"                                  // Register for multiplied nid value as address
                    ".reg .u64 a17;\n\t"                                 // Register for final address to load from color
                    ".reg .u64 a25;\n\t"                                 // Register for final address to load from max_d
                    ".reg .s32 a69;\n\t"                                 // Register for color
                    ".reg .s32 a79;\n\t"                                 // Register for max_d
                    ".reg .pred a41;\n\t"                                // Register for predicate
                    ".reg .pred a49;\n\t"                                // Register for predicate

                    "ld.s32 a1, [%0+0];\n\t"                             // Load nid
                    "ld.s32 a33, [%1+0];\n\t"                            // Load nid
                    "mul.wide.s32 a9, a1, 4;\n\t"                        // Multiply nid for x address calculation
                    "add.u64 a17, m101, a9;\n\t"                         // Final address calculation for color
                    "add.u64 a25, m102, a9;\n\t"                         // Final address calculation for max_d
                    "ld.s32 a69, [a17];\n\t"                             // Load color
                    "setp.eq.s32 a41, a69, -1;\n\t"                      // Do predicate
                    "setp.gt.s32 a49, a33, 1;\n\t"                       // Do predicate

                    "COLOR1_PUSH_PRED_BODY_1_1:\n\t"
                    "@!a41 bra COLOR1_PUSH_NEIGH_END_1;\n\t"             // Check if node is processed
                    "@!a49 bra COLOR1_PUSH_NEIGH_END_1;\n\t"             // Check if node is well connected
                    "atom.max.s32 a79, [a25], m99;\n\t"                  // Find max
                    "mov.s32 m100, 1;\n\t"                               // Set local_cont

                    "COLOR1_PUSH_NEIGH_END_1:\n\t"

                    :                                                    // Outputs
                    : "l"(col_base_addr), "l"(col_cnt_base_addr)         // Inputs
                );
            }

            asm volatile
            (
              "mov.s32 %0, m100;\n\t"                          // Store local_cont

              : "=r"(cont_tid)                                 // No outputs
              :                                                // Inputs
            );
        }
    }

    cont[blockIdx.x * blockDim.x + threadIdx.x] = cont_tid;
/*
    if (threadIdx.x == 0) {
        __denovo_gpuEpilogue(SPECIAL_REGION);
        __denovo_gpuEpilogue(READ_ONLY_REGION);
        __denovo_gpuEpilogue(default_reg);
        __denovo_gpuEpilogue(max_reg);
    }
    */
}


// // push version of color1- use atomicMax to update every neighbor for assigned node
// // with max of current value and this nodes value
// __global__ void color1_push(int *row, int *col, int *node_value,
//                             int *col_cnt, int *color_array,
//                             int *cont, int *max_d, const int color,
//                             const int num_nodes, const int num_edges,
//                             region_t default_reg, region_t max_reg)
// {
//     // Get my thread workitem id
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int cont_tid = false;
// 
//     if (threadIdx.x == 0) {
//         __denovo_setAcquireRegion(SPECIAL_REGION);
//         __denovo_addAcquireRegion(READ_ONLY_REGION);
//         __denovo_addAcquireRegion(default_reg);
//         __denovo_addAcquireRegion(max_reg);
//     }
//     __syncthreads();
// 
//     for (; tid < num_nodes; tid += blockDim.x * gridDim.x) {
//         // If the vertex is still not colored
//         if (color_array[tid] == -1) {
//             // Get the start and end pointer of the neighbor list
//             const int start = row[tid];
//             const int end = row[tid + 1];
// 
//             const int this_node_val = node_value[tid];
// 
//             // Navigate the neighbor list, update max with this value
//             for (int edge = start; edge < end; edge++) {
//                 const int nid = col[edge];
//                 const int neigh_out_deg = col_cnt[edge];
// 
//                 // Determine if the vertex value is the maximum in the neighborhood
//                 if (color_array[nid] == -1 && neigh_out_deg > 1) {
//                     cont_tid = true;
//                     atomicMax(&max_d[nid], this_node_val);
//                 }
//             }
//         }
//     }
// 
//     cont[blockIdx.x * blockDim.x + threadIdx.x] = cont_tid;
// 
//     if (threadIdx.x == 0) {
//         __denovo_gpuEpilogue(SPECIAL_REGION);
//         __denovo_gpuEpilogue(READ_ONLY_REGION);
//         __denovo_gpuEpilogue(default_reg);
//         __denovo_gpuEpilogue(max_reg);
//     }
// }


// only difference here is we need to reset max_d to -1 if this node is still unset
__global__ void color2_push(int *node_value, int *color_array, int *max_d,
                            const int color, const int num_nodes,
                            const int num_edges)
{
    // Get my workitem id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
/*
    if (threadIdx.x == 0) {
        __denovo_setAcquireRegion(SPECIAL_REGION);
        __denovo_addAcquireRegion(READ_ONLY_REGION);
        __denovo_addAcquireRegion(default_reg);
        __denovo_addAcquireRegion(max_reg);
    }
    __syncthreads();
*/
    for (; tid < num_nodes; tid += blockDim.x * gridDim.x) {
        // If the vertex is still not colored
        if (color_array[tid] == -1) {
            const int max_tid = atomicAdd(&max_d[tid], 0);

            if (node_value[tid] >= max_tid) {
                // Assign a color
                color_array[tid] = color;
            } else {
                atomicOr(&max_d[tid], -1);
            }
        }
    }
/*
    if (threadIdx.x == 0) {
        __denovo_gpuEpilogue(SPECIAL_REGION);
        __denovo_gpuEpilogue(READ_ONLY_REGION);
        __denovo_gpuEpilogue(default_reg);
        __denovo_gpuEpilogue(max_reg);
    }
    */
}
