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

//#define BIGNUM 99999999
#define EPSILON 0.0000001

#define NOT_PROCESSED -1
#define INACTIVE      -2
#define INDEPENDENT   2
/**
* init kernel
* @param s_array   set array
* @param c_array   status array
* @param cu_array  status update array
* @param num_nodes number of vertices
* @param num_edges number of edges
*/
__global__ void
init(int *s_array, int *c_array, int *cu_array, int num_nodes, int num_edges)
{
    // Get my workitem id
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_nodes) {
        // Set the status array: not processed
        c_array[tid] = -1;
        cu_array[tid] = -1;
        s_array[tid] = 0;
    }
}


/**
* mis1 kernel
* @param row          csr pointer array
* @param col          csr column index array
* @param node_value   node value array
* @param s_array      set array
* @param c_array node status array
* @param min_array    node value array
* @param cont node    value array
* @param num_nodes    number of vertices
* @param num_edges    number of edges
*/
__global__ void
mis1(int *row, int *col, int *node_value, int *s_array, int *c_array,
     int *min_array, bool *cont, int num_gpu_nodes, int num_edges
)
{
    const int tx = threadIdx.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int my_task = tid; 
    int edge = 0;
    bool local_cont = false;
/*
    asm volatile
    (
        ".reg .s32 mINDEPENDENT;\n\t"                          // Register for s_array addr
        ".reg .s32 mNOTPROCESSED;\n\t"                         // Register for s_array addr
        ".reg .s32 mINACTIVE;\n\t"                             // Register for s_array addr

        "mov.s32 mINDEPENDENT, %0;\n\t"
        "mov.s32 mNOTPROCESSED, %1;\n\t"
        "mov.s32 mINACTIVE, %2;\n\t"

        :                                                      // Outputs
        : "r"(INDEPENDENT), "r"(NOT_PROCESSED), "r"(INACTIVE)  // Inputs
    );
*/
    /*
    if (tx == 0) {
        __denovo_setAcquireRegion(SPECIAL_REGION);
        __denovo_addAcquireRegion(READ_ONLY_REGION);
        __denovo_addAcquireRegion(default_reg);
        __denovo_addAcquireRegion(rel_reg);
    }
    __syncthreads();
*/
    for (; my_task < num_gpu_nodes; my_task += blockDim.x * gridDim.x) {
        // If the vertex is not processed
        if (c_array[my_task] == NOT_PROCESSED) {
            local_cont = true;

            // Get the start and end pointers
            const int row_start = row[my_task];
            const int row_end = row[my_task + 1];

            const int my_node_value = node_value[my_task];
/*
            asm volatile
            (
                ".reg .u64 m99;\n\t"                   // Register for c_array addr
                ".reg .u64 m100;\n\t"                  // Register for node_value addr
                ".reg .s32 m101;\n\t"                  // Register for my_node_value

                "mov.u64 m99, %0;\n\t"
                "mov.u64 m100, %1;\n\t"
                "mov.s32 m101, %2;\n\t"

                :                                                  // Outputs
                : "l"(c_array), "l"(min_array), "r"(my_node_value) // Inputs
            );
*/
            // Navigate the neighbor list and find the min
            for (int edge = row_start; edge < row_end; edge++) {
                const int neighbor = col[edge];

              if (c_array[neighbor] == NOT_PROCESSED) {
                    atomicMin(&min_array[neighbor], my_node_value);
                }
            }
/*
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

                    ".reg .u64 m17;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m18;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m19;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m20;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m21;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m22;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m23;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m24;\n\t"                                 // Register for multiplied nid value as address

                    ".reg .u64 m25;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 m26;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 m27;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 m28;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 m29;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 m30;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 m31;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 m32;\n\t"                                 // Register for final address to load from c

                    ".reg .s32 m33;\n\t"                                 // Register for c
                    ".reg .s32 m34;\n\t"                                 // Register for c
                    ".reg .s32 m35;\n\t"                                 // Register for c
                    ".reg .s32 m36;\n\t"                                 // Register for c
                    ".reg .s32 m37;\n\t"                                 // Register for c
                    ".reg .s32 m38;\n\t"                                 // Register for c
                    ".reg .s32 m39;\n\t"                                 // Register for c
                    ".reg .s32 m40;\n\t"                                 // Register for c

                    ".reg .u64 m65;\n\t"                                 // Register for final address to load from min_value
                    ".reg .u64 m66;\n\t"                                 // Register for final address to load from min_value
                    ".reg .u64 m67;\n\t"                                 // Register for final address to load from min_value
                    ".reg .u64 m68;\n\t"                                 // Register for final address to load from min_value
                    ".reg .u64 m69;\n\t"                                 // Register for final address to load from min_value
                    ".reg .u64 m70;\n\t"                                 // Register for final address to load from min_value
                    ".reg .u64 m71;\n\t"                                 // Register for final address to load from min_value
                    ".reg .u64 m72;\n\t"                                 // Register for final address to load from min_value

                    ".reg .s32 m73;\n\t"                                 // Register for min_value
                    ".reg .s32 m74;\n\t"                                 // Register for min_value
                    ".reg .s32 m75;\n\t"                                 // Register for min_value
                    ".reg .s32 m76;\n\t"                                 // Register for min_value
                    ".reg .s32 m77;\n\t"                                 // Register for min_value
                    ".reg .s32 m78;\n\t"                                 // Register for min_value
                    ".reg .s32 m79;\n\t"                                 // Register for min_value
                    ".reg .s32 m80;\n\t"                                 // Register for min_value

                    ".reg .pred m49;\n\t"                                // Register for c predicate
                    ".reg .pred m50;\n\t"                                // Register for c predicate
                    ".reg .pred m51;\n\t"                                // Register for c predicate
                    ".reg .pred m52;\n\t"                                // Register for c predicate
                    ".reg .pred m53;\n\t"                                // Register for c predicate
                    ".reg .pred m54;\n\t"                                // Register for c predicate
                    ".reg .pred m55;\n\t"                                // Register for c predicate
                    ".reg .pred m56;\n\t"                                // Register for c predicate

                    "ld.s32 m1, [%0+0];\n\t"                             // Load nid
                    "ld.s32 m2, [%0+4];\n\t"                             // Load nid
                    "ld.s32 m3, [%0+8];\n\t"                             // Load nid
                    "ld.s32 m4, [%0+12];\n\t"                            // Load nid
                    "ld.s32 m5, [%0+16];\n\t"                            // Load nid
                    "ld.s32 m6, [%0+20];\n\t"                            // Load nid
                    "ld.s32 m7, [%0+24];\n\t"                            // Load nid
                    "ld.s32 m8, [%0+28];\n\t"                            // Load nid

                    "mul.wide.s32 m17, m1, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 m18, m2, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 m19, m3, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 m20, m4, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 m21, m5, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 m22, m6, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 m23, m7, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 m24, m8, 4;\n\t"                       // Multiply nid for address calculation

                    "add.u64 m25, m99, m17;\n\t"                         // Final address calculation for c
                    "add.u64 m26, m99, m18;\n\t"                         // Final address calculation for c
                    "add.u64 m27, m99, m19;\n\t"                         // Final address calculation for c
                    "add.u64 m28, m99, m20;\n\t"                         // Final address calculation for c
                    "add.u64 m29, m99, m21;\n\t"                         // Final address calculation for c
                    "add.u64 m30, m99, m22;\n\t"                         // Final address calculation for c
                    "add.u64 m31, m99, m23;\n\t"                         // Final address calculation for c
                    "add.u64 m32, m99, m24;\n\t"                         // Final address calculation for c

                    "add.u64 m65, m100, m17;\n\t"                        // Final address calculation for min_value
                    "add.u64 m66, m100, m18;\n\t"                        // Final address calculation for min_value
                    "add.u64 m67, m100, m19;\n\t"                        // Final address calculation for min_value
                    "add.u64 m68, m100, m20;\n\t"                        // Final address calculation for min_value
                    "add.u64 m69, m100, m21;\n\t"                        // Final address calculation for min_value
                    "add.u64 m70, m100, m22;\n\t"                        // Final address calculation for min_value
                    "add.u64 m71, m100, m23;\n\t"                        // Final address calculation for min_value
                    "add.u64 m72, m100, m24;\n\t"                        // Final address calculation for min_value

                    "ld.s32 m33, [m25];\n\t"                             // Load c
                    "ld.s32 m34, [m26];\n\t"                             // Load c
                    "ld.s32 m35, [m27];\n\t"                             // Load c
                    "ld.s32 m36, [m28];\n\t"                             // Load c
                    "ld.s32 m37, [m29];\n\t"                             // Load c
                    "ld.s32 m38, [m30];\n\t"                             // Load c
                    "ld.s32 m39, [m31];\n\t"                             // Load c
                    "ld.s32 m40, [m32];\n\t"                             // Load c

                    "setp.eq.s32 m49, m33, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 m50, m34, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 m51, m35, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 m52, m36, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 m53, m37, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 m54, m38, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 m55, m39, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 m56, m40, mNOTPROCESSED;\n\t"           // Value for predicate

                    "MIS1_PRED_BODY_8_1:\n\t"                            // Predicate body
                    "@!m49 bra MIS1_PRED_BODY_8_2;\n\t"                  // Predicate on value of c
                    "atom.min.s32 m73, [m65], m101;\n\t"                 // Do min of node_value

                    "MIS1_PRED_BODY_8_2:\n\t"                            // Predicate body
                    "@!m50 bra MIS1_PRED_BODY_8_3;\n\t"                  // Predicate on value of c
                    "atom.min.s32 m74, [m66], m101;\n\t"                 // Do min of node_value

                    "MIS1_PRED_BODY_8_3:\n\t"                            // Predicate body
                    "@!m51 bra MIS1_PRED_BODY_8_4;\n\t"                  // Predicate on value of c
                    "atom.min.s32 m75, [m67], m101;\n\t"                 // Do min of node_value

                    "MIS1_PRED_BODY_8_4:\n\t"                            // Predicate body
                    "@!m52 bra MIS1_PRED_BODY_8_5;\n\t"                  // Predicate on value of c
                    "atom.min.s32 m76, [m68], m101;\n\t"                 // Do min of node_value

                    "MIS1_PRED_BODY_8_5:\n\t"                            // Predicate body
                    "@!m53 bra MIS1_PRED_BODY_8_6;\n\t"                  // Predicate on value of c
                    "atom.min.s32 m77, [m69], m101;\n\t"                 // Do min of node_value

                    "MIS1_PRED_BODY_8_6:\n\t"                            // Predicate body
                    "@!m54 bra MIS1_PRED_BODY_8_7;\n\t"                  // Predicate on value of c
                    "atom.min.s32 m78, [m70], m101;\n\t"                 // Do min of node_value

                    "MIS1_PRED_BODY_8_7:\n\t"                            // Predicate body
                    "@!m55 bra MIS1_PRED_BODY_8_8;\n\t"                  // Predicate on value of c
                    "atom.min.s32 m79, [m71], m101;\n\t"                 // Do min of node_value

                    "MIS1_PRED_BODY_8_8:\n\t"                            // Predicate body
                    "@!m56 bra MIS1_NEIGH_LOOP_8;\n\t"                   // Predicate on value of c
                    "atom.min.s32 m80, [m72], m101;\n\t"                 // Do min of node_value

                    "MIS1_NEIGH_LOOP_8:\n\t"

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

                    ".reg .u64 q17;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 q18;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 q19;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 q20;\n\t"                                 // Register for multiplied nid value as address

                    ".reg .u64 q25;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 q26;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 q27;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 q28;\n\t"                                 // Register for final address to load from c

                    ".reg .s32 q33;\n\t"                                 // Register for c
                    ".reg .s32 q34;\n\t"                                 // Register for c
                    ".reg .s32 q35;\n\t"                                 // Register for c
                    ".reg .s32 q36;\n\t"                                 // Register for c

                    ".reg .u64 q65;\n\t"                                 // Register for final address to load from min_value
                    ".reg .u64 q66;\n\t"                                 // Register for final address to load from min_value
                    ".reg .u64 q67;\n\t"                                 // Register for final address to load from min_value
                    ".reg .u64 q68;\n\t"                                 // Register for final address to load from min_value

                    ".reg .s32 q73;\n\t"                                 // Register for node_value
                    ".reg .s32 q74;\n\t"                                 // Register for node_value
                    ".reg .s32 q75;\n\t"                                 // Register for node_value
                    ".reg .s32 q76;\n\t"                                 // Register for node_value

                    ".reg .pred q49;\n\t"                                // Register for c predicate
                    ".reg .pred q50;\n\t"                                // Register for c predicate
                    ".reg .pred q51;\n\t"                                // Register for c predicate
                    ".reg .pred q52;\n\t"                                // Register for c predicate

                    "ld.s32 q1, [%0+0];\n\t"                             // Load nid
                    "ld.s32 q2, [%0+4];\n\t"                             // Load nid
                    "ld.s32 q3, [%0+8];\n\t"                             // Load nid
                    "ld.s32 q4, [%0+12];\n\t"                            // Load nid

                    "mul.wide.s32 q17, q1, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 q18, q2, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 q19, q3, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 q20, q4, 4;\n\t"                       // Multiply nid for address calculation

                    "add.u64 q25, m99, q17;\n\t"                         // Final address calculation for c
                    "add.u64 q26, m99, q18;\n\t"                         // Final address calculation for c
                    "add.u64 q27, m99, q19;\n\t"                         // Final address calculation for c
                    "add.u64 q28, m99, q20;\n\t"                         // Final address calculation for c

                    "add.u64 q65, m100, q17;\n\t"                        // Final address calculation for node_value
                    "add.u64 q66, m100, q18;\n\t"                        // Final address calculation for node_value
                    "add.u64 q67, m100, q19;\n\t"                        // Final address calculation for node_value
                    "add.u64 q68, m100, q20;\n\t"                        // Final address calculation for node_value

                    "ld.s32 q33, [q25];\n\t"                             // Load c
                    "ld.s32 q34, [q26];\n\t"                             // Load c
                    "ld.s32 q35, [q27];\n\t"                             // Load c
                    "ld.s32 q36, [q28];\n\t"                             // Load c

                    "setp.eq.s32 q49, q33, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 q50, q34, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 q51, q35, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 q52, q36, mNOTPROCESSED;\n\t"           // Value for predicate

                    "MIS1_PRED_BODY_4_1:\n\t"                            // Predicate body
                    "@!q49 bra MIS1_PRED_BODY_4_2;\n\t"                  // Predicate on value of c
                    "atom.min.s32 q73, [q65], m101;\n\t"                 // Do min of node_value

                    "MIS1_PRED_BODY_4_2:\n\t"                            // Predicate body
                    "@!q50 bra MIS1_PRED_BODY_4_3;\n\t"                  // Predicate on value of c
                    "atom.min.s32 q74, [q66], m101;\n\t"                 // Do min of node_value

                    "MIS1_PRED_BODY_4_3:\n\t"                            // Predicate body
                    "@!q51 bra MIS1_PRED_BODY_4_4;\n\t"                  // Predicate on value of c
                    "atom.min.s32 q75, [q67], m101;\n\t"                 // Do min of node_value

                    "MIS1_PRED_BODY_4_4:\n\t"                            // Predicate body
                    "@!q52 bra MIS1_NEIGH_LOOP_4;\n\t"                   // Predicate on value of c
                    "atom.min.s32 q76, [q68], m101;\n\t"                 // Do min of node_value

                    "MIS1_NEIGH_LOOP_4:\n\t"

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

                    ".reg .u64 t17;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 t18;\n\t"                                 // Register for multiplied nid value as address

                    ".reg .u64 t25;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 t26;\n\t"                                 // Register for final address to load from c

                    ".reg .s32 t33;\n\t"                                 // Register for c
                    ".reg .s32 t34;\n\t"                                 // Register for c

                    ".reg .u64 t65;\n\t"                                 // Register for final address to load from node_value
                    ".reg .u64 t66;\n\t"                                 // Register for final address to load from node_value

                    ".reg .s32 t73;\n\t"                                 // Register for node_value
                    ".reg .s32 t74;\n\t"                                 // Register for node_value

                    ".reg .pred t49;\n\t"                                // Register for c predicate
                    ".reg .pred t50;\n\t"                                // Register for c predicate

                    "ld.s32 t1, [%0+0];\n\t"                             // Load nid
                    "ld.s32 t2, [%0+4];\n\t"                             // Load nid

                    "mul.wide.s32 t17, t1, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 t18, t2, 4;\n\t"                       // Multiply nid for address calculation

                    "add.u64 t25, m99, t17;\n\t"                         // Final address calculation for c
                    "add.u64 t26, m99, t18;\n\t"                         // Final address calculation for c

                    "add.u64 t65, m100, t17;\n\t"                        // Final address calculation for node_value
                    "add.u64 t66, m100, t18;\n\t"                        // Final address calculation for node_value

                    "ld.s32 t33, [t25];\n\t"                             // Load c
                    "ld.s32 t34, [t26];\n\t"                             // Load c

                    "setp.eq.s32 t49, t33, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 t50, t34, mNOTPROCESSED;\n\t"           // Value for predicate

                    "MIS1_PRED_BODY_2_1:\n\t"                            // Predicate body
                    "@!t49 bra MIS1_PRED_BODY_2_2;\n\t"                  // Predicate on value of c
                    "atom.min.s32 t73, [t65], m101;\n\t"                 // Do min of node_value

                    "MIS1_PRED_BODY_2_2:\n\t"                            // Predicate body
                    "@!t50 bra MIS1_NEIGH_LOOP_2;\n\t"                   // Predicate on value of c
                    "atom.min.s32 t74, [t66], m101;\n\t"                 // Do min of node_value

                    "MIS1_NEIGH_LOOP_2:\n\t"

                    :                                                    // Outputs
                    : "l"(col_base_addr)                                 // Inputs
                );
            }

            for (; edge < row_end; edge++) {
                int * const col_base_addr = &col[edge];

                asm volatile
                (
                    ".reg .s32 a1;\n\t"                                  // Register for nid loaded from col
                    ".reg .u64 a17;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 a25;\n\t"                                 // Register for final address to load from c
                    ".reg .s32 a33;\n\t"                                 // Register for c
                    ".reg .u64 a65;\n\t"                                 // Register for final address to load from node_value
                    ".reg .s32 a73;\n\t"                                 // Register for node_value

                    ".reg .pred a49;\n\t"                                // Register for s predicate

                    "ld.s32 a1, [%0+0];\n\t"                             // Load nid
                    "mul.wide.s32 a17, a1, 4;\n\t"                       // Multiply nid for s address calculation
                    "add.u64 a25, m99, a17;\n\t"                         // Final address calculation for c
                    "add.u64 a65, m100, a17;\n\t"                        // Final address calculation for node_value
                    "ld.s32 a33, [a25];\n\t"                             // Load c

                    "setp.eq.s32 a49, a33, mNOTPROCESSED;\n\t"           // Value for predicate

                    "MIS1_PRED_BODY_1_1:\n\t"                            // Predicate body
                    "@!a49 bra MIS1_NEIGH_LOOP_1;\n\t"                   // Predicate on value of c
                    "atom.min.s32 a73, [a65], m101;\n\t"                 // Do min of node_value

                    "MIS1_NEIGH_LOOP_1:\n\t"

                    :                                                    // Outputs
                    : "l"(col_base_addr)                                 // Inputs
                );
            }
            */
        }
    }

    
    /*
    if (tx == 0) {
        __denovo_gpuEpilogue(SPECIAL_REGION);
        __denovo_gpuEpilogue(READ_ONLY_REGION);
        __denovo_gpuEpilogue(default_reg);
        __denovo_gpuEpilogue(rel_reg);
    }
    */
    cont[tid] = local_cont;
}


/**
 * mis2 kernel
 * @param row          csr pointer array
 * @param col          csr column index array
 * @param node_value   node value array
 * @param s_array      set array
 * @param c_array      status array
 * @param cu_array     status update array
 * @param min_array    node value array
 * @param num_nodes    number of vertices
 * @param num_edges    number of edges
 */
__global__ void
mis2(int *row, int *col, int *node_value, int *s_array, int *c_array,
     int *cu_array, int *min_array, int num_gpu_nodes, int num_edges)
{
    const int tx = threadIdx.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int edge = 0;
/*
    asm volatile
    (
        ".reg .s32 mINDEPENDENT;\n\t"                          // Register for s_array addr
        ".reg .s32 mNOTPROCESSED;\n\t"                         // Register for s_array addr
        ".reg .s32 mINACTIVE;\n\t"                             // Register for s_array addr

        "mov.s32 mINDEPENDENT, %0;\n\t"
        "mov.s32 mNOTPROCESSED, %1;\n\t"
        "mov.s32 mINACTIVE, %2;\n\t"

        :                                                      // Outputs
        : "r"(INDEPENDENT), "r"(NOT_PROCESSED), "r"(INACTIVE)  // Inputs
    );
    */
/*
    if (tx == 0) {
        __denovo_setAcquireRegion(SPECIAL_REGION);
        __denovo_addAcquireRegion(READ_ONLY_REGION);
        __denovo_addAcquireRegion(default_reg);
        __denovo_addAcquireRegion(rel_reg);
    }
    __syncthreads();
*/
    for (; tid < num_gpu_nodes; tid += blockDim.x * gridDim.x) {
        const int my_min_value = atomicAdd(&min_array[tid], 0);

        if (node_value[tid] <= my_min_value && c_array[tid] == NOT_PROCESSED) {
            // -1: Not processed -2: Inactive 2: Independent set
            // Put the item into the independent set
            s_array[tid] = INDEPENDENT;

            // Get the start and end pointers
            const int row_start = row[tid];
            const int row_end = row[tid + 1];

            // Set the status to inactive
            //cu_array[tid] = INACTIVE;

            atomicOr(&cu_array[tid], INACTIVE);
/*
            asm volatile
            (
                ".reg .u64 m99;\n\t"                   // Register for c_array addr
                ".reg .u64 m100;\n\t"                  // Register for cu_array addr

                "mov.u64 m99, %0;\n\t"
                "mov.u64 m100, %1;\n\t"

                :                                      // Outputs
                : "l"(c_array), "l"(cu_array)          // Inputs
            );
         */   
             // Mark all the neighbors inactive
             for (int edge = row_start; edge < row_end; edge++) {
                 const int neighbor = col[edge];

                 if (c_array[neighbor] == NOT_PROCESSED) {
                     //use status update array to avoid race
                     //cu_array[neighbor] = INACTIVE;
                     atomicOr(&cu_array[neighbor], INACTIVE);
                 }
             }
/*
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

                    ".reg .u64 m17;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m18;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m19;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m20;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m21;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m22;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m23;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 m24;\n\t"                                 // Register for multiplied nid value as address

                    ".reg .u64 m25;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 m26;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 m27;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 m28;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 m29;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 m30;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 m31;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 m32;\n\t"                                 // Register for final address to load from c

                    ".reg .s32 m33;\n\t"                                 // Register for c
                    ".reg .s32 m34;\n\t"                                 // Register for c
                    ".reg .s32 m35;\n\t"                                 // Register for c
                    ".reg .s32 m36;\n\t"                                 // Register for c
                    ".reg .s32 m37;\n\t"                                 // Register for c
                    ".reg .s32 m38;\n\t"                                 // Register for c
                    ".reg .s32 m39;\n\t"                                 // Register for c
                    ".reg .s32 m40;\n\t"                                 // Register for c

                    ".reg .u64 m65;\n\t"                                 // Register for final address to load from cu
                    ".reg .u64 m66;\n\t"                                 // Register for final address to load from cu
                    ".reg .u64 m67;\n\t"                                 // Register for final address to load from cu
                    ".reg .u64 m68;\n\t"                                 // Register for final address to load from cu
                    ".reg .u64 m69;\n\t"                                 // Register for final address to load from cu
                    ".reg .u64 m70;\n\t"                                 // Register for final address to load from cu
                    ".reg .u64 m71;\n\t"                                 // Register for final address to load from cu
                    ".reg .u64 m72;\n\t"                                 // Register for final address to load from cu

                    ".reg .b32 m73;\n\t"                                 // Register for cu
                    ".reg .b32 m74;\n\t"                                 // Register for cu
                    ".reg .b32 m75;\n\t"                                 // Register for cu
                    ".reg .b32 m76;\n\t"                                 // Register for cu
                    ".reg .b32 m77;\n\t"                                 // Register for cu
                    ".reg .b32 m78;\n\t"                                 // Register for cu
                    ".reg .b32 m79;\n\t"                                 // Register for cu
                    ".reg .b32 m80;\n\t"                                 // Register for cu

                    ".reg .pred m49;\n\t"                                // Register for c predicate
                    ".reg .pred m50;\n\t"                                // Register for c predicate
                    ".reg .pred m51;\n\t"                                // Register for c predicate
                    ".reg .pred m52;\n\t"                                // Register for c predicate
                    ".reg .pred m53;\n\t"                                // Register for c predicate
                    ".reg .pred m54;\n\t"                                // Register for c predicate
                    ".reg .pred m55;\n\t"                                // Register for c predicate
                    ".reg .pred m56;\n\t"                                // Register for c predicate

                    "ld.s32 m1, [%0+0];\n\t"                             // Load nid
                    "ld.s32 m2, [%0+4];\n\t"                             // Load nid
                    "ld.s32 m3, [%0+8];\n\t"                             // Load nid
                    "ld.s32 m4, [%0+12];\n\t"                            // Load nid
                    "ld.s32 m5, [%0+16];\n\t"                            // Load nid
                    "ld.s32 m6, [%0+20];\n\t"                            // Load nid
                    "ld.s32 m7, [%0+24];\n\t"                            // Load nid
                    "ld.s32 m8, [%0+28];\n\t"                            // Load nid

                    "mul.wide.s32 m17, m1, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 m18, m2, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 m19, m3, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 m20, m4, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 m21, m5, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 m22, m6, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 m23, m7, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 m24, m8, 4;\n\t"                       // Multiply nid for address calculation

                    "add.u64 m25, m99, m17;\n\t"                         // Final address calculation for c
                    "add.u64 m26, m99, m18;\n\t"                         // Final address calculation for c
                    "add.u64 m27, m99, m19;\n\t"                         // Final address calculation for c
                    "add.u64 m28, m99, m20;\n\t"                         // Final address calculation for c
                    "add.u64 m29, m99, m21;\n\t"                         // Final address calculation for c
                    "add.u64 m30, m99, m22;\n\t"                         // Final address calculation for c
                    "add.u64 m31, m99, m23;\n\t"                         // Final address calculation for c
                    "add.u64 m32, m99, m24;\n\t"                         // Final address calculation for c

                    "add.u64 m65, m100, m17;\n\t"                        // Final address calculation for cu
                    "add.u64 m66, m100, m18;\n\t"                        // Final address calculation for cu
                    "add.u64 m67, m100, m19;\n\t"                        // Final address calculation for cu
                    "add.u64 m68, m100, m20;\n\t"                        // Final address calculation for cu
                    "add.u64 m69, m100, m21;\n\t"                        // Final address calculation for cu
                    "add.u64 m70, m100, m22;\n\t"                        // Final address calculation for cu
                    "add.u64 m71, m100, m23;\n\t"                        // Final address calculation for cu
                    "add.u64 m72, m100, m24;\n\t"                        // Final address calculation for cu

                    "ld.s32 m33, [m25];\n\t"                             // Load c
                    "ld.s32 m34, [m26];\n\t"                             // Load c
                    "ld.s32 m35, [m27];\n\t"                             // Load c
                    "ld.s32 m36, [m28];\n\t"                             // Load c
                    "ld.s32 m37, [m29];\n\t"                             // Load c
                    "ld.s32 m38, [m30];\n\t"                             // Load c
                    "ld.s32 m39, [m31];\n\t"                             // Load c
                    "ld.s32 m40, [m32];\n\t"                             // Load c

                    "setp.eq.s32 m49, m33, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 m50, m34, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 m51, m35, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 m52, m36, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 m53, m37, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 m54, m38, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 m55, m39, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 m56, m40, mNOTPROCESSED;\n\t"           // Value for predicate

                    "MIS2_PRED_BODY_8_1:\n\t"                            // Predicate body
                    "@!m49 bra MIS2_PRED_BODY_8_2;\n\t"                  // Predicate on value of c
                    "atom.or.b32 m73, [m65], mINACTIVE;\n\t"             // Set cu

                    "MIS2_PRED_BODY_8_2:\n\t"                            // Predicate body
                    "@!m50 bra MIS2_PRED_BODY_8_3;\n\t"                  // Predicate on value of c
                    "atom.or.b32 m74, [m66], mINACTIVE;\n\t"             // Set cu

                    "MIS2_PRED_BODY_8_3:\n\t"                            // Predicate body
                    "@!m51 bra MIS2_PRED_BODY_8_4;\n\t"                  // Predicate on value of c
                    "atom.or.b32 m75, [m67], mINACTIVE;\n\t"             // Set cu

                    "MIS2_PRED_BODY_8_4:\n\t"                            // Predicate body
                    "@!m52 bra MIS2_PRED_BODY_8_5;\n\t"                  // Predicate on value of c
                    "atom.or.b32 m76, [m68], mINACTIVE;\n\t"             // Set cu

                    "MIS2_PRED_BODY_8_5:\n\t"                            // Predicate body
                    "@!m53 bra MIS2_PRED_BODY_8_6;\n\t"                  // Predicate on value of c
                    "atom.or.b32 m77, [m69], mINACTIVE;\n\t"             // Set cu

                    "MIS2_PRED_BODY_8_6:\n\t"                            // Predicate body
                    "@!m54 bra MIS2_PRED_BODY_8_7;\n\t"                  // Predicate on value of c
                    "atom.or.b32 m78, [m70], mINACTIVE;\n\t"             // Set cu

                    "MIS2_PRED_BODY_8_7:\n\t"                            // Predicate body
                    "@!m55 bra MIS2_PRED_BODY_8_8;\n\t"                  // Predicate on value of c
                    "atom.or.b32 m79, [m71], mINACTIVE;\n\t"             // Set cu

                    "MIS2_PRED_BODY_8_8:\n\t"                            // Predicate body
                    "@!m56 bra MIS2_NEIGH_LOOP_8;\n\t"                   // Predicate on value of c
                    "atom.or.b32 m80, [m72], mINACTIVE;\n\t"             // Set cu

                    "MIS2_NEIGH_LOOP_8:\n\t"

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

                    ".reg .u64 q17;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 q18;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 q19;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 q20;\n\t"                                 // Register for multiplied nid value as address

                    ".reg .u64 q25;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 q26;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 q27;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 q28;\n\t"                                 // Register for final address to load from c

                    ".reg .s32 q33;\n\t"                                 // Register for c
                    ".reg .s32 q34;\n\t"                                 // Register for c
                    ".reg .s32 q35;\n\t"                                 // Register for c
                    ".reg .s32 q36;\n\t"                                 // Register for c

                    ".reg .u64 q65;\n\t"                                 // Register for final address to load from cu
                    ".reg .u64 q66;\n\t"                                 // Register for final address to load from cu
                    ".reg .u64 q67;\n\t"                                 // Register for final address to load from cu
                    ".reg .u64 q68;\n\t"                                 // Register for final address to load from cu

                    ".reg .b32 q73;\n\t"                                 // Register for cu
                    ".reg .b32 q74;\n\t"                                 // Register for cu
                    ".reg .b32 q75;\n\t"                                 // Register for cu
                    ".reg .b32 q76;\n\t"                                 // Register for cu

                    ".reg .pred q49;\n\t"                                // Register for c predicate
                    ".reg .pred q50;\n\t"                                // Register for c predicate
                    ".reg .pred q51;\n\t"                                // Register for c predicate
                    ".reg .pred q52;\n\t"                                // Register for c predicate

                    "ld.s32 q1, [%0+0];\n\t"                             // Load nid
                    "ld.s32 q2, [%0+4];\n\t"                             // Load nid
                    "ld.s32 q3, [%0+8];\n\t"                             // Load nid
                    "ld.s32 q4, [%0+12];\n\t"                            // Load nid

                    "mul.wide.s32 q17, q1, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 q18, q2, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 q19, q3, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 q20, q4, 4;\n\t"                       // Multiply nid for address calculation

                    "add.u64 q25, m99, q17;\n\t"                         // Final address calculation for c
                    "add.u64 q26, m99, q18;\n\t"                         // Final address calculation for c
                    "add.u64 q27, m99, q19;\n\t"                         // Final address calculation for c
                    "add.u64 q28, m99, q20;\n\t"                         // Final address calculation for c

                    "add.u64 q65, m100, q17;\n\t"                        // Final address calculation for cu
                    "add.u64 q66, m100, q18;\n\t"                        // Final address calculation for cu
                    "add.u64 q67, m100, q19;\n\t"                        // Final address calculation for cu
                    "add.u64 q68, m100, q20;\n\t"                        // Final address calculation for cu

                    "ld.s32 q33, [q25];\n\t"                             // Load c
                    "ld.s32 q34, [q26];\n\t"                             // Load c
                    "ld.s32 q35, [q27];\n\t"                             // Load c
                    "ld.s32 q36, [q28];\n\t"                             // Load c

                    "setp.eq.s32 q49, q33, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 q50, q34, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 q51, q35, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 q52, q36, mNOTPROCESSED;\n\t"           // Value for predicate

                    "MIS2_PRED_BODY_4_1:\n\t"                            // Predicate body
                    "@!q49 bra MIS2_PRED_BODY_4_2;\n\t"                  // Predicate on value of c
                    "atom.or.b32 q73, [q65], mINACTIVE;\n\t"             // Set cu

                    "MIS2_PRED_BODY_4_2:\n\t"                            // Predicate body
                    "@!q50 bra MIS2_PRED_BODY_4_3;\n\t"                  // Predicate on value of c
                    "atom.or.b32 q74, [q66], mINACTIVE;\n\t"             // Set cu

                    "MIS2_PRED_BODY_4_3:\n\t"                            // Predicate body
                    "@!q51 bra MIS2_PRED_BODY_4_4;\n\t"                  // Predicate on value of c
                    "atom.or.b32 q75, [q67], mINACTIVE;\n\t"             // Set cu

                    "MIS2_PRED_BODY_4_4:\n\t"                            // Predicate body
                    "@!q52 bra MIS2_NEIGH_LOOP_4;\n\t"                   // Predicate on value of c
                    "atom.or.b32 q76, [q68], mINACTIVE;\n\t"             // Set cu

                    "MIS2_NEIGH_LOOP_4:\n\t"

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

                    ".reg .u64 t17;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 t18;\n\t"                                 // Register for multiplied nid value as address

                    ".reg .u64 t25;\n\t"                                 // Register for final address to load from c
                    ".reg .u64 t26;\n\t"                                 // Register for final address to load from c

                    ".reg .s32 t33;\n\t"                                 // Register for c
                    ".reg .s32 t34;\n\t"                                 // Register for c

                    ".reg .u64 t65;\n\t"                                 // Register for final address to load from cu
                    ".reg .u64 t66;\n\t"                                 // Register for final address to load from cu

                    ".reg .b32 t73;\n\t"                                 // Register for cu
                    ".reg .b32 t74;\n\t"                                 // Register for cu

                    ".reg .pred t49;\n\t"                                // Register for c predicate
                    ".reg .pred t50;\n\t"                                // Register for c predicate

                    "ld.s32 t1, [%0+0];\n\t"                             // Load nid
                    "ld.s32 t2, [%0+4];\n\t"                             // Load nid

                    "mul.wide.s32 t17, t1, 4;\n\t"                       // Multiply nid for address calculation
                    "mul.wide.s32 t18, t2, 4;\n\t"                       // Multiply nid for address calculation

                    "add.u64 t25, m99, t17;\n\t"                         // Final address calculation for c
                    "add.u64 t26, m99, t18;\n\t"                         // Final address calculation for c

                    "add.u64 t65, m100, t17;\n\t"                        // Final address calculation for cu
                    "add.u64 t66, m100, t18;\n\t"                        // Final address calculation for cu

                    "ld.s32 t33, [t25];\n\t"                             // Load c
                    "ld.s32 t34, [t26];\n\t"                             // Load c

                    "setp.eq.s32 t49, t33, mNOTPROCESSED;\n\t"           // Value for predicate
                    "setp.eq.s32 t50, t34, mNOTPROCESSED;\n\t"           // Value for predicate

                    "MIS2_PRED_BODY_2_1:\n\t"                            // Predicate body
                    "@!t49 bra MIS2_PRED_BODY_2_2;\n\t"                  // Predicate on value of c
                    "atom.or.b32 t73, [t65], mINACTIVE;\n\t"             // Set cu

                    "MIS2_PRED_BODY_2_2:\n\t"                            // Predicate body
                    "@!t50 bra MIS2_NEIGH_LOOP_2;\n\t"                   // Predicate on value of c
                    "atom.or.b32 t74, [t66], mINACTIVE;\n\t"             // Set cu

                    "MIS2_NEIGH_LOOP_2:\n\t"

                    :                                                    // Outputs
                    : "l"(col_base_addr)                                 // Inputs
                );
            }

            for (; edge < row_end; edge++) {
                int * const col_base_addr = &col[edge];

                asm volatile
                (
                    ".reg .s32 a1;\n\t"                                  // Register for nid loaded from col
                    ".reg .u64 a17;\n\t"                                 // Register for multiplied nid value as address
                    ".reg .u64 a25;\n\t"                                 // Register for final address to load from c
                    ".reg .s32 a33;\n\t"                                 // Register for c
                    ".reg .u64 a65;\n\t"                                 // Register for final address to load from cu
                    ".reg .b32 a73;\n\t"                                 // Register for cu

                    ".reg .pred a49;\n\t"                                // Register for s predicate

                    "ld.s32 a1, [%0+0];\n\t"                             // Load nid
                    "mul.wide.s32 a17, a1, 4;\n\t"                       // Multiply nid for address calculation
                    "add.u64 a25, m99, a17;\n\t"                         // Final address calculation for c
                    "add.u64 a65, m100, a17;\n\t"                        // Final address calculation for cu
                    "ld.s32 a33, [a25];\n\t"                             // Load c

                    "setp.eq.s32 a49, a33, mNOTPROCESSED;\n\t"           // Value for predicate

                    "MIS2_PRED_BODY_1_1:\n\t"                            // Predicate body
                    "@!a49 bra MIS2_NEIGH_LOOP_1;\n\t"                   // Predicate on value of c
                    "atom.or.b32 a73, [a65], mINACTIVE;\n\t"             // Set cu

                    "MIS2_NEIGH_LOOP_1:\n\t"

                    :                                                    // Outputs
                    : "l"(col_base_addr)                                 // Inputs
                );
            
            }
        */      
      }
    }
/*
    if (tx == 0) {
        __denovo_gpuEpilogue(SPECIAL_REGION);
        __denovo_gpuEpilogue(READ_ONLY_REGION);
        __denovo_gpuEpilogue(default_reg);
        __denovo_gpuEpilogue(rel_reg);
    }
    */
}


/**
 * mis3 kernel
 * @param cu_array     status update array
 * @param  c_array     status array
 * @param num_nodes    number of vertices
 */
__global__ void
mis3(int *cu_array, int *c_array, int *min_array, int num_gpu_nodes)
{
    const int tx = threadIdx.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
/*
    if (tx == 0) {
        __denovo_setAcquireRegion(SPECIAL_REGION);
        __denovo_addAcquireRegion(READ_ONLY_REGION);
        __denovo_addAcquireRegion(default_reg);
        __denovo_addAcquireRegion(rel_reg);
    }
    __syncthreads();
*/
    for (; tid < num_gpu_nodes; tid += blockDim.x * gridDim.x) {
        //set the status array
        const int status = atomicAdd(&cu_array[tid], 0);
        if (status == INACTIVE) {
            c_array[tid] = status;
        } else {
            atomicOr(&min_array[tid], INT_MAX);
        }
    }
/*
    if (tx == 0) {
        __denovo_gpuEpilogue(SPECIAL_REGION);
        __denovo_gpuEpilogue(READ_ONLY_REGION);
        __denovo_gpuEpilogue(default_reg);
        __denovo_gpuEpilogue(rel_reg);
    }
*/    
}
