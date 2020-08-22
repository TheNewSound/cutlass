/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Matrix multiply
*/
#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#if ((__CUDACC_VER_MAJOR__ > 5))
#define CUTLASS_ARCH_MMA_SM30_SUPPORTED
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300))

#if ((__CUDACC_VER_MAJOR__ > 5))
#define CUTLASS_ARCH_MMA_SM30_ENABLED
#endif

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

  /////////////////////////////////////////////////////////////////////////////////////////////////

  /// Matrix multiply-add operation
  template <
    /// Layout of A matrix
    typename LayoutA,
    /// Layout of B matrix
    typename LayoutB,
    /// Layout of C matrix
    typename LayoutC
  >
  struct Mma<gemm::GemmShape<1, 1, 1>, 1, uint8_t, LayoutA, uint8_t, LayoutB, int32_t, LayoutC, OpMinimumAdd> {

    using Shape = gemm::GemmShape<1, 1, 1>;

    CUTLASS_HOST_DEVICE
    void operator()(
      Array<int32_t, 1> &d,
      Array<uint8_t, 1> const &a,
      Array<uint8_t, 1> const &b,
      Array<int32_t, 1> const &c
    ) {

      d[0] = min(a[0],b[0]) + c[0];
    }
  };

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Matrix minimum accumulate 114 - S32 accumulation
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix minimum-add operation: U16 = min(U8,U8) + U16
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct Mma<
  gemm::GemmShape<1,1,4>,
  1,
  uint8_t,
  LayoutA,
  uint8_t,
  LayoutB,
  int32_t,
  LayoutC,
  OpMinimumAdd> {

  using Shape = gemm::GemmShape<1, 1, 4>;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<int32_t, 1> &d,
    Array<uint8_t, 4> const &a,
    Array<uint8_t, 4> const &b,
    Array<int32_t, 1> const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_SM30_ENABLED)

    unsigned const &A = reinterpret_cast<unsigned const &>(a);
    unsigned const &B = reinterpret_cast<unsigned const &>(b);

    asm volatile("vmin4.s32.u32.u32.add %0, %1, %2, %3;"
                 : "=r"(d[0])
                 : "r"(A), "r"(B), "r"(c[0]));
#else

    d[0] = c[0];

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < 4; ++k) {
      d[0] += min(a[k], b[k]);
    }

#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass
