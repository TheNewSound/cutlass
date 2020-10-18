
/*
  Generated by gemm_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "gemm_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////


// Gemm operator cutlass_tensorop_u4_i8832gemm_u4_256x128_128x2_n64t64_align32
using cutlass_tensorop_u4_i8832gemm_u4_256x128_128x2_n64t64_align32_base = 
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::uint4b_t, cutlass::layout::ColumnMajorInterleaved<64>, cutlass::ComplexTransform::kNone, 32,
    cutlass::uint4b_t, cutlass::layout::RowMajorInterleaved<64>, cutlass::ComplexTransform::kNone, 32,
    cutlass::uint4b_t, cutlass::layout::ColumnMajorInterleaved<64>,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<8, 8, 32>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      cutlass::uint4b_t,
      16,
      int32_t,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    cutlass::arch::OpMultiplyAddSaturate
>::GemmKernel;

// Define named type
struct cutlass_tensorop_u4_i8832gemm_u4_256x128_128x2_n64t64_align32 : 
  public cutlass_tensorop_u4_i8832gemm_u4_256x128_128x2_n64t64_align32_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_u4_i8832gemm_u4_256x128_128x2_n64t64_align32(Manifest &manifest) {



  manifest.append(new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_u4_i8832gemm_u4_256x128_128x2_n64t64_align32>
    >("cutlass_tensorop_u4_i8832gemm_u4_256x128_128x2_n64t64_align32"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

