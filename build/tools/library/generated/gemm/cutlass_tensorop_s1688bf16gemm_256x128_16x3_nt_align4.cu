
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


// Gemm operator cutlass_tensorop_s1688bf16gemm_256x128_16x3_nt_align4
using cutlass_tensorop_s1688bf16gemm_256x128_16x3_nt_align4_base = 
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    float, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 4,    // transposed B operand
    float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 4,    // transposed A operand
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    cutlass::arch::OpMultiplyAddFastBF16
>::GemmKernel;

// Define named type
struct cutlass_tensorop_s1688bf16gemm_256x128_16x3_nt_align4 : 
  public cutlass_tensorop_s1688bf16gemm_256x128_16x3_nt_align4_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_s1688bf16gemm_256x128_16x3_nt_align4(Manifest &manifest) {



  manifest.append(new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_s1688bf16gemm_256x128_16x3_nt_align4>
    >("cutlass_tensorop_s1688bf16gemm_256x128_16x3_nt_align4"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

