
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


// Gemm operator cutlass_tensorop_c1688gemm_64x64_16x4_nc_align1
using cutlass_tensorop_c1688gemm_64x64_16x4_nc_align1_base = 
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::complex<float>, cutlass::layout::RowMajor, cutlass::ComplexTransform::kConjugate, 1,    // transposed B operand
    cutlass::complex<float>, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,    // transposed A operand
    cutlass::complex<float>, cutlass::layout::RowMajor,
    cutlass::complex<float>,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::complex<float>,
      1,
      cutlass::complex<float>,
      cutlass::complex<float>
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    cutlass::arch::OpMultiplyAddComplex
>::GemmKernel;

// Define named type
struct cutlass_tensorop_c1688gemm_64x64_16x4_nc_align1 : 
  public cutlass_tensorop_c1688gemm_64x64_16x4_nc_align1_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_c1688gemm_64x64_16x4_nc_align1(Manifest &manifest) {



  manifest.append(new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_c1688gemm_64x64_16x4_nc_align1>
    >("cutlass_tensorop_c1688gemm_64x64_16x4_nc_align1"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

