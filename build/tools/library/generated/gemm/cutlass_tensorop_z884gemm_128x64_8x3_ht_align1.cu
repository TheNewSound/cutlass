
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


// Gemm operator cutlass_tensorop_z884gemm_128x64_8x3_ht_align1
using cutlass_tensorop_z884gemm_128x64_8x3_ht_align1_base = 
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::complex<double>, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,    // transposed B operand
    cutlass::complex<double>, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kConjugate, 1,    // transposed A operand
    cutlass::complex<double>, cutlass::layout::RowMajor,
    cutlass::complex<double>,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::complex<double>,
      1,
      cutlass::complex<double>,
      cutlass::complex<double>
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    cutlass::arch::OpMultiplyAddComplex
>::GemmKernel;

// Define named type
struct cutlass_tensorop_z884gemm_128x64_8x3_ht_align1 : 
  public cutlass_tensorop_z884gemm_128x64_8x3_ht_align1_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_z884gemm_128x64_8x3_ht_align1(Manifest &manifest) {



  manifest.append(new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_z884gemm_128x64_8x3_ht_align1>
    >("cutlass_tensorop_z884gemm_128x64_8x3_ht_align1"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

