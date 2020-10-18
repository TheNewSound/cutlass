
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


  // Gemm operator cutlass_tensorop_f16_s1688gemm_planar_complex_array_f16_64x128_32x2_ch_align8
  using Operation_cutlass_tensorop_f16_s1688gemm_planar_complex_array_f16_64x128_32x2_ch_align8 = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kConjugate, 8,
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kConjugate, 8,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationPlanarComplex<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd
  >::GemmArrayKernel;

  struct cutlass_tensorop_f16_s1688gemm_planar_complex_array_f16_64x128_32x2_ch_align8 : public Operation_cutlass_tensorop_f16_s1688gemm_planar_complex_array_f16_64x128_32x2_ch_align8 { };


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_f16_s1688gemm_planar_complex_array_f16_64x128_32x2_ch_align8(Manifest &manifest) {



  manifest.append(new GemmPlanarComplexArrayOperation<
    cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_f16_s1688gemm_planar_complex_array_f16_64x128_32x2_ch_align8>
  >("cutlass_tensorop_f16_s1688gemm_planar_complex_array_f16_64x128_32x2_ch_align8"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

