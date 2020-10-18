
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


// Gemm operator cutlass_tensorop_s8_i8816gemm_s8_256x128_64x2_tn_align16
using cutlass_tensorop_s8_i8816gemm_s8_256x128_64x2_tn_align16_base = 
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,    // transposed B operand
    int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,    // transposed A operand
    int8_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      int8_t,
      16,
      int32_t,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    cutlass::arch::OpMultiplyAddSaturate
>::GemmKernel;

// Define named type
struct cutlass_tensorop_s8_i8816gemm_s8_256x128_64x2_tn_align16 : 
  public cutlass_tensorop_s8_i8816gemm_s8_256x128_64x2_tn_align16_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass_tensorop_s8_i8816gemm_s8_256x128_64x2_tn_align16(Manifest &manifest) {



  manifest.append(new GemmUniversalOperation<
      cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_s8_i8816gemm_s8_256x128_64x2_tn_align16>
    >("cutlass_tensorop_s8_i8816gemm_s8_256x128_64x2_tn_align16"));



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

