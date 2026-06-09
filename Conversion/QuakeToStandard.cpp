#include "Conversion/QuakeToStandard.h"
#include "Dialect/Quake/QuakeDialect.h"
#include "Dialect/Quake/QuakeOps.h"
#include "Dialect/Quake/QuakeTypes.h"
#include "Dialect/CC/CCDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cmath>
#include <complex>

namespace mlir {
namespace quake_to_standard {

#define GEN_PASS_DEF_QUAKETOSTANDARD
#include "Conversion/QuakeToStandard.h.inc"

// ---------------------------------------------------------------------------
// Statevector type helpers
//
// Representation: tensor<(2 * 2^nQubits) x f32>
//   flat interleaved real/imag f32 values
//   sv[2*k]   = Re(amplitude[k])
//   sv[2*k+1] = Im(amplitude[k])
// ---------------------------------------------------------------------------

static RankedTensorType svTensorType(MLIRContext *ctx, int64_t nQubits) {
  int64_t nF32 = 2LL * (1LL << nQubits);
  return RankedTensorType::get({nF32}, Float32Type::get(ctx));
}

// ---------------------------------------------------------------------------
// Gate builders — take current tensor sv, return updated tensor.
//
// All gate builders use linalg.generic for parallel execution on CPU/GPU.
// The statevector is reshaped to expose qubit dimensions as tensor axes,
// allowing IREE to map each independent amplitude pair to a SIMD lane or
// CUDA thread block.
//
// Initial state is a function argument (not a compile-time constant) to
// prevent IREE from constant-folding the entire circuit at compile time.
// ---------------------------------------------------------------------------

// Apply 2×2 unitary with f32 SSA Value matrix entries via linalg.generic.
// Decomposes the statevector around qubitIdx into four 2D slices
// (qubit=0/1 × re/im), applies the matrix element-wise, and reassembles.
static Value buildApplyUnitaryV(OpBuilder &b, Location loc, Value sv,
                                int64_t nQubits, int64_t qubitIdx,
                                Value U00r, Value U00i,
                                Value U01r, Value U01i,
                                Value U10r, Value U10i,
                                Value U11r, Value U11i) {
  MLIRContext *ctx = b.getContext();
  auto f32Ty = Float32Type::get(ctx);

  // Decompose the 2^nQubits index space around the target qubit:
  //   nUpper = 2^(nQubits - qubitIdx - 1)  — upper-bit combinations
  //   nLower = 2^qubitIdx                  — lower-bit combinations
  // Reshape tensor<2*2^n x f32> → tensor<nUpper × 2 × nLower × 2>
  //   dim 0: upper bits, dim 1: target qubit (0 or 1),
  //   dim 2: lower bits, dim 3: re/im
  int64_t nUpper = 1LL << (nQubits - qubitIdx - 1);
  int64_t nLower = 1LL << qubitIdx;

  // ── 1. Expand flat sv to 4D.
  auto sv4dTy = RankedTensorType::get({nUpper, 2, nLower, 2}, f32Ty);
  Value sv4d = b.create<tensor::ExpandShapeOp>(loc, sv4dTy, sv,
      ReassociationIndices{{0, 1, 2, 3}});

  // ── 2. Slice out four 2D tensors (one per qubit-state × re/im combination).
  auto sliceTy = RankedTensorType::get({nUpper, nLower}, f32Ty);
  auto mkSlice = [&](int64_t qBit, int64_t ri) -> Value {
    SmallVector<OpFoldResult> off = {b.getIndexAttr(0),     b.getIndexAttr(qBit),
                                     b.getIndexAttr(0),     b.getIndexAttr(ri)};
    SmallVector<OpFoldResult> sz  = {b.getIndexAttr(nUpper), b.getIndexAttr(1),
                                     b.getIndexAttr(nLower), b.getIndexAttr(1)};
    SmallVector<OpFoldResult> st  = {b.getIndexAttr(1), b.getIndexAttr(1),
                                     b.getIndexAttr(1), b.getIndexAttr(1)};
    return b.create<tensor::ExtractSliceOp>(loc, sliceTy, sv4d, off, sz, st);
  };
  Value a0r = mkSlice(0, 0), a0i = mkSlice(0, 1);
  Value a1r = mkSlice(1, 0), a1i = mkSlice(1, 1);

  // ── 3. linalg.generic: 1D parallel update over (nUpper * nLower) elements.
  //
  //       Flatten slices to 1D before dispatch. IREE's CUDA codegen maps a 2D
  //       linalg.generic as (gridDim.y=nUpper, gridDim.x=nLower). When nUpper
  //       is large (e.g. QFT CR1(ctrl=n-1, tgt=0) gives nUpper=2^(n-2)), this
  //       exceeds CUDA's gridDim.y limit of 65535 at n>=24. A 1D kernel uses
  //       only gridDim.x (limit 2^31-1) and preserves full parallelism since
  //       all iterator types remain parallel. CollapseShapeOp/ExpandShapeOp are
  //       metadata-only and fold away during IREE lowering.
  int64_t n1D = nUpper * nLower;
  auto flat1DTy = RankedTensorType::get({n1D}, f32Ty);
  auto flatten  = ReassociationIndices{{0, 1}};
  Value a0r_f = b.create<tensor::CollapseShapeOp>(loc, flat1DTy, a0r, flatten);
  Value a0i_f = b.create<tensor::CollapseShapeOp>(loc, flat1DTy, a0i, flatten);
  Value a1r_f = b.create<tensor::CollapseShapeOp>(loc, flat1DTy, a1r, flatten);
  Value a1i_f = b.create<tensor::CollapseShapeOp>(loc, flat1DTy, a1i, flatten);

  auto idMap = AffineMap::getMultiDimIdentityMap(1, ctx);
  SmallVector<AffineMap> maps(8, idMap);
  SmallVector<utils::IteratorType> iters(1, utils::IteratorType::parallel);
  auto mkEmpty = [&]() -> Value {
    return b.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{n1D}, f32Ty);
  };

  auto generic = b.create<linalg::GenericOp>(
      loc,
      TypeRange{flat1DTy, flat1DTy, flat1DTy, flat1DTy},
      ValueRange{a0r_f, a0i_f, a1r_f, a1i_f},
      ValueRange{mkEmpty(), mkEmpty(), mkEmpty(), mkEmpty()},
      maps, iters,
      [U00r, U00i, U01r, U01i, U10r, U10i, U11r, U11i](
          OpBuilder &nb, Location nl, ValueRange args) {
        Value a0r_v = args[0], a0i_v = args[1];
        Value a1r_v = args[2], a1i_v = args[3];
        auto mul = [&](Value x, Value y) { return nb.create<arith::MulFOp>(nl, x, y); };
        auto add = [&](Value x, Value y) { return nb.create<arith::AddFOp>(nl, x, y); };
        auto sub = [&](Value x, Value y) { return nb.create<arith::SubFOp>(nl, x, y); };
        // na0 = U00*a0 + U01*a1
        Value na0r_v = sub(add(mul(U00r, a0r_v), mul(U01r, a1r_v)),
                           add(mul(U00i, a0i_v), mul(U01i, a1i_v)));
        Value na0i_v = add(add(mul(U00r, a0i_v), mul(U00i, a0r_v)),
                           add(mul(U01r, a1i_v), mul(U01i, a1r_v)));
        // na1 = U10*a0 + U11*a1
        Value na1r_v = sub(add(mul(U10r, a0r_v), mul(U11r, a1r_v)),
                           add(mul(U10i, a0i_v), mul(U11i, a1i_v)));
        Value na1i_v = add(add(mul(U10r, a0i_v), mul(U10i, a0r_v)),
                           add(mul(U11r, a1i_v), mul(U11i, a1r_v)));
        nb.create<linalg::YieldOp>(nl, ValueRange{na0r_v, na0i_v, na1r_v, na1i_v});
      });

  // Expand 1D results back to [nUpper x nLower] for insert_slice.
  Value na0r = b.create<tensor::ExpandShapeOp>(loc, sliceTy, generic.getResult(0), flatten);
  Value na0i = b.create<tensor::ExpandShapeOp>(loc, sliceTy, generic.getResult(1), flatten);
  Value na1r = b.create<tensor::ExpandShapeOp>(loc, sliceTy, generic.getResult(2), flatten);
  Value na1i = b.create<tensor::ExpandShapeOp>(loc, sliceTy, generic.getResult(3), flatten);

  // ── 4. Reassemble the 4D tensor from the updated slices.
  Value empty4d = b.create<tensor::EmptyOp>(loc,
      ArrayRef<int64_t>{nUpper, 2, nLower, 2}, f32Ty);
  auto mkInsert = [&](Value slice, Value dest, int64_t qBit, int64_t ri) -> Value {
    SmallVector<OpFoldResult> off = {b.getIndexAttr(0),     b.getIndexAttr(qBit),
                                     b.getIndexAttr(0),     b.getIndexAttr(ri)};
    SmallVector<OpFoldResult> sz  = {b.getIndexAttr(nUpper), b.getIndexAttr(1),
                                     b.getIndexAttr(nLower), b.getIndexAttr(1)};
    SmallVector<OpFoldResult> st  = {b.getIndexAttr(1), b.getIndexAttr(1),
                                     b.getIndexAttr(1), b.getIndexAttr(1)};
    return b.create<tensor::InsertSliceOp>(loc, slice, dest, off, sz, st);
  };
  Value sv4d_new = mkInsert(na0r, empty4d, 0, 0);
  sv4d_new      = mkInsert(na0i, sv4d_new, 0, 1);
  sv4d_new      = mkInsert(na1r, sv4d_new, 1, 0);
  sv4d_new      = mkInsert(na1i, sv4d_new, 1, 1);

  // ── 5. Collapse back to flat tensor<2*2^nQubits x f32>.
  auto flatTy = sv.getType().cast<RankedTensorType>();
  return b.create<tensor::CollapseShapeOp>(loc, flatTy, sv4d_new,
      ReassociationIndices{{0, 1, 2, 3}});
}

// Convenience wrapper: create f32 constants from compile-time floats.
static Value buildApplyUnitary(OpBuilder &b, Location loc, Value sv,
                               int64_t nQubits, int64_t qubitIdx,
                               float u00r, float u00i,
                               float u01r, float u01i,
                               float u10r, float u10i,
                               float u11r, float u11i) {
  auto f32Ty = Float32Type::get(b.getContext());
  auto mk = [&](float v) -> Value {
    return b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, v));
  };
  return buildApplyUnitaryV(b, loc, sv, nQubits, qubitIdx,
      mk(u00r), mk(u00i), mk(u01r), mk(u01i),
      mk(u10r), mk(u10i), mk(u11r), mk(u11i));
}

// Apply CNOT: flip qubit `tgtIdx` when qubit `ctrlIdx` is |1>.
//
// CNOT(ctrl, tgt) = identity on ctrl=0 half + X(tgt) on ctrl=1 half.
// Implementation:
//   1. Reshape sv around ctrl qubit: tensor<N> → tensor<nCU × 2 × nCL × 2>
//   2. Extract ctrl=1 slice: tensor<nCU × nCL × 2>  (an (n-1)-qubit subspace)
//   3. Collapse to flat tensor<2·2^(n-1)> and apply X via buildApplyUnitaryV
//      (fully parallel linalg.generic over 2^(n-1) amplitude pairs)
//   4. Expand result back, insert into sv4d, collapse to flat
//
// Effective tgt index in the (n-1)-qubit subspace:
//   effTgt = tgtIdx - 1  when ctrlIdx < tgtIdx  (ctrl was below tgt)
//   effTgt = tgtIdx      otherwise
static Value buildApplyCNOT(OpBuilder &b, Location loc, Value sv,
                             int64_t nQubits, int64_t ctrlIdx, int64_t tgtIdx) {
  MLIRContext *ctx = b.getContext();
  auto f32Ty = Float32Type::get(ctx);

  int64_t nCU = 1LL << (nQubits - ctrlIdx - 1);  // upper-bit combinations
  int64_t nCL = 1LL << ctrlIdx;                   // lower-bit combinations

  // ── 1. Reshape sv to expose the ctrl qubit.
  //       tensor<2*2^n> → tensor<nCU × 2 × nCL × 2>
  //         dim 0: upper bits, dim 1: ctrl qubit, dim 2: lower bits, dim 3: re/im
  auto sv4dTy = RankedTensorType::get({nCU, 2, nCL, 2}, f32Ty);
  Value sv4d = b.create<tensor::ExpandShapeOp>(loc, sv4dTy, sv,
      ReassociationIndices{{0, 1, 2, 3}});

  // ── 2. Collapse sv4d dims 2,3 → tensor<nCU × 2 × (nCL*2)>.
  //       This makes the ctrl=1 extract produce a 2D tensor [nCU, nCL*2], which
  //       IREE distributes with a 1D grid (gridDim.x only, limit 2^31-1).
  //       A 3D [nCU, nCL, 2] extract would use gridDim.y = nCU, overflowing
  //       CUDA's 65535 limit when nCU > 65535 (n >= 18 with ctrlIdx = 1).
  auto sv3dTy = RankedTensorType::get({nCU, 2, nCL * 2}, f32Ty);
  SmallVector<ReassociationIndices> outerReassoc(3);
  outerReassoc[0].push_back(0);
  outerReassoc[1].push_back(1);
  outerReassoc[2].push_back(2);
  outerReassoc[2].push_back(3);
  Value sv_3d = b.create<tensor::CollapseShapeOp>(loc, sv3dTy, sv4d, outerReassoc);

  // ── 3. Extract the ctrl=1 sub-tensor as 2D [nCU, nCL*2].
  auto sliceTy = RankedTensorType::get({nCU, nCL * 2}, f32Ty);
  SmallVector<OpFoldResult> off = {b.getIndexAttr(0),   b.getIndexAttr(1),
                                   b.getIndexAttr(0)};
  SmallVector<OpFoldResult> sz  = {b.getIndexAttr(nCU), b.getIndexAttr(1),
                                   b.getIndexAttr(nCL * 2)};
  SmallVector<OpFoldResult> st  = {b.getIndexAttr(1),   b.getIndexAttr(1),
                                   b.getIndexAttr(1)};
  Value sv_ctrl1 = b.create<tensor::ExtractSliceOp>(loc, sliceTy, sv_3d, off, sz, st);

  // ── 4. Collapse ctrl=1 slice to flat (n-1)-qubit statevector.
  int64_t nSubF32 = 2LL * (1LL << (nQubits - 1));
  auto subFlatTy = RankedTensorType::get({nSubF32}, f32Ty);
  Value sv_sub = b.create<tensor::CollapseShapeOp>(loc, subFlatTy, sv_ctrl1,
      ReassociationIndices{{0, 1}});

  // ── 5. Apply X gate on the tgt qubit within the (n-1)-qubit subspace.
  //       Effective tgt index shifts down by 1 when ctrl was below tgt.
  int64_t effTgt = (ctrlIdx < tgtIdx) ? tgtIdx - 1 : tgtIdx;
  auto mk = [&](float v) -> Value {
    return b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, v));
  };
  // X = [[0, 1], [1, 0]] — swaps amplitude pairs with tgt bit = 0 and 1.
  Value sv_sub_new = buildApplyUnitaryV(b, loc, sv_sub, nQubits - 1, effTgt,
      mk(0.f), mk(0.f), mk(1.f), mk(0.f),
      mk(1.f), mk(0.f), mk(0.f), mk(0.f));

  // ── 6. Expand updated subspace back to the 2D slice shape.
  Value sv_ctrl1_new = b.create<tensor::ExpandShapeOp>(loc, sliceTy, sv_sub_new,
      ReassociationIndices{{0, 1}});

  // ── 7. Insert the updated ctrl=1 slice back into sv_3d.
  Value sv_3d_new = b.create<tensor::InsertSliceOp>(loc, sv_ctrl1_new, sv_3d,
      off, sz, st);

  // ── 8. Expand sv_3d_new back to 4D [nCU × 2 × nCL × 2].
  Value sv4d_new = b.create<tensor::ExpandShapeOp>(loc, sv4dTy, sv_3d_new,
      outerReassoc);

  // ── 9. Collapse the 4D tensor back to flat.
  auto flatTy = sv.getType().cast<RankedTensorType>();
  return b.create<tensor::CollapseShapeOp>(loc, flatTy, sv4d_new,
      ReassociationIndices{{0, 1, 2, 3}});
}

// Controlled-R1(λ): apply [[1,0],[0,exp(iλ)]] to tgt when ctrl=|1⟩.
// c = cos(λ), s = sin(λ) as f32 Values.
static Value buildApplyCR1(OpBuilder &b, Location loc, Value sv,
                            int64_t nQubits, int64_t ctrlIdx, int64_t tgtIdx,
                            Value c, Value s) {
  MLIRContext *ctx = b.getContext();
  auto f32Ty = Float32Type::get(ctx);

  int64_t nCU = 1LL << (nQubits - ctrlIdx - 1);
  int64_t nCL = 1LL << ctrlIdx;

  // ── 1. Reshape sv to expose the ctrl qubit.
  auto sv4dTy = RankedTensorType::get({nCU, 2, nCL, 2}, f32Ty);
  Value sv4d = b.create<tensor::ExpandShapeOp>(loc, sv4dTy, sv,
      ReassociationIndices{{0, 1, 2, 3}});

  // ── 2. Collapse sv4d dims 2,3 → tensor<nCU × 2 × (nCL*2)>.
  //       This makes the ctrl=1 extract produce a 2D tensor [nCU, nCL*2], which
  //       IREE distributes with a 1D grid (gridDim.x only, limit 2^31-1).
  //       A 3D [nCU, nCL, 2] extract would use gridDim.y = nCU, overflowing
  //       CUDA's 65535 limit when nCU > 65535 (n >= 18 with ctrlIdx = 1).
  auto sv3dTy = RankedTensorType::get({nCU, 2, nCL * 2}, f32Ty);
  SmallVector<ReassociationIndices> outerReassoc(3);
  outerReassoc[0].push_back(0);
  outerReassoc[1].push_back(1);
  outerReassoc[2].push_back(2);
  outerReassoc[2].push_back(3);
  Value sv_3d = b.create<tensor::CollapseShapeOp>(loc, sv3dTy, sv4d, outerReassoc);

  // ── 3. Extract the ctrl=1 sub-tensor as 2D [nCU, nCL*2].
  auto sliceTy = RankedTensorType::get({nCU, nCL * 2}, f32Ty);
  SmallVector<OpFoldResult> off = {b.getIndexAttr(0),   b.getIndexAttr(1),
                                   b.getIndexAttr(0)};
  SmallVector<OpFoldResult> sz  = {b.getIndexAttr(nCU), b.getIndexAttr(1),
                                   b.getIndexAttr(nCL * 2)};
  SmallVector<OpFoldResult> st  = {b.getIndexAttr(1),   b.getIndexAttr(1),
                                   b.getIndexAttr(1)};
  Value sv_ctrl1 = b.create<tensor::ExtractSliceOp>(loc, sliceTy, sv_3d, off, sz, st);

  // ── 4. Collapse ctrl=1 slice to flat (n-1)-qubit statevector.
  int64_t nSubF32 = 2LL * (1LL << (nQubits - 1));
  auto subFlatTy = RankedTensorType::get({nSubF32}, f32Ty);
  Value sv_sub = b.create<tensor::CollapseShapeOp>(loc, subFlatTy, sv_ctrl1,
      ReassociationIndices{{0, 1}});

  // ── 5. Apply R1(λ) = [[1,0],[0,c+is]] on tgt within (n-1)-qubit subspace.
  int64_t effTgt = (ctrlIdx < tgtIdx) ? tgtIdx - 1 : tgtIdx;
  Value zero = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, 0.f));
  Value one  = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, 1.f));
  Value sv_sub_new = buildApplyUnitaryV(b, loc, sv_sub, nQubits - 1, effTgt,
      one, zero, zero, zero,
      zero, zero, c, s);

  // ── 6. Expand updated subspace back to the 2D slice shape.
  Value sv_ctrl1_new = b.create<tensor::ExpandShapeOp>(loc, sliceTy, sv_sub_new,
      ReassociationIndices{{0, 1}});

  // ── 7. Insert the updated ctrl=1 slice back into sv_3d.
  Value sv_3d_new = b.create<tensor::InsertSliceOp>(loc, sv_ctrl1_new, sv_3d,
      off, sz, st);

  // ── 8. Expand sv_3d_new back to 4D [nCU × 2 × nCL × 2].
  Value sv4d_new = b.create<tensor::ExpandShapeOp>(loc, sv4dTy, sv_3d_new,
      outerReassoc);

  // ── 9. Collapse the 4D tensor back to flat.
  auto flatTy = sv.getType().cast<RankedTensorType>();
  return b.create<tensor::CollapseShapeOp>(loc, flatTy, sv4d_new,
      ReassociationIndices{{0, 1, 2, 3}});
}

// Deterministic Z-basis measurement bit: true when P(qubit=1) > 0.5.
// This is correct for basis states and intentionally not stochastic yet.
static Value buildMeasureZBit(OpBuilder &b, Location loc, Value sv,
                              int64_t nQubits, int64_t qubitIdx) {
  auto f32Ty = Float32Type::get(b.getContext());
  auto i64Ty = b.getI64Type();
  auto idxTy = b.getIndexType();

  int64_t nComplex = 1LL << nQubits;

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value nC = b.create<arith::ConstantIndexOp>(loc, nComplex);

  Value zeroF = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, 0.0f));
  Value halfF = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, 0.5f));
  Value qi = b.create<arith::ConstantIntOp>(loc, qubitIdx, i64Ty);
  Value one64 = b.create<arith::ConstantIntOp>(loc, 1, i64Ty);
  Value zero64 = b.create<arith::ConstantIntOp>(loc, 0, i64Ty);
  Value stride = b.create<arith::ShLIOp>(loc, one64, qi);

  auto loop = b.create<scf::ForOp>(
      loc, c0, nC, c1, ValueRange{zeroF},
      [sv, stride, one64, zero64, i64Ty, idxTy]
      (OpBuilder &body, Location l, Value k, ValueRange iters) {
        Value acc = iters[0];
        Value k64 = body.create<arith::IndexCastOp>(l, i64Ty, k);
        Value bit = body.create<arith::AndIOp>(l, k64, stride);
        Value isOne = body.create<arith::CmpIOp>(
            l, arith::CmpIPredicate::ne, bit, zero64);

        Value next = body.create<scf::IfOp>(
            l, isOne,
            [sv, acc, k64, one64, idxTy](OpBuilder &tb, Location tl) {
              Value kr64 = tb.create<arith::ShLIOp>(tl, k64, one64);
              Value ki64 = tb.create<arith::AddIOp>(tl, kr64, one64);
              Value kr = tb.create<arith::IndexCastOp>(tl, idxTy, kr64);
              Value ki = tb.create<arith::IndexCastOp>(tl, idxTy, ki64);
              Value re = tb.create<tensor::ExtractOp>(tl, sv, kr);
              Value im = tb.create<tensor::ExtractOp>(tl, sv, ki);
              Value re2 = tb.create<arith::MulFOp>(tl, re, re);
              Value im2 = tb.create<arith::MulFOp>(tl, im, im);
              Value prob = tb.create<arith::AddFOp>(tl, re2, im2);
              Value sum = tb.create<arith::AddFOp>(tl, acc, prob);
              tb.create<scf::YieldOp>(tl, sum);
            },
            [acc](OpBuilder &eb, Location el) {
              eb.create<scf::YieldOp>(el, acc);
            }).getResult(0);

        body.create<scf::YieldOp>(l, next);
      });

  return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
                                loop.getResult(0), halfF);
}

// Reset qubit `qubitIdx` to |0>. For each target-bit pair, move probability
// mass to the bit-0 amplitude and zero the bit-1 amplitude.
static Value buildApplyReset(OpBuilder &b, Location loc, Value sv,
                             int64_t nQubits, int64_t qubitIdx) {
  auto f32Ty = Float32Type::get(b.getContext());
  auto i64Ty = b.getI64Type();
  auto idxTy = b.getIndexType();

  int64_t nComplex = 1LL << nQubits;
  int64_t nPairs = nComplex / 2;

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value nP = b.create<arith::ConstantIndexOp>(loc, nPairs);
  Value zeroF = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, 0.0f));

  Value qi64 = b.create<arith::ConstantIntOp>(loc, qubitIdx, i64Ty);
  Value one64 = b.create<arith::ConstantIntOp>(loc, 1, i64Ty);
  Value stride = b.create<arith::ShLIOp>(loc, one64, qi64);
  Value lmask = b.create<arith::SubIOp>(loc, stride, one64);
  Value qp1 = b.create<arith::AddIOp>(loc, qi64, one64);

  auto loop = b.create<scf::ForOp>(
      loc, c0, nP, c1, ValueRange{sv},
      [zeroF, qi64, stride, lmask, qp1, one64, i64Ty, idxTy]
      (OpBuilder &body, Location l, Value j, ValueRange iters) {
        Value svCur = iters[0];

        Value j64 = body.create<arith::IndexCastOp>(l, i64Ty, j);
        Value lower = body.create<arith::AndIOp>(l, j64, lmask);
        Value uHalf = body.create<arith::ShRUIOp>(l, j64, qi64);
        Value upper = body.create<arith::ShLIOp>(l, uHalf, qp1);
        Value k0_64 = body.create<arith::AddIOp>(l, upper, lower);
        Value k1_64 = body.create<arith::AddIOp>(l, k0_64, stride);

        Value k0r64 = body.create<arith::ShLIOp>(l, k0_64, one64);
        Value k0i64 = body.create<arith::AddIOp>(l, k0r64, one64);
        Value k1r64 = body.create<arith::ShLIOp>(l, k1_64, one64);
        Value k1i64 = body.create<arith::AddIOp>(l, k1r64, one64);
        Value k0r = body.create<arith::IndexCastOp>(l, idxTy, k0r64);
        Value k0i = body.create<arith::IndexCastOp>(l, idxTy, k0i64);
        Value k1r = body.create<arith::IndexCastOp>(l, idxTy, k1r64);
        Value k1i = body.create<arith::IndexCastOp>(l, idxTy, k1i64);

        Value a0r = body.create<tensor::ExtractOp>(l, svCur, k0r);
        Value a0i = body.create<tensor::ExtractOp>(l, svCur, k0i);
        Value a1r = body.create<tensor::ExtractOp>(l, svCur, k1r);
        Value a1i = body.create<tensor::ExtractOp>(l, svCur, k1i);

        Value a0r2 = body.create<arith::MulFOp>(l, a0r, a0r);
        Value a0i2 = body.create<arith::MulFOp>(l, a0i, a0i);
        Value a1r2 = body.create<arith::MulFOp>(l, a1r, a1r);
        Value a1i2 = body.create<arith::MulFOp>(l, a1i, a1i);
        Value p0 = body.create<arith::AddFOp>(l, a0r2, a0i2);
        Value p1 = body.create<arith::AddFOp>(l, a1r2, a1i2);
        Value prob = body.create<arith::AddFOp>(l, p0, p1);
        Value resetAmp = body.create<math::SqrtOp>(l, prob);

        Value s0 = body.create<tensor::InsertOp>(l, resetAmp, svCur, k0r);
        Value s1 = body.create<tensor::InsertOp>(l, zeroF, s0, k0i);
        Value s2 = body.create<tensor::InsertOp>(l, zeroF, s1, k1r);
        Value s3 = body.create<tensor::InsertOp>(l, zeroF, s2, k1i);
        body.create<scf::YieldOp>(l, s3);
      });
  return loop.getResult(0);
}

// ---------------------------------------------------------------------------
// Rotation angle helpers
//
// Rotation gates accept either a compile-time arith.constant (constant path:
// trig folded at conversion time) or a runtime f64 SSA value, e.g. a function
// argument (dynamic path: math.cos/sin emitted into the IR).
// ---------------------------------------------------------------------------

// Ensure v is f64; extend from f32 if necessary.
static Value asF64(OpBuilder &b, Location loc, Value v) {
  auto f64Ty = b.getF64Type();
  if (v.getType() == f64Ty)
    return v;
  return b.create<arith::ExtFOp>(loc, f64Ty, v);
}

// Build runtime cos/sin Values truncated to f32.
// `angle` must already be f64. Returns {cos(angle), sin(angle)} as f32.
static std::pair<Value, Value> buildCosSin(OpBuilder &b, Location loc,
                                           Value angle) {
  auto f32Ty = Float32Type::get(b.getContext());
  Value c64 = b.create<math::CosOp>(loc, angle);
  Value s64 = b.create<math::SinOp>(loc, angle);
  Value c   = b.create<arith::TruncFOp>(loc, f32Ty, c64);
  Value s   = b.create<arith::TruncFOp>(loc, f32Ty, s64);
  return {c, s};
}

// ---------------------------------------------------------------------------
// Quantum function lowering
//
// Walks the quake ops in a function body in order, maintaining:
//   veqToSv: quake.veq value → current statevector tensor
//   refInfo: quake.ref value → (parent veq value, qubit index)
//
// Each gate op reads `veqToSv[veq]`, builds the updated tensor, and stores
// it back.  This threads tensor SSA values through sequential gate ops
// without requiring memref aliasing.
//
// The function's return type is updated to include the final statevector
// tensor so IREE does not DCE the entire function body.
// ---------------------------------------------------------------------------

static LogicalResult lowerQuantumFunction(func::FuncOp fn, MLIRContext *ctx) {
  // Map: original quake.veq SSA value → current tensor SSA value
  llvm::DenseMap<Value, Value> veqToSv;
  // Map: original quake.ref SSA value → (parent veq SSA value, qubit index)
  llvm::DenseMap<Value, std::pair<Value, int64_t>> refInfo;
  // Map: original quake.measure SSA value → lowered classical bit
  llvm::DenseMap<Value, Value> measToBit;

  SmallVector<Operation *> toErase;
  SmallVector<Type> addedArgTypes; // sv arg types prepended to the function
  Value finalSv; // last statevector tensor produced (returned from function)

  for (Block &block : fn.getBody()) {
    for (Operation &opRef : block.getOperations()) {
      Operation *op = &opRef;
      OpBuilder b(op);
      Location loc = op->getLoc();

      if (auto alloca = dyn_cast<quake::AllocaOp>(op)) {
        auto veqTy = alloca.getType().dyn_cast<quake::VeqType>();
        if (!veqTy)
          return op->emitError("expected VeqType result from alloca");
        if (!veqTy.hasSpecifiedSize())
          return op->emitError("dynamic qubit count not yet supported");
        int64_t nQubits = veqTy.getSize();
        // Add the initial statevector as a function argument rather than a
        // compile-time constant.  A dense constant would let IREE constant-fold
        // the entire circuit away; a runtime argument forces actual execution.
        auto svTy = svTensorType(ctx, nQubits);
        Value sv = fn.getBody().front().addArgument(svTy, loc);
        addedArgTypes.push_back(svTy);
        veqToSv[alloca.getResult()] = sv;
        finalSv = sv;
        toErase.push_back(op);

      } else if (auto exRef = dyn_cast<quake::ExtractRefOp>(op)) {
        Value veq = exRef.getVeq();
        if (!exRef.hasConstantIndex())
          return op->emitError("dynamic qubit index not yet supported");
        int64_t qi = (int64_t)exRef.getRawIndex();
        refInfo[exRef.getResult()] = {veq, qi};
        toErase.push_back(op);

      } else if (auto y = dyn_cast<quake::YOp>(op)) {
        if (!y.getControls().empty())
          return op->emitError("controlled-Y not yet supported");
        Value ref = y.getTargets()[0];
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("Y gate: ref not found in refInfo");
        auto [veq, qi] = it->second;
        int64_t nQubits = veq.getType().cast<quake::VeqType>().getSize();
        Value sv = veqToSv[veq];
        // Pauli-Y: [[0, -i], [i, 0]]
        Value new_sv = buildApplyUnitary(b, loc, sv, nQubits, qi,
            0.f,  0.f,  0.f, -1.f,
            0.f,  1.f,  0.f,  0.f);
        veqToSv[veq] = new_sv;
        finalSv = new_sv;
        toErase.push_back(op);

      } else if (auto z = dyn_cast<quake::ZOp>(op)) {
        auto ctrls = z.getControls();
        Value ref = z.getTargets()[0];
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("Z gate: ref not found in refInfo");
        auto [veq, qi] = it->second;
        int64_t nQubits = veq.getType().cast<quake::VeqType>().getSize();
        Value sv = veqToSv[veq];
        Value new_sv;
        if (ctrls.empty()) {
          // Pauli-Z: [[1, 0], [0, -1]]
          new_sv = buildApplyUnitary(b, loc, sv, nQubits, qi,
              1.f, 0.f,  0.f, 0.f,
              0.f, 0.f, -1.f, 0.f);
        } else if (ctrls.size() == 1) {
          Value ctrlRef = ctrls[0];
          auto cit = refInfo.find(ctrlRef);
          if (cit == refInfo.end())
            return op->emitError("CZ: control ref not found in refInfo");
          if (cit->second.first != veq)
            return op->emitError("CZ across different qvectors not yet supported");
          int64_t ctrlQi = cit->second.second;
          constexpr float k = 0.7071067811865476f; // 1/sqrt(2)
          // CZ(c, t) = H(t); CNOT(c, t); H(t).
          Value h0 = buildApplyUnitary(b, loc, sv, nQubits, qi,
              k, 0.f,  k, 0.f,
              k, 0.f, -k, 0.f);
          Value cx = buildApplyCNOT(b, loc, h0, nQubits, ctrlQi, qi);
          new_sv = buildApplyUnitary(b, loc, cx, nQubits, qi,
              k, 0.f,  k, 0.f,
              k, 0.f, -k, 0.f);
        } else {
          return op->emitError("Z with >1 controls not yet supported");
        }
        veqToSv[veq] = new_sv;
        finalSv = new_sv;
        toErase.push_back(op);

      } else if (auto h = dyn_cast<quake::HOp>(op)) {
        if (!h.getControls().empty())
          return op->emitError("controlled-H not yet supported");
        Value ref = h.getTargets()[0];
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("H gate: ref not found in refInfo");
        auto [veq, qi] = it->second;
        auto veqTy = veq.getType().cast<quake::VeqType>();
        int64_t nQubits = veqTy.getSize();
        Value sv = veqToSv[veq];
        constexpr float k = 0.7071067811865476f; // 1/sqrt(2)
        Value new_sv = buildApplyUnitary(b, loc, sv, nQubits, qi,
            k, 0.f,  k, 0.f,
            k, 0.f, -k, 0.f);
        veqToSv[veq] = new_sv;
        finalSv = new_sv;
        toErase.push_back(op);

      } else if (auto x = dyn_cast<quake::XOp>(op)) {
        auto ctrls = x.getControls();
        Value ref = x.getTargets()[0];
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("X gate: target ref not found in refInfo");
        auto [veq, tgtQi] = it->second;
        auto veqTy = veq.getType().cast<quake::VeqType>();
        int64_t nQubits = veqTy.getSize();
        Value sv = veqToSv[veq];
        Value new_sv;
        if (ctrls.empty()) {
          // Pauli-X (NOT gate)
          new_sv = buildApplyUnitary(b, loc, sv, nQubits, tgtQi,
              0.f, 0.f,  1.f, 0.f,
              1.f, 0.f,  0.f, 0.f);
        } else if (ctrls.size() == 1) {
          Value ctrlRef = ctrls[0];
          auto cit = refInfo.find(ctrlRef);
          if (cit == refInfo.end())
            return op->emitError("CNOT: control ref not found in refInfo");
          int64_t ctrlQi = cit->second.second;
          new_sv = buildApplyCNOT(b, loc, sv, nQubits, ctrlQi, tgtQi);
        } else {
          return op->emitError("X with >1 controls not yet supported");
        }
        veqToSv[veq] = new_sv;
        finalSv = new_sv;
        toErase.push_back(op);

      } else if (auto swap = dyn_cast<quake::SwapOp>(op)) {
        if (!swap.getControls().empty())
          return op->emitError("controlled-SWAP not yet supported");
        Value ref0 = swap.getTargets()[0];
        Value ref1 = swap.getTargets()[1];
        auto it0 = refInfo.find(ref0);
        auto it1 = refInfo.find(ref1);
        if (it0 == refInfo.end() || it1 == refInfo.end())
          return op->emitError("SWAP: target ref not found in refInfo");
        auto [veq0, qi0] = it0->second;
        auto [veq1, qi1] = it1->second;
        if (veq0 != veq1)
          return op->emitError("SWAP across different qvectors not yet supported");
        if (qi0 == qi1)
          return op->emitError("SWAP target refs must be distinct");
        int64_t nQubits = veq0.getType().cast<quake::VeqType>().getSize();
        Value sv = veqToSv[veq0];
        // SWAP(a, b) = CNOT(a, b); CNOT(b, a); CNOT(a, b).
        Value s0 = buildApplyCNOT(b, loc, sv, nQubits, qi0, qi1);
        Value s1 = buildApplyCNOT(b, loc, s0, nQubits, qi1, qi0);
        Value new_sv = buildApplyCNOT(b, loc, s1, nQubits, qi0, qi1);
        veqToSv[veq0] = new_sv;
        finalSv = new_sv;
        toErase.push_back(op);

      } else if (auto s = dyn_cast<quake::SOp>(op)) {
        if (!s.getControls().empty())
          return op->emitError("controlled-S not yet supported");
        Value ref = s.getTargets()[0];
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("S: target ref not found in refInfo");
        auto [veq, qi] = it->second;
        int64_t nQubits = veq.getType().cast<quake::VeqType>().getSize();
        Value sv = veqToSv[veq];
        // S = [[1, 0], [0, i]], S† = [[1, 0], [0, -i]]
        float sign = s.isAdj() ? -1.f : 1.f;
        Value new_sv = buildApplyUnitary(b, loc, sv, nQubits, qi,
            1.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, sign);
        veqToSv[veq] = new_sv;
        finalSv = new_sv;
        toErase.push_back(op);

      } else if (auto t = dyn_cast<quake::TOp>(op)) {
        if (!t.getControls().empty())
          return op->emitError("controlled-T not yet supported");
        Value ref = t.getTargets()[0];
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("T: target ref not found in refInfo");
        auto [veq, qi] = it->second;
        int64_t nQubits = veq.getType().cast<quake::VeqType>().getSize();
        Value sv = veqToSv[veq];
        constexpr float k = 0.7071067811865476f; // 1/sqrt(2)
        float imag = t.isAdj() ? -k : k;
        // T = [[1, 0], [0, exp(i*pi/4)]]
        Value new_sv = buildApplyUnitary(b, loc, sv, nQubits, qi,
            1.f, 0.f, 0.f, 0.f,
            0.f, 0.f, k, imag);
        veqToSv[veq] = new_sv;
        finalSv = new_sv;
        toErase.push_back(op);

      } else if (auto r1 = dyn_cast<quake::R1Op>(op)) {
        auto ctrls = r1.getControls();
        if (ctrls.size() > 1)
          return op->emitError("R1 with >1 controls not yet supported");
        Value angleVal = r1.getParameters()[0];
        Value ref = r1.getTargets()[0];
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("R1: target ref not found in refInfo");
        auto [veq, qi] = it->second;
        int64_t nQubits = veq.getType().cast<quake::VeqType>().getSize();
        Value sv = veqToSv[veq];
        Value new_sv;
        if (ctrls.size() == 1) {
          // Controlled-R1: apply phase gate to tgt only when ctrl=|1⟩.
          Value ctrlRef = ctrls[0];
          auto cit = refInfo.find(ctrlRef);
          if (cit == refInfo.end())
            return op->emitError("CR1: control ref not found in refInfo");
          int64_t ctrlQi = cit->second.second;
          if (auto cstOp = angleVal.getDefiningOp<arith::ConstantOp>()) {
            double lam = cstOp.getValue().cast<FloatAttr>().getValueAsDouble();
            if (r1.isAdj()) lam *= -1.0;
            auto f32Ty = Float32Type::get(b.getContext());
            Value c = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, (float)std::cos(lam)));
            Value s = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, (float)std::sin(lam)));
            new_sv = buildApplyCR1(b, loc, sv, nQubits, ctrlQi, qi, c, s);
          } else {
            auto f64Ty = b.getF64Type();
            Value lam = asF64(b, loc, angleVal);
            if (r1.isAdj()) {
              Value neg1 = b.create<arith::ConstantOp>(loc, FloatAttr::get(f64Ty, -1.0));
              lam = b.create<arith::MulFOp>(loc, lam, neg1);
            }
            auto [c, s] = buildCosSin(b, loc, lam);
            new_sv = buildApplyCR1(b, loc, sv, nQubits, ctrlQi, qi, c, s);
          }
        } else if (auto cstOp = angleVal.getDefiningOp<arith::ConstantOp>()) {
          // Uncontrolled, constant angle: fold trig at conversion time.
          double lam = cstOp.getValue().cast<FloatAttr>().getValueAsDouble();
          if (r1.isAdj()) lam *= -1.0;
          double c = std::cos(lam), s = std::sin(lam);
          new_sv = buildApplyUnitary(b, loc, sv, nQubits, qi,
              1.f, 0.f,       0.f, 0.f,
              0.f, 0.f, (float)c, (float)s);
        } else {
          // Uncontrolled, dynamic angle: runtime math.cos / math.sin.
          auto f64Ty = b.getF64Type();
          auto f32Ty = Float32Type::get(b.getContext());
          Value lam = asF64(b, loc, angleVal);
          if (r1.isAdj()) {
            Value neg1 = b.create<arith::ConstantOp>(loc, FloatAttr::get(f64Ty, -1.0));
            lam = b.create<arith::MulFOp>(loc, lam, neg1);
          }
          auto [c, s] = buildCosSin(b, loc, lam);
          Value zero = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, 0.f));
          Value one  = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, 1.f));
          new_sv = buildApplyUnitaryV(b, loc, sv, nQubits, qi,
              one, zero, zero, zero,
              zero, zero, c, s);
        }
        veqToSv[veq] = new_sv;
        finalSv = new_sv;
        toErase.push_back(op);

      } else if (auto rx = dyn_cast<quake::RxOp>(op)) {
        if (!rx.getControls().empty())
          return op->emitError("controlled-Rx not yet supported");
        Value angleVal = rx.getParameters()[0];
        Value ref = rx.getTargets()[0];
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("Rx: target ref not found in refInfo");
        auto [veq, qi] = it->second;
        int64_t nQubits = veq.getType().cast<quake::VeqType>().getSize();
        Value sv = veqToSv[veq];
        Value new_sv;
        if (auto cstOp = angleVal.getDefiningOp<arith::ConstantOp>()) {
          // Constant path: fold trig at conversion time.
          double theta = cstOp.getValue().cast<FloatAttr>().getValueAsDouble();
          double c = std::cos(theta / 2.0), s = std::sin(theta / 2.0);
          // RX(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
          new_sv = buildApplyUnitary(b, loc, sv, nQubits, qi,
              (float)c, 0.f,       0.f, (float)-s,
              0.f,      (float)-s, (float)c, 0.f);
        } else {
          // Dynamic path: runtime angle via math.cos / math.sin.
          auto f64Ty = b.getF64Type();
          auto f32Ty = Float32Type::get(b.getContext());
          Value half  = b.create<arith::ConstantOp>(loc, FloatAttr::get(f64Ty, 0.5));
          Value ht    = b.create<arith::MulFOp>(loc, asF64(b, loc, angleVal), half);
          auto [c, s] = buildCosSin(b, loc, ht);
          Value zero  = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, 0.f));
          Value ns    = b.create<arith::NegFOp>(loc, s);
          // RX(θ) = [[c, -i·s], [-i·s, c]]
          new_sv = buildApplyUnitaryV(b, loc, sv, nQubits, qi,
              c, zero, zero, ns, zero, ns, c, zero);
        }
        veqToSv[veq] = new_sv;
        finalSv = new_sv;
        toErase.push_back(op);

      } else if (auto ry = dyn_cast<quake::RyOp>(op)) {
        if (!ry.getControls().empty())
          return op->emitError("controlled-Ry not yet supported");
        Value angleVal = ry.getParameters()[0];
        Value ref = ry.getTargets()[0];
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("Ry: target ref not found in refInfo");
        auto [veq, qi] = it->second;
        int64_t nQubits = veq.getType().cast<quake::VeqType>().getSize();
        Value sv = veqToSv[veq];
        Value new_sv;
        if (auto cstOp = angleVal.getDefiningOp<arith::ConstantOp>()) {
          // Constant path: fold trig at conversion time.
          double theta = cstOp.getValue().cast<FloatAttr>().getValueAsDouble();
          double c = std::cos(theta / 2.0), s = std::sin(theta / 2.0);
          // RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
          new_sv = buildApplyUnitary(b, loc, sv, nQubits, qi,
              (float)c,  0.f, (float)-s, 0.f,
              (float)s,  0.f, (float)c,  0.f);
        } else {
          // Dynamic path: runtime angle via math.cos / math.sin.
          auto f64Ty = b.getF64Type();
          auto f32Ty = Float32Type::get(b.getContext());
          Value half  = b.create<arith::ConstantOp>(loc, FloatAttr::get(f64Ty, 0.5));
          Value ht    = b.create<arith::MulFOp>(loc, asF64(b, loc, angleVal), half);
          auto [c, s] = buildCosSin(b, loc, ht);
          Value zero  = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, 0.f));
          Value ns    = b.create<arith::NegFOp>(loc, s);
          // RY(θ) = [[c, -s], [s, c]]
          new_sv = buildApplyUnitaryV(b, loc, sv, nQubits, qi,
              c, zero, ns, zero, s, zero, c, zero);
        }
        veqToSv[veq] = new_sv;
        finalSv = new_sv;
        toErase.push_back(op);

      } else if (auto rz = dyn_cast<quake::RzOp>(op)) {
        if (!rz.getControls().empty())
          return op->emitError("controlled-Rz not yet supported");
        Value angleVal = rz.getParameters()[0];
        Value ref = rz.getTargets()[0];
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("Rz: target ref not found in refInfo");
        auto [veq, qi] = it->second;
        int64_t nQubits = veq.getType().cast<quake::VeqType>().getSize();
        Value sv = veqToSv[veq];
        Value new_sv;
        if (auto cstOp = angleVal.getDefiningOp<arith::ConstantOp>()) {
          // Constant path: fold trig at conversion time.
          double lam = cstOp.getValue().cast<FloatAttr>().getValueAsDouble();
          double c = std::cos(lam / 2.0), s = std::sin(lam / 2.0);
          // RZ(λ) = [[exp(-iλ/2), 0], [0, exp(iλ/2)]]
          //       = [[cos-i·sin, 0], [0, cos+i·sin]]
          new_sv = buildApplyUnitary(b, loc, sv, nQubits, qi,
              (float)c, (float)-s,  0.f, 0.f,
              0.f,      0.f,        (float)c, (float)s);
        } else {
          // Dynamic path: runtime angle via math.cos / math.sin.
          auto f64Ty = b.getF64Type();
          auto f32Ty = Float32Type::get(b.getContext());
          Value half  = b.create<arith::ConstantOp>(loc, FloatAttr::get(f64Ty, 0.5));
          Value ht    = b.create<arith::MulFOp>(loc, asF64(b, loc, angleVal), half);
          auto [c, s] = buildCosSin(b, loc, ht);
          Value zero  = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, 0.f));
          Value ns    = b.create<arith::NegFOp>(loc, s);
          // RZ(λ) = [[c-i·s, 0], [0, c+i·s]]
          new_sv = buildApplyUnitaryV(b, loc, sv, nQubits, qi,
              c, ns, zero, zero, zero, zero, c, s);
        }
        veqToSv[veq] = new_sv;
        finalSv = new_sv;
        toErase.push_back(op);

      } else if (auto mz = dyn_cast<quake::MzOp>(op)) {
        // veq-level mz (sampling kernels): result is unused; statevector carries
        // the full probability distribution — erase and let Python sample.
        if (mz.getTargets().size() == 1 &&
            mz.getTargets()[0].getType().isa<quake::VeqType>()) {
          toErase.push_back(op);
        } else {
          // Single-qubit ref mz: compute deterministic bit from P(qubit=1) > 0.5.
          if (mz.getTargets().size() != 1)
            return op->emitError("Mz: only single-qubit measurement supported");
          Value ref = mz.getTargets()[0];
          auto it = refInfo.find(ref);
          if (it == refInfo.end())
            return op->emitError("Mz: target ref not found in refInfo");
          auto [veq, qi] = it->second;
          int64_t nQubits = veq.getType().cast<quake::VeqType>().getSize();
          Value sv = veqToSv[veq];
          Value bit = buildMeasureZBit(b, loc, sv, nQubits, qi);
          measToBit[mz.getMeasOut()] = bit;
          toErase.push_back(op);
        }

      } else if (auto discr = dyn_cast<quake::DiscriminateOp>(op)) {
        auto it = measToBit.find(discr.getMeasurement());
        if (it == measToBit.end())
          return op->emitError("Discriminate: measurement not found");
        auto intTy = discr.getResult().getType().dyn_cast<IntegerType>();
        if (!intTy || intTy.getWidth() != 1)
          return op->emitError("Discriminate: only i1 results supported");
        discr.getResult().replaceAllUsesWith(it->second);
        toErase.push_back(op);

      } else if (auto reset = dyn_cast<quake::ResetOp>(op)) {
        Value ref = reset.getTargets();
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("Reset: target ref not found in refInfo");
        auto [veq, qi] = it->second;
        int64_t nQubits = veq.getType().cast<quake::VeqType>().getSize();
        Value sv = veqToSv[veq];
        Value new_sv = buildApplyReset(b, loc, sv, nQubits, qi);
        veqToSv[veq] = new_sv;
        finalSv = new_sv;
        toErase.push_back(op);

      } else if (isa<quake::DeallocOp>(op)) {
        toErase.push_back(op);
      }
      // All other ops (func.return, etc.) pass through unchanged.
    }
  }

  // Erase quake ops (in reverse order to avoid use-before-def issues)
  for (Operation *op : llvm::reverse(toErase))
    op->erase();

  if (!finalSv)
    return fn->emitError("no statevector produced — no quake.alloca found");

  auto oldFnTy = fn.getFunctionType();

  // Build updated input list: original inputs + sv args added for each alloca.
  SmallVector<Type> newInputs(oldFnTy.getInputs().begin(),
                              oldFnTy.getInputs().end());
  newInputs.append(addedArgTypes.begin(), addedArgTypes.end());

  if (!oldFnTy.getResults().empty()) {
    fn.setFunctionType(FunctionType::get(ctx, newInputs, oldFnTy.getResults()));
    return success();
  }

  // Update void quantum functions to return the final statevector tensor.
  SmallVector<Type> newResults(oldFnTy.getResults().begin(),
                               oldFnTy.getResults().end());
  newResults.push_back(finalSv.getType());
  fn.setFunctionType(FunctionType::get(ctx, newInputs, newResults));

  // Update every return op to include the final statevector.
  fn.walk([&](func::ReturnOp ret) {
    OpBuilder rb(ret);
    SmallVector<Value> newOperands(ret.getOperands().begin(),
                                   ret.getOperands().end());
    newOperands.push_back(finalSv);
    rb.create<func::ReturnOp>(ret.getLoc(), newOperands);
    ret.erase();
  });

  return success();
}

// ---------------------------------------------------------------------------
// Pass driver
// ---------------------------------------------------------------------------

struct QuakeToStandard : impl::QuakeToStandardBase<QuakeToStandard> {
  using QuakeToStandardBase::QuakeToStandardBase;

  // Declare dialects produced by this pass so the pass manager loads them
  // before runOnOperation() is called.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    // Lower each function that contains quake ops.
    SmallVector<func::FuncOp> toProcess;
    module->walk([&](func::FuncOp fn) {
      bool hasQuake = false;
      fn->walk([&](Operation *op) {
        if (isa<quake::AllocaOp>(op)) hasQuake = true;
      });
      if (hasQuake) toProcess.push_back(fn);
    });

    for (func::FuncOp fn : toProcess) {
      if (failed(lowerQuantumFunction(fn, ctx))) {
        signalPassFailure();
        return;
      }
    }

    // Strip CUDA-Q runtime boilerplate that IREE cannot parse:
    //   - all llvm.func ops (typed-pointer signatures unknown to IREE)
    //   - declaration-only private func.func ops (runtime glue with no body)
    SmallVector<Operation *> toErase;
    module->walk([&](Operation *op) {
      if (isa<LLVM::LLVMFuncOp>(op)) {
        toErase.push_back(op);
      } else if (auto fn = dyn_cast<func::FuncOp>(op)) {
        if (fn->hasAttr("quake.cudaq_run") ||
            fn.getSymName().ends_with(".run.entry") ||
            (fn.isPrivate() && fn.empty()))
          toErase.push_back(fn);
      }
    });
    for (Operation *op : toErase)
      op->erase();

    // Remove cc.* and quake.* module-level attributes (CUDA-Q metadata that
    // references types IREE does not know about).
    SmallVector<StringAttr> attrsToRemove;
    for (NamedAttribute na : module->getAttrs()) {
      StringRef name = na.getName().getValue();
      if (name.starts_with("cc.") || name.starts_with("quake."))
        attrsToRemove.push_back(na.getName());
    }
    for (StringAttr name : attrsToRemove)
      module->removeAttr(name);
  }
};

} // namespace quake_to_standard
} // namespace mlir

namespace quake {
void registerQuakeToStandardPass() {
  mlir::quake_to_standard::registerQuakeToStandardPass();
}
}
