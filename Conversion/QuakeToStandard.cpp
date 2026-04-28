#include "Conversion/QuakeToStandard.h"
#include "Dialect/Quake/QuakeDialect.h"
#include "Dialect/Quake/QuakeOps.h"
#include "Dialect/Quake/QuakeTypes.h"
#include "Dialect/CC/CCDialect.h"
#include "Dialect/CC/CCOps.h"
#include "Dialect/CC/CCTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

// Build |0...0> initial statevector: all zeros except sv[0] = 1.0
static Value buildInitStatevector(OpBuilder &b, Location loc, int64_t nQubits) {
  MLIRContext *ctx = b.getContext();
  auto ty = svTensorType(ctx, nQubits);
  int64_t nF32 = ty.getNumElements();

  auto f32Ty = Float32Type::get(ctx);
  SmallVector<Attribute> elems(nF32, FloatAttr::get(f32Ty, 0.0f));
  elems[0] = FloatAttr::get(f32Ty, 1.0f);
  return b.create<arith::ConstantOp>(loc, DenseElementsAttr::get(ty, elems));
}

// ---------------------------------------------------------------------------
// Gate builders — take current tensor sv, return updated tensor.
//
// Bit arithmetic is entirely in i64.  We convert i64 → index explicitly
// (arith.index_cast) before each tensor.extract / tensor.insert call.
// We use arith.addi (not arith.ori) for non-overlapping bit fields —
// IREE's VM lowering leaves arith.index_cast(arith.addi(i64)) as-is,
// but adds unresolvable builtin.unrealized_conversion_cast for
// arith.index_cast(arith.ori / arith.shli).
// ---------------------------------------------------------------------------

// Apply 2×2 unitary [[u00,u01],[u10,u11]] to qubit `qubitIdx` of `sv`.
static Value buildApplyUnitary(OpBuilder &b, Location loc, Value sv,
                               int64_t nQubits, int64_t qubitIdx,
                               float u00r, float u00i,
                               float u01r, float u01i,
                               float u10r, float u10i,
                               float u11r, float u11i) {
  MLIRContext *ctx = b.getContext();
  auto f32Ty = Float32Type::get(ctx);
  auto i64Ty = b.getI64Type();
  auto idxTy = b.getIndexType();

  int64_t nComplex = 1LL << nQubits;
  int64_t nPairs   = nComplex / 2;

  Value c0  = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1  = b.create<arith::ConstantIndexOp>(loc, 1);
  Value nP  = b.create<arith::ConstantIndexOp>(loc, nPairs);

  // Matrix constants (captured as Value in lambda — const-safe)
  Value U00r = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, u00r));
  Value U00i = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, u00i));
  Value U01r = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, u01r));
  Value U01i = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, u01i));
  Value U10r = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, u10r));
  Value U10i = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, u10i));
  Value U11r = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, u11r));
  Value U11i = b.create<arith::ConstantOp>(loc, FloatAttr::get(f32Ty, u11i));

  // i64 constants for bit arithmetic (captured as Value)
  Value qi_64    = b.create<arith::ConstantIntOp>(loc, qubitIdx, i64Ty);
  Value c1_64    = b.create<arith::ConstantIntOp>(loc, 1, i64Ty);
  Value stride   = b.create<arith::ShLIOp>(loc, c1_64, qi_64);   // 1 << qi
  Value lmask    = b.create<arith::SubIOp>(loc, stride, c1_64);  // stride - 1
  Value qp1      = b.create<arith::AddIOp>(loc, qi_64, c1_64);   // qi + 1

  // scf.for carries the statevector tensor as iter_arg
  auto loop = b.create<scf::ForOp>(
      loc, c0, nP, c1, ValueRange{sv},
      [U00r, U00i, U01r, U01i, U10r, U10i, U11r, U11i,
       qi_64, stride, lmask, qp1, c1_64, i64Ty, idxTy]
      (OpBuilder &body, Location l, Value j, ValueRange iters) {
        Value sv_cur = iters[0];

        // Compute complex pair indices k0 (bit-qi=0) and k1 = k0 | stride
        // using addi instead of ori (bits are non-overlapping)
        Value j64   = body.create<arith::IndexCastOp>(l, i64Ty, j);
        Value lower = body.create<arith::AndIOp>(l, j64, lmask);
        Value uHalf = body.create<arith::ShRUIOp>(l, j64, qi_64);
        Value upper = body.create<arith::ShLIOp>(l, uHalf, qp1);
        Value k0_64 = body.create<arith::AddIOp>(l, upper, lower); // addi
        Value k1_64 = body.create<arith::AddIOp>(l, k0_64, stride); // addi

        // f32 indices: 2*k, 2*k+1
        Value k0r_64 = body.create<arith::ShLIOp>(l, k0_64, c1_64);
        Value k0i_64 = body.create<arith::AddIOp>(l, k0r_64, c1_64);
        Value k1r_64 = body.create<arith::ShLIOp>(l, k1_64, c1_64);
        Value k1i_64 = body.create<arith::AddIOp>(l, k1r_64, c1_64);
        Value k0r = body.create<arith::IndexCastOp>(l, idxTy, k0r_64);
        Value k0i = body.create<arith::IndexCastOp>(l, idxTy, k0i_64);
        Value k1r = body.create<arith::IndexCastOp>(l, idxTy, k1r_64);
        Value k1i = body.create<arith::IndexCastOp>(l, idxTy, k1i_64);

        // Load amplitudes a0 = sv[k0], a1 = sv[k1]
        Value a0r = body.create<tensor::ExtractOp>(l, sv_cur, k0r);
        Value a0i = body.create<tensor::ExtractOp>(l, sv_cur, k0i);
        Value a1r = body.create<tensor::ExtractOp>(l, sv_cur, k1r);
        Value a1i = body.create<tensor::ExtractOp>(l, sv_cur, k1i);

        // na0 = U00 * a0 + U01 * a1  (complex multiply-add)
        // na0r = U00r*a0r - U00i*a0i + U01r*a1r - U01i*a1i
        Value t0 = body.create<arith::MulFOp>(l, U00r, a0r);
        Value t1 = body.create<arith::MulFOp>(l, U00i, a0i);
        Value t2 = body.create<arith::MulFOp>(l, U01r, a1r);
        Value t3 = body.create<arith::MulFOp>(l, U01i, a1i);
        Value na0r = body.create<arith::SubFOp>(l,
            body.create<arith::AddFOp>(l, t0, t2),
            body.create<arith::AddFOp>(l, t1, t3));
        // na0i = U00r*a0i + U00i*a0r + U01r*a1i + U01i*a1r
        Value t4 = body.create<arith::MulFOp>(l, U00r, a0i);
        Value t5 = body.create<arith::MulFOp>(l, U00i, a0r);
        Value t6 = body.create<arith::MulFOp>(l, U01r, a1i);
        Value t7 = body.create<arith::MulFOp>(l, U01i, a1r);
        Value na0i = body.create<arith::AddFOp>(l,
            body.create<arith::AddFOp>(l, t4, t5),
            body.create<arith::AddFOp>(l, t6, t7));

        // na1 = U10 * a0 + U11 * a1
        Value t8  = body.create<arith::MulFOp>(l, U10r, a0r);
        Value t9  = body.create<arith::MulFOp>(l, U10i, a0i);
        Value t10 = body.create<arith::MulFOp>(l, U11r, a1r);
        Value t11 = body.create<arith::MulFOp>(l, U11i, a1i);
        Value na1r = body.create<arith::SubFOp>(l,
            body.create<arith::AddFOp>(l, t8, t10),
            body.create<arith::AddFOp>(l, t9, t11));
        Value t12 = body.create<arith::MulFOp>(l, U10r, a0i);
        Value t13 = body.create<arith::MulFOp>(l, U10i, a0r);
        Value t14 = body.create<arith::MulFOp>(l, U11r, a1i);
        Value t15 = body.create<arith::MulFOp>(l, U11i, a1r);
        Value na1i = body.create<arith::AddFOp>(l,
            body.create<arith::AddFOp>(l, t12, t13),
            body.create<arith::AddFOp>(l, t14, t15));

        // Store updated amplitudes back into tensor (functional update)
        Value s0 = body.create<tensor::InsertOp>(l, na0r, sv_cur, k0r);
        Value s1 = body.create<tensor::InsertOp>(l, na0i, s0, k0i);
        Value s2 = body.create<tensor::InsertOp>(l, na1r, s1, k1r);
        Value s3 = body.create<tensor::InsertOp>(l, na1i, s2, k1i);
        body.create<scf::YieldOp>(l, s3);
      });
  return loop.getResult(0);
}

// Apply CNOT: flip qubit `tgtIdx` when qubit `ctrlIdx` is |1>.
static Value buildApplyCNOT(OpBuilder &b, Location loc, Value sv,
                             int64_t nQubits, int64_t ctrlIdx, int64_t tgtIdx) {
  auto i64Ty = b.getI64Type();
  auto idxTy = b.getIndexType();

  int64_t nComplex = 1LL << nQubits;
  int64_t nPairs   = nComplex / 2;

  Value c0  = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1  = b.create<arith::ConstantIndexOp>(loc, 1);
  Value nP  = b.create<arith::ConstantIndexOp>(loc, nPairs);

  Value ti      = b.create<arith::ConstantIntOp>(loc, tgtIdx, i64Ty);
  Value ci      = b.create<arith::ConstantIntOp>(loc, ctrlIdx, i64Ty);
  Value c1_64   = b.create<arith::ConstantIntOp>(loc, 1, i64Ty);
  Value c0_64   = b.create<arith::ConstantIntOp>(loc, 0, i64Ty);
  Value stTgt   = b.create<arith::ShLIOp>(loc, c1_64, ti);   // 1 << tgt
  Value stCtrl  = b.create<arith::ShLIOp>(loc, c1_64, ci);   // 1 << ctrl
  Value lmask   = b.create<arith::SubIOp>(loc, stTgt, c1_64); // stTgt - 1
  Value tp1     = b.create<arith::AddIOp>(loc, ti, c1_64);   // tgt + 1

  auto loop = b.create<scf::ForOp>(
      loc, c0, nP, c1, ValueRange{sv},
      [ti, stTgt, stCtrl, lmask, tp1, c1_64, c0_64, i64Ty, idxTy]
      (OpBuilder &body, Location l, Value j, ValueRange iters) {
        Value sv_cur = iters[0];

        Value j64   = body.create<arith::IndexCastOp>(l, i64Ty, j);
        Value lower = body.create<arith::AndIOp>(l, j64, lmask);
        Value uHalf = body.create<arith::ShRUIOp>(l, j64, ti);
        Value upper = body.create<arith::ShLIOp>(l, uHalf, tp1);
        Value k0_64 = body.create<arith::AddIOp>(l, upper, lower);
        Value k1_64 = body.create<arith::AddIOp>(l, k0_64, stTgt);

        // f32 indices
        Value k0r_64 = body.create<arith::ShLIOp>(l, k0_64, c1_64);
        Value k0i_64 = body.create<arith::AddIOp>(l, k0r_64, c1_64);
        Value k1r_64 = body.create<arith::ShLIOp>(l, k1_64, c1_64);
        Value k1i_64 = body.create<arith::AddIOp>(l, k1r_64, c1_64);
        Value k0r = body.create<arith::IndexCastOp>(l, idxTy, k0r_64);
        Value k0i = body.create<arith::IndexCastOp>(l, idxTy, k0i_64);
        Value k1r = body.create<arith::IndexCastOp>(l, idxTy, k1r_64);
        Value k1i = body.create<arith::IndexCastOp>(l, idxTy, k1i_64);

        // Swap amplitudes k0 ↔ k1 only when ctrl bit of k0 is set.
        Value ctrlBit = body.create<arith::AndIOp>(l, k0_64, stCtrl);
        Value isCtrl  = body.create<arith::CmpIOp>(
            l, arith::CmpIPredicate::ne, ctrlBit, c0_64);

        // scf.if with result carries the (possibly updated) tensor.
        // The builder variant (cond, thenFn, elseFn) infers result types
        // from the scf.yield inside each region.
        Value updated = body.create<scf::IfOp>(
            l, isCtrl,
            [sv_cur, k0r, k0i, k1r, k1i](OpBuilder &tb, Location tl) {
              Value a0r = tb.create<tensor::ExtractOp>(tl, sv_cur, k0r);
              Value a0i = tb.create<tensor::ExtractOp>(tl, sv_cur, k0i);
              Value a1r = tb.create<tensor::ExtractOp>(tl, sv_cur, k1r);
              Value a1i = tb.create<tensor::ExtractOp>(tl, sv_cur, k1i);
              Value s0  = tb.create<tensor::InsertOp>(tl, a1r, sv_cur, k0r);
              Value s1  = tb.create<tensor::InsertOp>(tl, a1i, s0, k0i);
              Value s2  = tb.create<tensor::InsertOp>(tl, a0r, s1, k1r);
              Value s3  = tb.create<tensor::InsertOp>(tl, a0i, s2, k1i);
              tb.create<scf::YieldOp>(tl, s3);
            },
            [sv_cur](OpBuilder &eb, Location el) {
              eb.create<scf::YieldOp>(el, sv_cur);
            }).getResult(0);

        body.create<scf::YieldOp>(l, updated);
      });
  return loop.getResult(0);
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

  SmallVector<Operation *> toErase;
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
        Value sv = buildInitStatevector(b, loc, nQubits);
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
        if (!z.getControls().empty())
          return op->emitError("controlled-Z not yet supported");
        Value ref = z.getTargets()[0];
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("Z gate: ref not found in refInfo");
        auto [veq, qi] = it->second;
        int64_t nQubits = veq.getType().cast<quake::VeqType>().getSize();
        Value sv = veqToSv[veq];
        // Pauli-Z: [[1, 0], [0, -1]]
        Value new_sv = buildApplyUnitary(b, loc, sv, nQubits, qi,
            1.f, 0.f,  0.f, 0.f,
            0.f, 0.f, -1.f, 0.f);
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

      } else if (auto rx = dyn_cast<quake::RxOp>(op)) {
        if (!rx.getControls().empty())
          return op->emitError("controlled-Rx not yet supported");
        Value angleVal = rx.getParameters()[0];
        auto cstOp = angleVal.getDefiningOp<arith::ConstantOp>();
        if (!cstOp)
          return op->emitError("Rx: only constant angles supported");
        double theta = cstOp.getValue().cast<FloatAttr>().getValueAsDouble();
        double c = std::cos(theta / 2.0), s = std::sin(theta / 2.0);
        Value ref = rx.getTargets()[0];
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("Rx: target ref not found in refInfo");
        auto [veq, qi] = it->second;
        int64_t nQubits = veq.getType().cast<quake::VeqType>().getSize();
        Value sv = veqToSv[veq];
        // RX(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
        Value new_sv = buildApplyUnitary(b, loc, sv, nQubits, qi,
            (float)c, 0.f,       0.f, (float)-s,
            0.f,      (float)-s, (float)c, 0.f);
        veqToSv[veq] = new_sv;
        finalSv = new_sv;
        toErase.push_back(op);

      } else if (auto ry = dyn_cast<quake::RyOp>(op)) {
        if (!ry.getControls().empty())
          return op->emitError("controlled-Ry not yet supported");
        Value angleVal = ry.getParameters()[0];
        auto cstOp = angleVal.getDefiningOp<arith::ConstantOp>();
        if (!cstOp)
          return op->emitError("Ry: only constant angles supported");
        double theta = cstOp.getValue().cast<FloatAttr>().getValueAsDouble();
        double c = std::cos(theta / 2.0), s = std::sin(theta / 2.0);
        Value ref = ry.getTargets()[0];
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("Ry: target ref not found in refInfo");
        auto [veq, qi] = it->second;
        int64_t nQubits = veq.getType().cast<quake::VeqType>().getSize();
        Value sv = veqToSv[veq];
        // RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
        Value new_sv = buildApplyUnitary(b, loc, sv, nQubits, qi,
            (float)c,  0.f, (float)-s, 0.f,
            (float)s,  0.f, (float)c,  0.f);
        veqToSv[veq] = new_sv;
        finalSv = new_sv;
        toErase.push_back(op);

      } else if (auto rz = dyn_cast<quake::RzOp>(op)) {
        if (!rz.getControls().empty())
          return op->emitError("controlled-Rz not yet supported");
        Value angleVal = rz.getParameters()[0];
        auto cstOp = angleVal.getDefiningOp<arith::ConstantOp>();
        if (!cstOp)
          return op->emitError("Rz: only constant angles supported");
        double lam = cstOp.getValue().cast<FloatAttr>().getValueAsDouble();
        double c = std::cos(lam / 2.0), s = std::sin(lam / 2.0);
        Value ref = rz.getTargets()[0];
        auto it = refInfo.find(ref);
        if (it == refInfo.end())
          return op->emitError("Rz: target ref not found in refInfo");
        auto [veq, qi] = it->second;
        int64_t nQubits = veq.getType().cast<quake::VeqType>().getSize();
        Value sv = veqToSv[veq];
        // RZ(λ) = [[exp(-iλ/2), 0], [0, exp(iλ/2)]]
        //       = [[cos-i·sin, 0], [0, cos+i·sin]]
        Value new_sv = buildApplyUnitary(b, loc, sv, nQubits, qi,
            (float)c, (float)-s,  0.f, 0.f,
            0.f,      0.f,        (float)c, (float)s);
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

  // Update the function to return the final statevector tensor.
  auto oldFnTy = fn.getFunctionType();
  SmallVector<Type> newResults(oldFnTy.getResults().begin(),
                               oldFnTy.getResults().end());
  newResults.push_back(finalSv.getType());
  fn.setFunctionType(FunctionType::get(ctx, oldFnTy.getInputs(), newResults));

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
        if (fn.isPrivate() && fn.empty())
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
