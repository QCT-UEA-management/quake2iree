#include "Conversion/QuakeToStandard.h"
#include "Dialect/Quake/QuakeDialect.h"
#include "Dialect/Quake/QuakeOps.h"
#include "Dialect/Quake/QuakeTypes.h"

#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace quake_to_standard {

#define GEN_PASS_DEF_QUAKETOSTANDARD
#include "Conversion/QuakeToStandard.h.inc"

/// Type converter for Quake types → Tensor types
class QuakeToStandardTypeConverter : public TypeConverter {
public:
  QuakeToStandardTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });

    addConversion([ctx](quake::RefType) -> Type {
      return RankedTensorType::get({}, IntegerType::get(ctx, 1));
    });

    addConversion([ctx](quake::VeqType t) -> Type {
      int64_t size = t.hasSpecifiedSize() ? t.getSize() : ShapedType::kDynamic;
      return RankedTensorType::get({size}, IntegerType::get(ctx, 1));
    });
  }
};

/// Pattern to convert `quake.alloca` → `tensor.empty`
struct ConvertAlloca : public OpConversionPattern<quake::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quake::AllocaOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type origType = op.getType();  // !quake.veq<?>, !quake.veq<2>, etc.
    Type convertedType = getTypeConverter()->convertType(origType);
    if (!convertedType)
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");

    auto tensorTy = llvm::dyn_cast<RankedTensorType>(convertedType);
    if (!tensorTy)
      return rewriter.notifyMatchFailure(op, "Expected RankedTensorType");

    // Static shape case
    if (tensorTy.hasStaticShape()) {
      Value empty = rewriter.create<tensor::EmptyOp>(
          loc, tensorTy.getShape(), tensorTy.getElementType());
      rewriter.replaceOp(op, empty);
      return success();
    }

    // Dynamic shape
    Value sizeVal = op.getSize();

    // If no explicit size operand, try extracting from the type
    if (!sizeVal) {
      if (auto veqTy = llvm::dyn_cast<quake::VeqType>(origType)) {
        if (veqTy.hasSpecifiedSize()) {
          int64_t size = veqTy.getSize();
          sizeVal = rewriter.create<arith::ConstantIndexOp>(loc, size);
        } else {
          return rewriter.notifyMatchFailure(op, "Dynamic veq with no size operand");
        }
      }
    }

    Value castSize = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), sizeVal);
    SmallVector<Value> dynamicDims{castSize};

    Value empty = rewriter.create<tensor::EmptyOp>(
        loc, tensorTy.getShape(), tensorTy.getElementType(), dynamicDims);
    rewriter.replaceOp(op, empty);
    return success();
  }
};


/// Pattern to convert `quake.veq_size` → `tensor.dim`.
struct ConvertVeqSize : public OpConversionPattern<quake::VeqSizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quake::VeqSizeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value veq = adaptor.getVeq();
    auto tensorTy = llvm::dyn_cast<RankedTensorType>(veq.getType());
    if (!tensorTy || tensorTy.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1D tensor for veq");

    Value dim = rewriter.create<tensor::DimOp>(loc, veq, 0);
    rewriter.replaceOp(op, dim);
    return success();
  }
};


/// Pattern to convert `quake.init_state` (InitializeStateOp)
/// → `func.call @__quake_init_state_f32/_f64`.
///
/// Semantics:
/// - Takes a vector of qubits (`!quake.veq`) and a state tensor (`tensor<?xcomplex<f32/f64>>`)
/// - Calls a runtime helper to initialize the qubits with the provided amplitudes.
/// - Returns the same qubit tensor as the initialized result (RAII semantics).
struct ConvertInitState : public OpConversionPattern<quake::InitializeStateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quake::InitializeStateOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Operands after type conversion
    Value targets = adaptor.getTargets(); // tensor<?xi1> or tensor<Nxi1>
    Value state   = adaptor.getState();   // tensor<?xcomplex<fx>>

    // Verify operand types.
    auto qT = dyn_cast<RankedTensorType>(targets.getType());
    auto sT = dyn_cast<RankedTensorType>(state.getType());
    if (!qT || qT.getRank() != 1 || !qT.getElementType().isInteger(1))
      return rewriter.notifyMatchFailure(op, "expected 1D tensor<i1> for qubits");
    if (!sT || sT.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1D tensor for state");

    // Validate complex element type.
    Type elemTy = sT.getElementType();
    auto complexTy = dyn_cast<ComplexType>(elemTy);
    if (!complexTy)
      return rewriter.notifyMatchFailure(op, "state must have complex element type");

    Type floatTy = complexTy.getElementType();
    bool isF32 = floatTy.isF32();
    bool isF64 = floatTy.isF64();
    if (!isF32 && !isF64)
      return rewriter.notifyMatchFailure(op, "complex element must be f32 or f64");

    // Ensure runtime function declaration exists.
    MLIRContext *ctx = rewriter.getContext();
    auto funcTy = FunctionType::get(ctx, {qT, sT}, {});
    StringRef calleeName = isF32 ? "__quake_init_state_f32" : "__quake_init_state_f64";

    // Insert declaration if missing.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    SymbolTable symTab(module);
    if (!symTab.lookup(calleeName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto decl = rewriter.create<func::FuncOp>(loc, calleeName, funcTy);
      decl.setPrivate(); // internal linkage
    }

    // Emit runtime call.
    rewriter.create<func::CallOp>(
        loc, calleeName, TypeRange{}, ValueRange{targets, state});

    // Replace the original op with the same (initialized) qubit tensor.
    // Quake semantics: init_state returns a new !quake.veq, but the resource is reused.
    rewriter.replaceOp(op, targets);

    return success();
  }
};



/// Conversion pass driver
struct QuakeToStandard : impl::QuakeToStandardBase<QuakeToStandard> {
  using QuakeToStandardBase::QuakeToStandardBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *module = getOperation();

    QuakeToStandardTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);
    patterns.add<ConvertAlloca>(typeConverter, context);
    patterns.add<ConvertVeqSize>(typeConverter, context);
    patterns.add<ConvertInitState>(typeConverter, context);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<complex::ComplexDialect>();
    target.addIllegalDialect<quake::QuakeDialect>();

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter.isLegal(&op.getBody());
        });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace quake_to_standard
} // namespace mlir

namespace quake {
void registerQuakeToStandardPass() {
  mlir::quake_to_standard::registerQuakeToStandardPass();
}
}
