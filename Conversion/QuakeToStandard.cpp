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
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace quake_to_standard {

#define GEN_PASS_DEF_QUAKETOSTANDARD
#include "Conversion/QuakeToStandard.h.inc"

/// Type converter for Quake types -> Tensor types
class QuakeToStandardTypeConverter : public TypeConverter {
public:
  QuakeToStandardTypeConverter(MLIRContext *ctx) {
    // Identity conversion for unknown types.
    addConversion([](Type type) { return type; });

    // Convert quake.ref -> tensor<i1>
    addConversion([ctx](quake::RefType) -> Type {
      return RankedTensorType::get({}, IntegerType::get(ctx, 1));
    });

    // Convert quake.veq<?> -> tensor<?xi1>
    addConversion([ctx](quake::VeqType) -> Type {
      // We assume dynamic shape to avoid needing getSize()
      return RankedTensorType::get({ShapedType::kDynamic},
                                   IntegerType::get(ctx, 1));
    });
  }
};

/// Pattern to convert quake.alloca → tensor.empty
struct ConvertAlloca : public OpConversionPattern<quake::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quake::AllocaOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type origType = op.getResult().getType();
    Type convertedType = getTypeConverter()->convertType(origType);
    if (!convertedType)
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");

    auto tensorTy = llvm::dyn_cast<RankedTensorType>(convertedType);
    if (!tensorTy)
      return rewriter.notifyMatchFailure(op, "Expected RankedTensorType");

    // Static shape: just create a tensor.empty
    if (tensorTy.hasStaticShape()) {
      Value empty = rewriter.create<tensor::EmptyOp>(
          loc, tensorTy.getShape(), tensorTy.getElementType());
      rewriter.replaceOp(op, empty);
      return success();
    }

    // Dynamic shape case
    if (op.getNumOperands() < 1)
      return rewriter.notifyMatchFailure(op, "Missing dynamic size operand");

    Value sizeVal = op.getOperand(0);
    Value castSize = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), sizeVal);
    SmallVector<Value> dynamicDims{castSize};

    Value empty = rewriter.create<tensor::EmptyOp>(
        loc, tensorTy.getShape(), tensorTy.getElementType(), dynamicDims);
    rewriter.replaceOp(op, empty);
    return success();
  }
};

/// Main conversion pass driver
struct QuakeToStandard : impl::QuakeToStandardBase<QuakeToStandard> {
  using QuakeToStandardBase::QuakeToStandardBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addIllegalDialect<quake::QuakeDialect>();

    QuakeToStandardTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);
    patterns.add<ConvertAlloca>(typeConverter, context);

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
