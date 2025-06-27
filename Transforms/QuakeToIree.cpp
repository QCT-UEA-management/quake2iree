#include "mlir/Conversion/PatternRewriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// Include your Quake dialect headers
#include "Dialect/Quake/QuakeOps.h"

using namespace mlir;

namespace {

struct QuakeZOpLowering : public OpConversionPattern<quake::ZOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quake::ZOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    // Erase Z since Z|0> = |0>, so no classical change
    rewriter.eraseOp(op);
    return success();
  }
};

struct QuakeMzOpLowering : public OpConversionPattern<quake::MzOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quake::MzOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Create tensor<1xi1> with `false` value
    auto resultType = RankedTensorType::get({1}, rewriter.getI1Type());

    Value falseVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
    Value zeroIndex = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    Value tensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    Value filled = rewriter.create<tensor::InsertOp>(loc, falseVal, tensor, ValueRange{zeroIndex});

    if (!op.getResults().empty())
      rewriter.replaceOp(op, filled);
    else
      rewriter.eraseOp(op);

    return success();
  }
};

struct QuakeAllocaLowering : public OpConversionPattern<quake::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quake::AllocaOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    // Replace with dummy tensor just to hold space
    auto loc = op.getLoc();
    Value size = operands[0];

    // Create tensor<?xi1> as placeholder
    auto i1Ty = rewriter.getI1Type();
    auto shape = ShapedType::get({ShapedType::kDynamic}, i1Ty);
    Value tensor = rewriter.create<tensor::EmptyOp>(loc, ValueRange{size}, i1Ty);
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

struct QuakeExtractRefLowering : public OpConversionPattern<quake::ExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quake::ExtractRefOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    // Drop to a sub-element (placeholder)
    Value tensor = operands[0];
    Value index = operands[1];
    // Simply forward tensor index as-is
    rewriter.replaceOp(op, {index});
    return success();
  }
};

struct ConvertQuakeToIREEPass
    : public PassWrapper<ConvertQuakeToIREEPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();

    ConversionTarget target(ctx);
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalOp<ModuleOp>();

    // Mark Quake dialect ops illegal
    target.addIllegalDialect<quake::QuakeDialect>();

    RewritePatternSet patterns(&ctx);
    patterns.add<QuakeZOpLowering, QuakeMzOpLowering,
                 QuakeAllocaLowering, QuakeExtractRefLowering>(ctx);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertQuakeToIREEPass() {
  return std::make_unique<ConvertQuakeToIREEPass>();
}
