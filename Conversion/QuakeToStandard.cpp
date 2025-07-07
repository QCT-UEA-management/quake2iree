#include "Conversion/QuakeToStandard.h"
#include "Dialect/Quake/QuakeDialect.h"  // Added: fixes missing QuakeDialect
#include "Dialect/Quake/QuakeOps.h"
#include "Dialect/Quake/QuakeTypes.h"

#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"



namespace mlir{
namespace quake_to_standard {  

#define GEN_PASS_DEF_QUAKETOSTANDARD
#include "Conversion/QuakeToStandard.h.inc"

/// Type converter for Quake types.
class QuakeToStandardTypeConverter : public TypeConverter {
public:
  QuakeToStandardTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });

    addConversion([ctx](quake::RefType) -> Type {
      return MemRefType::get({}, IntegerType::get(ctx, 1));
    });

    addConversion([ctx](quake::VeqType type) -> Type {
      int64_t size = type.getSize();
      return MemRefType::get({size >= 0 ? size : ShapedType::kDynamic},
                             IntegerType::get(ctx, 1));
    });
  }
};

/// Pattern to convert quake.alloca → memref.alloc
struct ConvertAlloca : public OpConversionPattern<quake::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quake::AllocaOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type oldType = op.getResult().getType();
    Type converted = getTypeConverter()->convertType(oldType);
    if (!converted)
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");

    auto memrefTy = llvm::dyn_cast<MemRefType>(converted);
    if (!memrefTy)
      return rewriter.notifyMatchFailure(op, "Expected MemRefType after conversion");

    // Handle dynamic allocation (for veq<?>)
    if (!memrefTy.hasStaticShape()) {  // Fixed incorrect method
      if (!op.getSize())
        return rewriter.notifyMatchFailure(op, "Missing dynamic size operand");
      Value alloc = rewriter.create<mlir::memref::AllocOp>(
          loc, memrefTy, ValueRange{op.getSize()});
      rewriter.replaceOp(op, alloc);
      return success();
    }

    // Static-size or ref case
    Value alloc = rewriter.create<mlir::memref::AllocOp>(loc, memrefTy);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

/// Main conversion pass driver
struct QuakeToStandard : impl::QuakeToStandardBase<QuakeToStandard> {
  using QuakeToStandardBase::QuakeToStandardBase;  // Fixed base alias

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
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

} 
}

