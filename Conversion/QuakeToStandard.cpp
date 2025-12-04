#include "Conversion/QuakeToStandard.h"
#include "Dialect/Quake/QuakeDialect.h"
#include "Dialect/Quake/QuakeOps.h"
#include "Dialect/Quake/QuakeTypes.h"
#include "Dialect/CC/CCDialect.h"
#include "Dialect/CC/CCOps.h"
#include "Dialect/CC/CCTypes.h"

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
      // Represent a single quantum reference as a length-1 tensor<i1>
      return RankedTensorType::get({1}, IntegerType::get(ctx, 1));
    });

    addConversion([ctx](quake::VeqType t) -> Type {
      int64_t size = t.hasSpecifiedSize() ? t.getSize() : ShapedType::kDynamic;
      return RankedTensorType::get({size}, IntegerType::get(ctx, 1));
    });
  }
};


/// Pattern to convert `quake.dealloc` → erased (no-op in Standard MLIR)
struct ConvertDealloc : public OpConversionPattern<quake::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quake::DeallocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // In Quake, dealloc is used to release quantum memory (qubits),
    // but in Standard MLIR, tensors are value types, so there’s no explicit free.
    // We simply remove the operation.
    rewriter.eraseOp(op);
    return success();
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


struct ConvertInitState : public OpConversionPattern<quake::InitializeStateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quake::InitializeStateOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value qvTensor = adaptor.getTargets();
    Value statePtr = adaptor.getState(); // Ignored

    auto qvTy = qvTensor.getType().dyn_cast<RankedTensorType>();
    if (!qvTy || qvTy.getRank() != 1 || !qvTy.getElementType().isInteger(1))
      return rewriter.notifyMatchFailure(op, "expected rank-1 tensor<i1>");

    // If you don't want any amplitude materialization, just "RAII-return" the veq:
    rewriter.replaceOp(op, qvTensor);
    return success();
  }
};


/// Pattern to convert `quake.concat` → sequence of `tensor.insert_slice`
/// operations. LLVM 16–compatible fallback (no `tensor.concat`).
struct ConvertConcat : public OpConversionPattern<quake::ConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quake::ConcatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ValueRange inputs = adaptor.getQbits();
    if (inputs.empty())
      return rewriter.notifyMatchFailure(op, "no inputs to concat");

    // Normalize all inputs to rank-1 tensor<i1>.
    SmallVector<Value> oneDInputs;  // we need to concatenate tensors, so let's include them in oneDInputs. oneDInputs acts as a **staging area** for normalized tensors
    int64_t totalLen = 0;
    bool allStatic = true;

    for (Value v : inputs) {
      auto rtt = dyn_cast<RankedTensorType>(v.getType());

      //Check type and element type
      if (!rtt || !rtt.getElementType().isInteger(1))
        return rewriter.notifyMatchFailure(op, "expected tensor<i1> or tensor<?xi1>");

      // Handle rank-0 tensors (scalar case)
      if (rtt.getRank() == 0) {
        // Promote tensor<i1> → tensor<1xi1>
        auto scalar = rewriter.create<tensor::ExtractOp>(loc, v).getResult(); // i1
        auto oneTy  = RankedTensorType::get({1}, rewriter.getI1Type()); // {1} means the shape is [1], so oneTy is tensor<1xi1>.
        Value v1    = rewriter.create<tensor::SplatOp>(loc, oneTy, scalar).getResult(); // Create a tensor of shape {1} filled with that scalar
        oneDInputs.push_back(v1);  // Adds the newly created rank-1 tensor to the list of normalized inputs
        totalLen += 1; // Updates the total static length of the concatenated tensor
        continue;
      }

      // Handle rank-1 tensors
      if (rtt.getRank() == 1) {
        oneDInputs.push_back(v);
        if (rtt.isDynamicDim(0))
          allStatic = false;
        else
          totalLen += rtt.getDimSize(0);
        continue;
      }

      return rewriter.notifyMatchFailure(op, "unsupported tensor rank for concat");
    }

    // Now elemTy will be used as the element type for the result tensor that we are 
    // about to create. Below we create the destination tensor (using tensor::EmptyOp), 
    // we must specify: The shape (static or dynamic) and the element type.
    Type elemTy = rewriter.getI1Type();  

    // Does the Quake result type specify a fixed size?
    // If not, treat the result as dynamic.
    auto quakeResultTy = op.getType().dyn_cast<quake::VeqType>(); // retrieves the result type of the quake.concat operation
    bool resultDynamic = !quakeResultTy || !quakeResultTy.hasSpecifiedSize(); // determines whether the result length is dynamic

    // Create the destination tensor (either fully static or 1D dynamic).
    Value result;
    if (!resultDynamic && allStatic) {
      result = rewriter.create<tensor::EmptyOp>(loc,
                                                llvm::ArrayRef<int64_t>{totalLen},
                                                elemTy
                                                ).getResult();
    } else {
      // Compute total length dynamically.
      Value totalSize = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
      for (Value v : oneDInputs) {
        auto ty = v.getType().cast<RankedTensorType>();
        Value lenVal;
        if (ty.isDynamicDim(0)) {
          auto dimOp = rewriter.create<tensor::DimOp>(loc, v, 0);
          lenVal = dimOp.getResult();
        } else {
          auto cst = rewriter.create<arith::ConstantIndexOp>(loc, ty.getDimSize(0));
          lenVal = cst.getResult();
        }
        auto add = rewriter.create<arith::AddIOp>(loc, totalSize, lenVal);
        totalSize = add.getResult();
      }
      SmallVector<Value> dynSizes{totalSize};
      result = rewriter
                 .create<tensor::EmptyOp>(loc,
                                          llvm::ArrayRef<int64_t>{ShapedType::kDynamic},
                                          elemTy, dynSizes)
                 .getResult();
    }

    // Insert each slice in order.
    Value offset = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
    for (Value v : oneDInputs) {
      auto ty = v.getType().cast<RankedTensorType>();

      // Build `sizes` as OpFoldResult: Attr for static, Value for dynamic.
      OpFoldResult lenOfr;
      Value        lenValForAdd; // for offset update

      if (ty.isDynamicDim(0)) {
        auto dimOp = rewriter.create<tensor::DimOp>(loc, v, 0);
        lenValForAdd = dimOp.getResult();
        lenOfr = lenValForAdd; // dynamic size via SSA
      } else {
        int64_t n = ty.getDimSize(0);
        lenOfr = rewriter.getIndexAttr(n); // static size as attribute
        lenValForAdd = rewriter.create<arith::ConstantIndexOp>(loc, n).getResult();
      }

      SmallVector<OpFoldResult> offsets{offset};                   // dynamic offset ok
      SmallVector<OpFoldResult> sizes{lenOfr};                     // <-- key: attr if static
      SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)}; // static stride=1

      result = rewriter.create<tensor::InsertSliceOp>(loc, v, result,
                                                offsets, sizes, strides)
                 .getResult();

      // offset += len
      auto add = rewriter.create<arith::AddIOp>(loc, offset, lenValForAdd);
      offset = add.getResult();
    }

    rewriter.replaceOp(op, result);
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



struct ConvertExtractRef : public OpConversionPattern<quake::ExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quake::ExtractRefOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc(); // Captures the operation’s source location
    Value veqTensor = adaptor.getVeq(); // normalized tensor<?xi1>
    Value indexVal = adaptor.getIndex(); // dynamic index if provided

    auto veqTy = veqTensor.getType().dyn_cast<RankedTensorType>();
    if (!veqTy || veqTy.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "expected rank-1 tensor for veq");

    // Result type: tensor<1xi1>
    auto elemTy = rewriter.getI1Type();
    auto oneTy = RankedTensorType::get({1}, elemTy);

    // Compute offset: dynamic if indexVal exists, else static from rawIndex
    OpFoldResult offset;
    if (indexVal) {
      offset = indexVal; // dynamic offset
    } else {
      offset = rewriter.getIndexAttr(op.getRawIndex()); // static offset
    }

    SmallVector<OpFoldResult> offsets{offset};
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(1)}; // size = 1
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)}; // stride = 1

    Value slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, oneTy, veqTensor, offsets, sizes, strides);

    rewriter.replaceOp(op, slice);
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
    patterns.add<ConvertDealloc>(typeConverter, context);
    patterns.add<ConvertAlloca>(typeConverter, context);
    patterns.add<ConvertInitState>(typeConverter, context);    
    patterns.add<ConvertVeqSize>(typeConverter, context);
    patterns.add<ConvertConcat>(typeConverter, context);
    patterns.add<ConvertExtractRef>(typeConverter, context);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<complex::ComplexDialect>();
    target.addLegalDialect<cudaq::cc::CCDialect>();
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
