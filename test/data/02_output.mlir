module {
  func.func @main() {
    %0 = tensor.empty() : tensor<4xi1>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %0, %c0 : tensor<4xi1>
    return
  }
}

