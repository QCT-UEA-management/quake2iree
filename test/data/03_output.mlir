module {
  func.func @dealloc_example() {
    %0 = tensor.empty() : tensor<4xi1>
    return
  }
}

