module {
  func.func @init_from_complex(%arg0: !cc.ptr<complex<f64>>) {
    %0 = tensor.empty() : tensor<4xi1>
    return
  }
}

