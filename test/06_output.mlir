module {
  func.func @test_initialize_state_static(%arg0: !cc.ptr<complex<f64>>) {
    %0 = tensor.empty() : tensor<4xi1>
    %c4 = arith.constant 4 : index
    %1 = arith.index_cast %c4 : index to i64
    %c1_i64 = arith.constant 1 : i64
    %2 = arith.shli %c1_i64, %1 : i64
    %3 = arith.index_cast %2 : i64 to index
    %4 = call @__quake_load_state(%arg0, %3) : (!cc.ptr<complex<f64>>, index) -> tensor<?xcomplex<f64>>
    %5 = call @__quake_init_state_from_tensor(%0, %4) : (tensor<4xi1>, tensor<?xcomplex<f64>>) -> tensor<4xi1>
    return
  }
  func.func private @__quake_load_state(!cc.ptr<complex<f64>>, index) -> tensor<?xcomplex<f64>>
  func.func private @__quake_init_state_from_tensor(tensor<4xi1>, tensor<?xcomplex<f64>>) -> tensor<4xi1>
}

