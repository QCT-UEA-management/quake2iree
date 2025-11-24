module {
  func.func @test_initialize_state_static(%state : !cc.ptr<complex<f64>>) {
    %qv = quake.alloca  !quake.veq<4>
    %new = quake.init_state %qv, %state
            : (!quake.veq<4>, !cc.ptr<complex<f64>>) -> !quake.veq<4>
    func.return
  }

  func.func private @__quake_load_state(!cc.ptr<complex<f64>>, index) -> tensor<?xcomplex<f64>>
  func.func private @__quake_init_state_from_tensor(tensor<4xi1>, tensor<?xcomplex<f64>>) -> tensor<4xi1>
}