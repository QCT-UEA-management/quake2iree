module {
  func.func @init_from_complex(%state : !cc.ptr<complex<f64>>) {
    %qv = quake.alloca !quake.veq<4>
    %new = quake.init_state %qv, %state
          : (!quake.veq<4>, !cc.ptr<complex<f64>>) -> !quake.veq<4>
    func.return
  }
}