module {
  // This global defines the initial state amplitudes (real CUDA-Q pattern)
  cc.global @rodata_0 : !cc.array<complex<f32> x 4> = [#complex<1.0 : f32, 0.0 : f32>,
                                                       #complex<0.0 : f32, 0.0 : f32>,
                                                       #complex<0.0 : f32, 0.0 : f32>,
                                                       #complex<0.0 : f32, 0.0 : f32>]
    {addr_space = 4}

  func.func @main() {
    %state = cc.address_of @rodata_0 : !cc.ptr<!cc.array<complex<f32> x 4>>
    %q = quake.alloca !quake.veq<2>
    %init = quake.init_state %q, %state :
      (!quake.veq<2>, !cc.ptr<!cc.array<complex<f32> x 4>>) -> !quake.veq<2>
    return
  }
}