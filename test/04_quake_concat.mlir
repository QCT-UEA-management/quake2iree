module {
  func.func @test_concat() {
    %r1 = quake.alloca  !quake.ref
    %r2 = quake.alloca  !quake.ref

    %v_concat = quake.concat %r1, %r2
      : (!quake.ref, !quake.ref) -> !quake.veq<2>

    func.return
  }
}

