module {
  func.func @test_dynamic(%size : i32) -> (!quake.veq<?>) {
    %q = quake.alloca[%size : i32] : !quake.veq<?>
    return %q : !quake.veq<?>
  }

  func.func @test_static() -> (!quake.veq<4>) {
    %q = quake.alloca : !quake.veq<4>
    return %q : !quake.veq<4>
  }

  func.func @test_scalar() -> (!quake.ref) {
    %q = quake.alloca : !quake.ref
    return %q : !quake.ref
  }
}
