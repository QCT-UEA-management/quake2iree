module {
  func.func @dealloc_example() {
    %q = quake.alloca !quake.veq<4>
    quake.dealloc %q : !quake.veq<4>
    return
  }
}