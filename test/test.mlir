module {
  func.func @main() {
    %0 = quake.alloca !quake.veq<2>
    quake.dealloc %0 : !quake.veq<2>
    return
  }
}
