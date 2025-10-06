module {
  func.func @main() {
    %q = quake.alloca !quake.veq<4>
    %s = quake.veq_size %q : (!quake.veq<4>) -> index
    return
  }
}
