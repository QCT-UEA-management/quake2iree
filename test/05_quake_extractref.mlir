module {
  func.func @test_extract_ref() {
    
    %zero = arith.constant 0 : index
    %qv = quake.alloca !quake.veq<1>
    %qr = quake.extract_ref %qv[%zero]
          : (!quake.veq<1>, index) -> !quake.ref


    func.return
  }
}