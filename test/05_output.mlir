module {
  func.func @test_extract_ref() {
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1xi1>
    %extracted_slice = tensor.extract_slice %0[%c0] [1] [1] : tensor<1xi1> to tensor<1xi1>
    return
  }
}

