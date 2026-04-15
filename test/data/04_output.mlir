module {
  func.func @test_concat() {
    %0 = tensor.empty() : tensor<1xi1>
    %1 = tensor.empty() : tensor<1xi1>
    %2 = tensor.empty() : tensor<2xi1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %inserted_slice = tensor.insert_slice %0 into %2[%c0] [1] [1] : tensor<1xi1> into tensor<2xi1>
    %3 = arith.addi %c0, %c1 : index
    %c1_0 = arith.constant 1 : index
    %inserted_slice_1 = tensor.insert_slice %1 into %inserted_slice[%3] [1] [1] : tensor<1xi1> into tensor<2xi1>
    %4 = arith.addi %3, %c1_0 : index
    return
  }
}

