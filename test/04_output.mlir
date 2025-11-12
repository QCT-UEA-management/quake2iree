module {
  func.func @test_concat() {
    %0 = tensor.empty() : tensor<1xi1>
    %1 = tensor.empty() : tensor<1xi1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %2 = arith.addi %c0, %c1 : index
    %c1_0 = arith.constant 1 : index
    %3 = arith.addi %2, %c1_0 : index
    %4 = tensor.empty(%3) : tensor<?xi1>
    %c0_1 = arith.constant 0 : index
    %c1_2 = arith.constant 1 : index
    %inserted_slice = tensor.insert_slice %0 into %4[%c0_1] [1] [1] : tensor<1xi1> into tensor<?xi1>
    %5 = arith.addi %c0_1, %c1_2 : index
    %c1_3 = arith.constant 1 : index
    %inserted_slice_4 = tensor.insert_slice %1 into %inserted_slice[%5] [1] [1] : tensor<1xi1> into tensor<?xi1>
    %6 = arith.addi %5, %c1_3 : index
    return
  }
}

