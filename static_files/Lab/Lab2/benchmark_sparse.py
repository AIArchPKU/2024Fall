import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import time
import numpy as np

from sparse_matmul import dds_mm

class SparseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def prepare_for_eval(self):
        print("=> preparing for eval.")
        sparse_weight = self.weight.t().to_sparse_bsr(blocksize=(32, 32))        
        shape = torch.tensor(sparse_weight.size())
        row_offset, col_indice, value = sparse_weight.crow_indices(), sparse_weight.col_indices(), sparse_weight.values()
        
        # pdb.set_trace()

        self.register_buffer("shape", shape)
        self.register_buffer("row_offset", row_offset)
        self.register_buffer("col_indice", col_indice)
        self.register_buffer("value", value)

    def forward(self, input: torch.Tensor):
        output = dds_mm(input, self.shape, self.value, self.row_offset, self.col_indice)
        return output


if __name__ == '__main__':

    torch.manual_seed(0)
    size = (1024, 1024)

    activation = torch.randn((1, 1024),
                             dtype=torch.float16, device='cuda')

    # # test case 1
    print("****** Test case 1 ******")
    matrix = torch.zeros(size)
    random_matrix = torch.rand(size)
    sparsity = 0.3
    matrix[random_matrix < sparsity] = torch.randn(matrix[random_matrix < sparsity].shape)

    weight = nn.Parameter(matrix.half()).cuda()

    fp_linear = nn.Linear(in_features=1024, out_features=1024, bias=False, device='cuda').eval().half()
    fp_linear.weight.data = weight
    fp_linear_out = fp_linear(activation)

    sparse_linear = SparseLinear(1024, 1024, bias=False, device='cuda')
    sparse_linear.weight.data = weight
    sparse_linear.prepare_for_eval()
    sparse_linear_out = sparse_linear(activation)

    diff = (sparse_linear_out - fp_linear_out).abs().max()
    if diff > 0.5:
        raise ValueError(f"Output difference too large: {diff.item()}")
    else:
        print(f"Test passed. Maximum difference: {diff.item()}")


    # pdb.set_trace()
    # test case 2
    print("****** Test case 2 ******")
    matrix = torch.zeros(size)
    random_matrix = torch.rand(size)
    sparsity = 0.6
    matrix[random_matrix < sparsity] = torch.randn(matrix[random_matrix < sparsity].shape)

    weight = nn.Parameter(matrix.half()).cuda()

    fp_linear = nn.Linear(in_features=1024, out_features=1024, bias=False, device='cuda').eval().half()
    fp_linear.weight.data = weight
    fp_linear_out = fp_linear(activation)

    sparse_linear = SparseLinear(1024, 1024, bias=False, device='cuda')
    sparse_linear.weight.data = weight
    sparse_linear.prepare_for_eval()
    sparse_linear_out = sparse_linear(activation)

    diff = (sparse_linear_out - fp_linear_out).abs().max()
    if diff > 0.5:
        raise ValueError(f"Output difference too large: {diff.item()}")
    else:
        print(f"Test passed. Maximum difference: {diff.item()}")


    # test case 3
    print("****** Test case 3 ******")
    matrix = torch.zeros(size)
    random_matrix = torch.rand(size)
    sparsity = 0.9
    matrix[random_matrix < sparsity] = torch.randn(matrix[random_matrix < sparsity].shape)

    weight = nn.Parameter(matrix.half()).cuda()

    fp_linear = nn.Linear(in_features=1024, out_features=1024, bias=False, device='cuda').eval().half()
    fp_linear.weight.data = weight
    fp_linear_out = fp_linear(activation)

    sparse_linear = SparseLinear(1024, 1024, bias=False, device='cuda')
    sparse_linear.weight.data = weight
    sparse_linear.prepare_for_eval()
    sparse_linear_out = sparse_linear(activation)

    diff = (sparse_linear_out - fp_linear_out).abs().max()
    if diff > 0.5:
        raise ValueError(f"Output difference too large: {diff.item()}")
    else:
        print(f"Test passed. Maximum difference: {diff.item()}")


    # compare latency
    nb_iters = 100
    warmup_iters = 10

    triton_latency = []
    
    for i in range(nb_iters):
        if i >= warmup_iters: 
            torch.cuda.synchronize()

        if i >= warmup_iters:
            torch.cuda.synchronize()
            start = time.time()
        sparse_linear(activation)
        if i >= warmup_iters: 
            torch.cuda.synchronize()
            end = time.time()
            triton_latency.append((end - start)*1000)
            # print((end - start)*1000)
        
        if i >= warmup_iters: 
            torch.cuda.synchronize()

    print("sparse linear avg", np.mean(np.array(triton_latency)))

    fp16_latency = []
    
    for i in range(nb_iters):
        if i >= warmup_iters: 
            torch.cuda.synchronize()

        if i >= warmup_iters:
            torch.cuda.synchronize()
            start = time.time()
        sparse_linear(activation)
        if i >= warmup_iters: 
            torch.cuda.synchronize()
            end = time.time()
            fp16_latency.append((end - start)*1000)
            # print((end - start)*1000)
        
        if i >= warmup_iters: 
            torch.cuda.synchronize()

    print("fp16 linear avg", np.mean(np.array(fp16_latency)))
    print("finish profiling")
