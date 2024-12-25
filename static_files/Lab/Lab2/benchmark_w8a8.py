import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import time
import numpy as np


class QuantizedLinear_w8a8(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.qmax = 127
        self.qmin = -128

    def quantize(self, x):
        # quantize weight offline
        scale = torch.max(torch.abs(x), dim=-1, keepdim=True)[0] / ((self.qmax - self.qmin) / 2)
        quantized_tensor = ((x / scale).round().clamp(self.qmin, self.qmax)).to(torch.int8)
        return quantized_tensor, scale

    def prepare_for_eval(self):
        # If we just want to do eval, we can pre-quantize the weights instead of doing it on the forward pass.
        print("=> preparing for eval.")
        qweight, weight_scale = self.quantize(self.weight)
        self.register_buffer("qweight", qweight)
        self.register_buffer("weight_scale", weight_scale)

    def forward(self, input: torch.Tensor):
        input_scale = torch.max(torch.abs(input), dim=-1, keepdim=True)[0] / self.qmax
        output = w8a16_mm(input, self.qweight.T, input_scale, self.weight_scale)
        return output


if __name__ == '__main__':

    torch.manual_seed(0)

    # # test case 1
    print("****** Test case 1 ******")
    in_chl = 4
    out_chl = 4
    
    activation = torch.tensor([[1, 2, 3, 4]],
                             dtype=torch.float16, device='cuda')
    
    weight = nn.Parameter(torch.tensor([[5, -1, 2, 0],    
                                        [-3, 4, 1, -2], 
                                        [5, 4, 0, -6],  
                                        [1, 0, -5, 3]],
                           dtype=torch.float16, device='cuda'))

    quantized_linear = QuantizedLinear_w8a8(in_chl, out_chl, bias=False, device='cuda')
    quantized_linear.weight.data = weight
    quantized_linear.prepare_for_eval()
    quantized_linear_out = quantized_linear(activation)

    fp_linear = nn.Linear(in_features=in_chl, out_features=out_chl, bias=False, device='cuda').eval().half()
    fp_linear.weight.data = weight
    fp_linear_out = fp_linear(activation)

    # pdb.set_trace()

    diff = (quantized_linear_out - fp_linear_out).abs().max()
    if diff > 0.5:
        raise ValueError(f"Output difference too large: {diff.item()}")
    else:
        print(f"Test passed. Maximum difference: {diff.item()}")

    # pdb.set_trace()

    # # test case 2
    print("****** Test case 2 ******")
    in_chl = 8
    out_chl = 6

    activation = torch.tensor([[1.0, -2.0, 3.0, 4.0, 0.5, -1.5, 2.5, 3.0],
                                [-1.0, 2.0, -3.0, -4.0, 1.5, 0.5, -2.5, -3.0]], 
                                dtype=torch.float16, device='cuda')

    weight = nn.Parameter(torch.tensor([[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8],  
                            [-0.9, 1.0, -1.1, 1.2, -1.3, 1.4, -1.5, 1.6],  
                            [0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9],  
                            [-1.0, 1.1, -1.2, 1.3, -1.4, 1.5, -1.6, 1.7],  
                            [0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0],  
                            [-0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -1.1, 1.2]],  
                            dtype=torch.float16, device='cuda'))

    quantized_linear = QuantizedLinear_w8a8(in_chl, out_chl, bias=False, device='cuda')
    quantized_linear.weight = weight
    quantized_linear.prepare_for_eval()
    quantized_linear_out = quantized_linear(activation)

    fp_linear = nn.Linear(in_features=in_chl, out_features=out_chl, bias=False, device='cuda').eval().half()
    fp_linear.weight.data = weight
    fp_linear_out = fp_linear(activation)

    # pdb.set_trace()

    diff = (quantized_linear_out - fp_linear_out).abs().max()
    if diff > 0.5:
        raise ValueError(f"Output difference too large: {diff.item()}")
    else:
        print(f"Test passed. Maximum difference: {diff.item()}")

    # pdb.set_trace()


    # test case 3
    print("****** Test case 3 ******")
    M = 5
    K = 1024
    N = 2048
    x = torch.randn(M, K, device='cuda', dtype=torch.float16)

    in_chl = K
    out_chl = N
    quantized_linear = QuantizedLinear_w8a8(in_chl, out_chl, bias=False, device='cuda')
    quantized_linear.prepare_for_eval()
    quantized_linear_out = quantized_linear(x)

    fp_linear = nn.Linear(in_features=in_chl, out_features=out_chl, bias=False, device='cuda').eval()
    fp_linear.weight.data = quantized_linear.weight.data
    fp16_linear = fp_linear.half()
    fp_linear.eval()
    fp_linear_out = fp16_linear(x)

    diff = (quantized_linear_out - fp_linear_out).abs().max()
    if diff > 0.5:
        raise ValueError(f"Output difference too large: {diff.item()}")
    else:
        print(f"Test passed. Maximum difference: {diff.item()}")

    # pdb.set_trace()

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
        quantized_linear(x)
        if i >= warmup_iters: 
            torch.cuda.synchronize()
            end = time.time()
            triton_latency.append((end - start)*1000)
            # print((end - start)*1000)
        
        if i >= warmup_iters: 
            torch.cuda.synchronize()

    print("w8a8 linear avg", np.mean(np.array(triton_latency)))


    fp16_latency = []
    
    for i in range(nb_iters):
        if i >= warmup_iters: 
            torch.cuda.synchronize()

        if i >= warmup_iters:
            torch.cuda.synchronize()
            start = time.time()
        fp16_linear(x)
        if i >= warmup_iters: 
            torch.cuda.synchronize()
            end = time.time()
            fp16_latency.append((end - start)*1000)
            # print((end - start)*1000)
        
        if i >= warmup_iters: 
            torch.cuda.synchronize()

    print("fp16 linear avg", np.mean(np.array(fp16_latency)))


    print("finish profiling")
