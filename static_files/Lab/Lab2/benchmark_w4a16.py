import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import time
import numpy as np

import math
from int4_matmul_rowwise_dequantize import w4a16_mm

class QuantizedLinear_w4a16(nn.Linear):
    def __init__(self, in_features, out_features, bias, device):
        super().__init__(in_features, out_features, bias, device)
        bits = 4
        groupsize = -1
        groupsize = in_features if groupsize == -1 else groupsize
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.groupsize = groupsize
        features_per_int = 32 // bits

        # assert out_features % features_per_int == 0, "out_features must be a multiple of features_per_int"

        self.register_buffer('qweight', torch.zeros((in_features // features_per_int, out_features), dtype=torch.int32, device='cuda'))
        self.register_buffer('scales', torch.zeros((math.ceil(in_features / groupsize), out_features), dtype=torch.float16, device='cuda'))
            
        if bias:
            self.register_buffer('bias', torch.empty(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)

        self.qmax = 7
        self.qmin = -8
        

    def prepare_for_eval(self):
        print("=> preparing for eval.")
        scales = torch.max(torch.abs(self.weight), dim=-1, keepdim=True)[0] / ((self.qmax - self.qmin) / 2)

        intweight = []
        for idx in range(self.in_features):
            g_idx = idx // self.groupsize
            q = torch.round((self.weight[:,idx]) / scales[g_idx]).clamp(self.qmin, self.qmax).to(torch.int32)
            intweight.append(q[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        
	    # Now pack the weights into uint32's
        i = 0
        row = 0
        # pdb.set_trace()
        while row < self.qweight.shape[0]:
            for j in range(i, i + (32 // self.bits)):
                self.qweight[row] |= (intweight[j] & 0xF) << (self.bits * (j - i))
            i += 32 // self.bits
            row += 1
        
        self.scales = scales.t().contiguous().to(torch.float16)

        self.weights_debug = torch.round((self.weight / scales)).clamp(self.qmin, self.qmax).to(torch.int32).t()
        self.scales_debug = scales
        self.int_weight_debug = intweight

    def forward(self, x):
        y = w4a16_mm(self.groupsize, x, self.qweight, self.scales, self.bias)
        return y
     

if __name__ == '__main__':
    # test case
    print("****** Test case ******")
    M = 5
    K = 1024
    N = 2048
    x = torch.randn(M, K, device='cuda', dtype=torch.float16)

    in_chl = K
    out_chl = N
    quantized_linear = QuantizedLinear_w4a16(in_chl, out_chl, bias=False, device='cuda')
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

    print("w4a16 linear avg", np.mean(np.array(triton_latency)))


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
