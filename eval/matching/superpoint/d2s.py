import torch
import torch.nn as nn


class DepthToSpace(nn.Module):
    def __init__(self, block_size: int):
        super(DepthToSpace, self).__init__()
        # here we are casting int to a tensor. This will cause a warning when converting the model to ONNX, 
        # but we are calling this module in an inference with a constant. Therefore it is not a problem
        self.block_size = torch.tensor(block_size, dtype=torch.int32)
        self.block_size_sq = torch.tensor(block_size*block_size, dtype=torch.int32)

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = (d_depth / self.block_size_sq).type(torch.int64)
        s_width = (d_width * self.block_size).type(torch.int32)
        s_height = (d_height * self.block_size).type(torch.int32)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output
