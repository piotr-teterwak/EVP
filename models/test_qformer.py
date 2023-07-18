from blip2_qformer import Blip2Qformer
import numpy as np
import torch

class_representations = np.load('../representations.npz')

representation_array = torch.concat([torch.Tensor(class_representations[idx]) for idx in class_representations]).view(1,-1,6656).cuda()

qformer = Blip2Qformer(input_width = 6656).cuda()

test = qformer(representation_array)

