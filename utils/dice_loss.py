""" calculation of DICE

Typical usage example:

diceitem = dice_coeff(pred, true_masks).item()
"""
import torch
from torch.autograd import Function

class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 1e-3
        # input = input[1:,:,:]
        # target = target[1:,:,:]
        self.inter = torch.dot(input.contiguous().view(-1), target.contiguous().view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        t = (2 * self.inter.float() + eps) / self.union.float()
        if t<0:
            print(self.union,self.inter)
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None
        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None
        return grad_input, grad_target

"""Dice coeff for batches

    Calculate the Dice coeff for batches

    Args:
        input: mask result
        target: label

    Returns:
        DICE: %
    """
def dice_coeff(input, target):
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)