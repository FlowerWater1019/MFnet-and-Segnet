from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch import Tensor


class MyAttack:

    def __init__(self, model:nn.Module, method:str, eps:float, alpha:float, steps:int, mask_channel:List[int]=None):
        self.model = model
        self.method = method
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.mask_channel = mask_channel or []

        ATTACK_FN = {
            'PGD':  self.PGD,
            'FGSM': self.FGSM,
        }
        self.attack_func = ATTACK_FN[self.method]
    
    def __call__(self, X:Tensor, Y:Tensor):
        return self.attack_func(X, Y)

    def mask_grad_channel(self, g:Tensor):
        for c in self.mask_channel:
            g[:, c, :, :] = 0
        return g

    def FGSM(self, X:Tensor, Y:Tensor):
        X.requires_grad = True
    
        with torch.enable_grad():
            logits = self.model(X)  
            loss = F.cross_entropy(logits, Y)
            g = grad(loss, X, loss)[0]  
        g = self.mask_grad_channel(g)

        AX = X + g.sign() * self.eps
        AX = AX.clamp(0.0, 1.0).detach()
        return AX

    def PGD(self, X:Tensor, Y:Tensor):
        X = X.detach().clone()
        AX = X.detach().clone() + (torch.rand_like(X) * 2 - 1) * self.eps
        for _ in range(self.steps):
            AX.requires_grad = True
      
            with torch.enable_grad():
                logits = self.model(AX)
                loss = F.cross_entropy(logits, Y, reduction='none')
                g = grad(loss, AX, loss)[0]
            g = self.mask_grad_channel(g)
            
            AX = AX.detach() + g.sign() * self.alpha
            DX = (AX - X).clamp(-self.eps, self.eps)
            AX = (X + DX).clamp(0, 1).detach()
        return AX
  
  
def get_attack(args, model):
    mask_channel = vars(args).get('mask_channel')
    return MyAttack(model, args.method, args.eps, args.alpha, args.steps, mask_channel)
