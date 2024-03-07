import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch import Tensor


class MyAttack:

    def __init__(self, model:nn.Module, method:str, eps:float, alpha:float, steps:int):
        self.model = model
        self.method = method
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
    
    def __call__(self, X:Tensor, Y:Tensor):
        ATTACK_FN = {
        'PGD':    self.PGD,
        'FGSM':   self.FGSM,
        }
        return ATTACK_FN[self.method](X, Y)
  
    def FGSM(self, X:Tensor, Y:Tensor):
        X.requires_grad = True
    
        with torch.enable_grad():
            logits = self.model(X)  
            loss = F.cross_entropy(logits, Y)
            g = grad(loss, X, loss)[0]  
            
        AX = X + g.sign() * self.eps
        AX = torch.clamp(AX, min=0.0, max=1.0).detach()
        return AX

    def PGD(self, X:Tensor, Y:Tensor):
        X_orig = X.detach().clone()
        AX = X_orig.detach().clone() + (torch.rand_like(X_orig) * 2 - 1) * self.eps
        for i in range(self.steps):
            AX.requires_grad = True
      
            with torch.enable_grad():
                logits = self.model(AX)
                loss = F.cross_entropy(logits, Y, reduction='none')
                g = grad(loss, AX, loss)[0]
        
            AX_new = AX.detach() + g.sign() * self.alpha
            DX = (AX_new - X_orig).clamp(-self.eps, self.eps)
            AX = (X_orig + DX).clamp(0, 1).detach()
        return AX
  
  
def get_attack(args, model):
    return MyAttack(model, args.method, args.eps, args.alpha, args.steps)
