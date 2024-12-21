from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch import Tensor
import torchattacks
import torchattacks.attacks
from torch.autograd import Variable
from torchattacks.attacks.autoattack import AutoAttack
from torchattacks.attacks.apgd import APGD
from torchattacks.attacks.fgsm import FGSM
from torchattacks.attacks.pgd import PGD
from torchattacks.attacks.eotpgd import EOTPGD
from torchattacks.attacks.mifgsm import MIFGSM


class MyAttack:
    def __init__(self, model:nn.Module, method:str, 
                 eps:float, alpha:float, steps:int, 
                 norm:str, mask_channel:List[int]=None, 
                 n_class=9):
        self.model = model
        self.method = method
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.mask_channel = mask_channel or []

        ATTACK_FN = {
            'PGD':  self.PGD,
            'FGSM': self.FGSM,
            'AUTOATTACK': self.AUTOATTACK,
            'APGD': self.APGD,
            'R_FGSM': self.R_FGSM,
            'R_PGD': self.R_PGD,
            'EOTPGD': self.EOTPGD,
            'MIFGSM': self.MIFGSM,
            'CW': self.CW, 
            'DeepFool': self.DeepFool,
            'PGDL2': self.PGDL2,
            'pixle': self.Pixle,
            'FAB': self.FAB,
            'EADL1': self.EADL1,
            'SEGPGD': self.SEGPGD
        }
        self.attack_func = ATTACK_FN[self.method]
    
    def __call__(self, X:Tensor, Y:Tensor):
        return self.attack_func(X, Y)

    def mask_grad_channel(self, g:Tensor):
        for c in self.mask_channel:
            g[:, c, :, :] = 0
        return g
    
    # Try this later
    def CW(self, X:Tensor, Y:Tensor):
        with torch.enable_grad():
            X.requires_grad = True
        
            atk_method = torchattacks.CW(self.model)
            breakpoint()
            AX = atk_method(X, Y)
        
        for c in self.mask_channel:
            AX[:, c, :, :] = X[:, c, :, :]
            
        return AX
    
    def Pixle(self, X:Tensor, Y:Tensor):
        with torch.enable_grad():
            X.requires_grad = True
        
            atk_method = torchattacks.Pixle(self.model)
            AX = atk_method(X, Y)
        
        for c in self.mask_channel:
            AX[:, c, :, :] = X[:, c, :, :]
            
        return AX
    
    def DeepFool(self, X:Tensor, Y:Tensor):
        with torch.enable_grad():
            X.requires_grad = True
        
            atk_method = torchattacks.DeepFool(self.model)
            AX = atk_method(X, Y)
        
        for c in self.mask_channel:
            AX[:, c, :, :] = X[:, c, :, :]
            
        return AX
    
    def PGDL2(self, X:Tensor, Y:Tensor):
        with torch.enable_grad():
            X.requires_grad = True
        
            atk_method = torchattacks.PGDL2(self.model, eps=3.0, alpha=1, steps=self.steps)
            AX = atk_method(X, Y)
            
        for c in self.mask_channel:
            AX[:, c, :, :] = X[:, c, :, :]
            
        return AX
    
    def FAB(self, X:Tensor, Y:Tensor):
        with torch.enable_grad():
            X.requires_grad = True
        
            atk_method = torchattacks.FAB(self.model)
            AX = atk_method(X, Y)
        
        for c in self.mask_channel:
            AX[:, c, :, :] = X[:, c, :, :]
            
        return AX
    
    def EADL1(self, X:Tensor, Y:Tensor):
        with torch.enable_grad():
            X.requires_grad = True
        
            atk_method = torchattacks.EADL1(self.model)
            AX = atk_method(X, Y)
        
        for c in self.mask_channel:
            AX[:, c, :, :] = X[:, c, :, :]
            
        return AX
    
    def EOTPGD(self, X:Tensor, Y:Tensor):
        with torch.enable_grad():
            X.requires_grad = True
        
            atk_method = torchattacks.EOTPGD(self.model)
            AX = atk_method(X, Y)
            
        for c in self.mask_channel:
            AX[:, c, :, :] = X[:, c, :, :]
            
        return AX
    
    def R_PGD(self, X:Tensor, Y:Tensor):
        with torch.enable_grad():
            X.requires_grad = True
        
            atk_method = torchattacks.PGD(self.model)
            AX = atk_method(X, Y)
            
        for c in self.mask_channel:
            AX[:, c, :, :] = X[:, c, :, :]
            
        return AX
    
    def R_FGSM(self, X:Tensor, Y:Tensor):
        with torch.enable_grad():
            X.requires_grad = True
        
            atk_method = torchattacks.FGSM(self.model)
            AX = atk_method(X, Y)
        
        for c in self.mask_channel:
            AX[:, c, :, :] = X[:, c, :, :]
            
        return AX
    
    def MIFGSM(self, X:Tensor, Y:Tensor):
        with torch.enable_grad():
            X.requires_grad = True
        
            atk_method = torchattacks.MIFGSM(self.model)
            AX = atk_method(X, Y)
        
        for c in self.mask_channel:
            AX[:, c, :, :] = X[:, c, :, :]
            
        return AX
    
    def APGD(self, X:Tensor, Y:Tensor):
        with torch.enable_grad():
            X.requires_grad = True
        
            atk_method = torchattacks.APGD(self.model)
            AX = atk_method(X, Y)
                
        for c in self.mask_channel:
            AX[:, c, :, :] = X[:, c, :, :]
            
        return AX
    
    def FGSM(self, X:Tensor, Y:Tensor):
        X.requires_grad = True
    
        with torch.enable_grad():
            logits = self.model(X)  
            
            if isinstance(logits, torch.Tensor) is False:
                logits = logits['out']
                
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
                
                if isinstance(logits, torch.Tensor) is False:
                    logits = logits['out']
                    
                loss = F.cross_entropy(logits, Y, reduction='none')
                g = grad(loss, AX, loss)[0]
            g = self.mask_grad_channel(g)
            
            AX = AX.detach() + g.sign() * self.alpha
            DX = (AX - X).clamp(-self.eps, self.eps)
            AX = (X + DX).clamp(0, 1).detach()
        return AX
  
    def AUTOATTACK(self, X:Tensor, Y:Tensor):
        atk_method = torchattacks.AutoAttack(self.model, norm='Linf', 
                                            eps=self.eps, version='standard', 
                                            n_classes=9, seed=None, verbose=False)
        AX = atk_method(X, Y)
        
        for c in self.mask_channel:
            AX[:, c, :, :] = X[:, c, :, :]
            
        return AX
        
    # def SEGPGD(self, image, new_images, new_labels):
    def SEGPGD(self, X:Tensor, Y:Tensor):
        image = X.to(dtype=torch.float32)
        new_images = Variable(image, requires_grad=True)
        new_labels = Variable(Y, requires_grad=False)
        
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        Total_iterations = 10
        eps = self.eps / Total_iterations
        for i in range(Total_iterations):
            new_images_d = new_images.detach()
            new_images_d.requires_grad_()
            with torch.enable_grad():
                logits = self.model(new_images_d)

                #logits vs new labels
                lamb = (i-1)/(Total_iterations*2)

                pred = torch.max(logits,1).values
                pred = torch.unsqueeze(pred,1)
                breakpoint()
                #    print(pred.shape)
                #    print(torch.unsqueeze(new_labels,1).shape)

                mask_t = pred == torch.unsqueeze(new_labels,1)
                mask_t = torch.squeeze(mask_t,1).int()
                np_mask_t = torch.unsqueeze(mask_t,1)

                mask_f = pred != torch.unsqueeze(new_labels,1)
                mask_f = torch.squeeze(mask_f,1).int()
                np_mask_f = torch.unsqueeze(mask_f,1)

                # need to be check the loss
                #    print((np_mask_t*logits).shape)
                #    print((new_labels).shape)
                loss_t = lamb* criterion(np_mask_t*logits, new_labels)
                loss_f = (1-lamb) * criterion(np_mask_f*logits, new_labels)
                loss = loss_t + loss_f
           
            grad = torch.autograd.grad(loss, [new_images_d])[0]
            image = image.detach() + eps * torch.sign(grad.detach())
            adversarial_x = torch.min(torch.max(new_images_d, new_images - eps*1), new_images + eps*1)

        for c in self.mask_channel:
            adversarial_x[:, c, :, :] = X[:, c, :, :]
        return adversarial_x
        
def get_attack(args, model):
    mask_channel = vars(args).get('mask_channel')
    return MyAttack(model, args.method, args.eps, args.alpha, args.steps, args.norm, mask_channel)

if __name__ == '__main__':
    x = torch.rand()