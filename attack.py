from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad
import torchattacks as TA

ATTACK_METHODS = [
    # ↓↓ native
    'PGD',
    'FGSM',
    'SEGPGD',
    # ↓↓ torchattacks
    'PGDL2',
    #'PGDRS',
    #'PGDRSL2',
    #'APGDLinf',    # OOM
    #'APGDTLinf',   # OOM
    #'APGDL2',      # OOM
    #'APGDTL2',     # OOM
    'TPGD',
    'UPGD',
    'EOTPGD',
    'MIFGSM',
    #'DIFGSM',      # output.shape mismatch
    #'TIFGSM',      # output.shape mismatch
    #'PIFGSM',      # only for clf model
    'VNIFGSM',
    'VMIFGSM',
    'SINIFGSM',
    #'DeepFool',    # only for clf model
    #'CW',          # only for clf model
    #'EADL1',       # CW-like, only for clf model
    #'JSMA',        # only for clf model
]


class MyAttack:

    def __init__(self, model:nn.Module, method:str, eps:float, alpha:float, steps:int, mask_channel:List[int]=None):
        self.model = model
        self.method = method
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.mask_channel = mask_channel or []

        ATTACK_FN = {
            'PGD':    self.PGD,
            'FGSM':   self.FGSM,
            'SEGPGD': self.SEGPGD,
        }
        self.attack_func = ATTACK_FN.get(self.method)

    ''' ↓↓ torchattacks dispacth '''

    @torch.enable_grad
    def __call_torchattacks__(self, X:Tensor, Y:Tensor):
        TORCHATTACKS_METHODS = {
            # Linf
            'TPGD':       lambda model: TA.TPGD    (model, self.eps, self.alpha, self.steps),
            'UPGD':       lambda model: TA.UPGD    (model, self.eps, self.alpha, self.steps),
            'PGDRS':      lambda model: TA.PGDRS   (model, self.eps, self.alpha, self.steps),
            'EOTPGD':     lambda model: TA.EOTPGD  (model, self.eps, self.alpha, self.steps),
            'MIFGSM':     lambda model: TA.MIFGSM  (model, self.eps, self.alpha, self.steps),
            'DIFGSM':     lambda model: TA.DIFGSM  (model, self.eps, self.alpha, self.steps),
            'TIFGSM':     lambda model: TA.TIFGSM  (model, self.eps, self.alpha, self.steps),
            'PIFGSM':     lambda model: TA.PIFGSM  (model, max_epsilon=self.eps, num_iter_set=self.steps),
            'VNIFGSM':    lambda model: TA.VNIFGSM (model, self.eps, self.alpha, self.steps),
            'VMIFGSM':    lambda model: TA.VMIFGSM (model, self.eps, self.alpha, self.steps),
            'SINIFGSM':   lambda model: TA.SINIFGSM(model, self.eps, self.alpha, self.steps),
            'APGDLinf':   lambda model: TA.APGD    (model, norm='Linf', eps=self.eps, steps=self.steps),
            'APGDTLinf':  lambda model: TA.APGDT   (model, norm='Linf', eps=self.eps, steps=self.steps),
            'DeepFool':   lambda model: TA.DeepFool(model, steps=self.steps),
            # L2
            'PGDRSL2':    lambda model: TA.PGDRSL2 (model, eps=1.0, alpha=0.2, steps=self.steps),
            'PGDL2':      lambda model: TA.PGDL2   (model, eps=1.0, alpha=0.2, steps=self.steps),
            'APGDL2':     lambda model: TA.APGD    (model, norm='L2', eps=1.0, steps=self.steps),
            'APGDTL2':    lambda model: TA.APGDT   (model, norm='L2', eps=1.0, steps=self.steps),
            'CW':         lambda model: TA.CW      (model, steps=self.steps),
            # L1
            'EADL1':      lambda model: TA.EADL1   (model, max_iterations=self.steps),
            # L0
            'JSMA':       lambda model: TA.JSMA    (model, self.eps),
        }

        atk = TORCHATTACKS_METHODS[self.method](self.model)
        AX = atk(X, Y)
        for c in self.mask_channel:
            AX[:, c, :, :] = X[:, c, :, :]
        return AX

    ''' ↓↓ native impl. '''

    @torch.enable_grad
    def __call__(self, X:Tensor, Y:Tensor):
        if self.attack_func is None:
            return self.__call_torchattacks__(X, Y)
        else:
            return self.attack_func(X, Y)

    def mask_grad_channel(self, g:Tensor) -> Tensor:
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
                
                if isinstance(logits, torch.Tensor) is False:
                    logits = logits['out']
                    
                loss = F.cross_entropy(logits, Y, reduction='none')
                g = grad(loss, AX, loss)[0]
            g = self.mask_grad_channel(g)
            
            AX = AX.detach() + g.sign() * self.alpha
            DX = (AX - X).clamp(-self.eps, self.eps)
            AX = (X + DX).clamp(0, 1).detach()
        return AX

    # https://github.com/u6630774/SegPGD/blob/main/attacks.py#L59
    def SEGPGD(self, X:Tensor, Y:Tensor):
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        alpha = self.eps / self.steps
        for i in range(self.steps):
            new_images_d = X.detach()
            new_images_d.requires_grad_()
            with torch.enable_grad():
                logits = self.model(new_images_d)
                pred = torch.max(logits,1).values
                pred = torch.unsqueeze(pred,1)

                mask_t = pred == torch.unsqueeze(Y,1)
                mask_t = torch.squeeze(mask_t,1).int()
                np_mask_t = torch.unsqueeze(mask_t,1)
                mask_f = pred != torch.unsqueeze(Y,1)
                mask_f = torch.squeeze(mask_f,1).int()
                np_mask_f = torch.unsqueeze(mask_f,1)

                #logits vs new labels
                lamb = (i-1)/(self.steps*2)
                loss_t = lamb* criterion(np_mask_t*logits, Y)
                loss_f = (1-lamb) * criterion(np_mask_f*logits, Y)
                loss = loss_t + loss_f

            g = grad(loss, [new_images_d])[0]
            new_images_d = new_images_d.detach() + alpha * torch.sign(g)
            adversarial_x = torch.min(torch.max(new_images_d, X - alpha), X + alpha)

        for c in self.mask_channel:
            adversarial_x[:, c, :, :] = X[:, c, :, :]
        return adversarial_x


def get_attack(args, model:str):
    mask_channel = vars(args).get('mask_channel')
    return MyAttack(model, args.method, args.eps, args.alpha, args.steps, mask_channel)
