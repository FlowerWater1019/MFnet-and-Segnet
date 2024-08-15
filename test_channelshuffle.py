import torch
from torch import Tensor
import torchvision.transforms.functional as TF
import torchvision.datasets as DS
import torchvision.models as M
from tqdm import tqdm

def transform(x:Tensor) -> Tensor:
    r, g, b = x
    r_avg, r_std = r.mean(), r.std()
    g_avg, g_std = g.mean(), g.std()
    g_hat = (r - r_avg) / r_std * g_std + g_avg
    return torch.stack([r, g_hat, b], dim=0)


@torch.inference_mode()
def run():
    dataset = DS.CIFAR10(root=r'C:\Python\CCPP', train=False, download=True)
    model = M.resnet50(pretrained=True).cuda()

    tot, ok = 0, 0
    for x, y in tqdm(dataset):
        # [B=1, C=3, H=224, W=224]
        x = TF.to_tensor(x).cuda()
        x = TF.resize(x, (224, 224))

        # original
        logits = model(x.unsqueeze(0))[0]
        pred = logits.argmax(-1).item()
        # transformed
        logits = model(transform(x).unsqueeze(0))[0]
        pred_trans = logits.argmax(-1).item()

        tot += 1
        if pred == pred_trans:
            ok += 1

    print('stable rate:', ok / tot)


if __name__ == '__main__':
    run()
