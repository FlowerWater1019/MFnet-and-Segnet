### train

> 我们会跑很多对比试验，所以需要做版本隔离
> 你需要魔改 train.py 使得能从命令行额外传入 `--exp <name> --source [rgb|inf|mix]` 来运行下列实验，并使得结果分别输出在 weights/\<name\>/*.pth

train the following configuration:

- [$\circ$] Seg-rgb: SegNet with rgb input (in_ch=3)
- [$\circ$] Seg-inf: SegNet with inf input (in_ch=1)
- [$\circ$] Seg-mix: SegNet with rgb+inf input (in_ch=4)
- [$\circ$] MF: MFNet with rgb+inf input (in_ch=4)

### attack

lauch attacks on `test` dataset split, gather metrcis to fill following table:

> Q: Seg-mix/MF 搭配 limit=rgb/inf 是什么意思，如何实现?
> A: 从数据意义上，这两个模型都需要输入两个成分，通常的PGD攻击会在整个输入(也就是两个成份上)同时添加扰动，而我们想要测试只在一个成分上添加扰动的攻击效果 ;) 
> 实现方式：魔改 attack.py 在取得 loss 后，给 loss 加一个 mask 以掩蔽另一个暂时不考虑的成分所产生的损失值，给 grad 也加同样的 mask 保证产生新的对抗样本时没有在该成分上引入噪声

参考实现 (你应该写得更灵活，而不是if-else写死各种情况)：

```python
# [B, C=4, H, W], C 方向 0~2 是 rgb, 3 是 inf
# 考虑现在 limit=rgb 而不能修改 inf
X = randn([1, 4, 224, 224])
# 产生mask
M = zeros_like(X)
M[:, :2, :, :] = 1   # allow rgb part
# 初始噪声
AX = X + randu_like(X, -eps, eps) * M
# 迭代噪声
logits = model(X)
loss = loss_fn(logits, Y) * M
g = grad(loss, X, loss) * M
AX_new = AX + g.sign() * step_size
......
```


### eval

> 同理，你需要魔改 test.py 使得能从命令行额外传入 `--exp <name> --split [train|val|test]` 来进行下面的实验

run infer on all dataset splits, gather metrcis to fill following table(Acc/mIoU): 

⚪ FGSM

> **e** for eps; **e** need /255

| name | limit | eps=4 | eps=8 | eps=12 | eps=16 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Seg-rgb |   -    | 18.30%/23.97% | 12.11%/23.02% | 8.24%/22.80% | 5.07%/23.31% |
| Seg-inf |   -    | 11.06%/23.07% | 4.36%/18.99% | 1.42%/19.56% | 0.22%/21.01% |
| Seg-mix |   -    | 22.73%/16.89% | 19.41%/12.00% | 17.47%/10.56% | 15.83%/9.81% |
| Seg-mix |  rgb   | 24.99%/20.08% | 21.80%/14.27% | 20.01%/11.91% | 18.64%/10.75% |
| Seg-mix |  inf   | 32.43%/34.67% | 27.00%/29.19% | 24.25%/22.66% | 22.25%/20.12% |
| MF      |  -     | 21.84%/22.43% | 16.31%/12.59% | 13.46%/9.82% | 11.72%/ 9.62% |
| MF      |  rgb   | 28.01%/25.91% | 23.85%/18.06% | 22.16%/16.06% | 20.92%/15.01% |
| MF      |  inf   | 47.78%/51.14% | 44.24%/49.72% | 39.85%/42.84% | 37.61%/34.62% |

⚪ PGD

> **e** for eps, **a** for alpha, **s** for steps; **e** and **a** need /255, and Acc here is overall_acc, not acc.mean(), Acc/mIoU

| name | limit | e=4,a=1,s=10 | e=8,a=1,s=10 | e=8,a=1,s=20 | e=16,a=1,s=20 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Seg-rgb |   -    | 10.28%/8.90% | 5.76%/6.70% | 11.79%/4.27% | 6.45%/2.81% | 
| Seg-inf |   -    | 9.17%/4.89% | 6.28%/3.40% | 11.00%/2.89% | 13.40%/2.94% | 
| Seg-mix |   -    | 15.75%/8.64% | 14.09%/7.52% | 15.96%/5.31% | 11.88%/4.74% | 
| Seg-mix |  rgb   | 16.72%/11.43% | 14.32%/9.30% | 15.53%/6.04% | 10.93%/4.97% | 
| Seg-mix |  inf   | 23.30%/21.93% | 18.33%/17.93% | 15.09%/13.53% | 10.97%/8.61% | 
| MF      |  -     | 12.93%/11.87% | 11.27%/6.70% | 12.64%/5.32% | 9.25%/3.40% | 
| MF      |  rgb   | 15.30%/16.04% | 12.38%/8.19% | 12.19%/6.25% | 7.34%/4.05% | 
| MF      |  inf   | 26.09%/35.67% | 20.45%/19.87% | 18.15%/16.87% | 10.07%/9.32% | 

多模态模型测试 ( 无数据融合 )

| name | Train | Val | Test |
| :-: | :-: | :-: | :-: |
| Seg-rgb | 58.84% | 50.07% | 49.67% |
| Seg-inf | 51.24% | 36.07% | 38.22% |
| Seg-mix | 62.37% | 50.38% | 50.61% |
| MF      | 82.15% | 62.54% | 57.70% |
| DeepLabV3-inf | 86.27% | 55.96% | 55.10% |
| DeepLabV3-rgb | 86.62% | 69.82% | 65.56% |
| DeepLabV3-mix | 86.45% | 63.45% | 60.41% |

将原数据集中图片经过两种融合模型融合后训练并测试得出结果(Acc / mIoU)

| Fuse + model | Train | Val | Test |
| :-: | :-: | :-: | :-: |
| MMIF-CDDFuse + SegNet | 60.42% | 37.66% | 39.99% |
| Dif-Fusion + SegNet | 52.59% | 34.63% | 35.79% |
| MMIF-CDDFuse + DeepLabV3 | 86.32% | 54.67% | 57.73% |
| Dif-Fusion + DeepLabV3 | 86.31% | 57.26% | 57.19% |

----

在PGD攻击, 参数为 eps = 8/255, alpha = 1/255, steps = 16. 下表中不同channel代表攻击的通道, 结果为acc/mIoU

| model | channel[0] | channel[1] | channel[2] | channel[ir] |
| :-: | :-: | :-: | :-: | :-: |
| SegNet-inf | - | - | - | 2.68% |
| SegNet-rgb | 11.58% | 8.82% | 11.68% | - |
| SegNet-mix | 17.61% | 13.03% | 16.60% | 15.34% |
| MFNet | 17.98% | 11.26% | 15.63% | 18.14% |
| DeepLabV3-inf | - | - | - | 10.24% |
| DeepLabV3-rgb | 22.72% | 22.41% | 24.48% | - |
| DeepLabV3-mix | 27.48% | 23.05% | 25.93% | 25.21% |


| model | channel[0+1] | channel[0+2] | channel[1+2] | channel[0+ir] | channel[1+ir] | channel[2+ir] |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| SegNet-rgb | 5.86% | 7.08% | 6.00% | - | - | - |
| SegNet-mix | 8.82% | 10.69% | 8.38% | 9.88% | 7.68% | 9.23% |
| MFNet | 7.87% | 10.29% | 7.99% | 9.60% | 6.96% | 9.05% |
| DeepLabV3-rgb | 20.60% | 21.54% | 21.30% | - | - | - |
| DeepLabV3-mix | 19.30% | 21.58% | 20.64% | 21.06% | 16.86% | 21.40% |


| model | channel[rgb] | channel[0+1+ir] | channel[0+2+ir] | channel[1+2+ir] | channel[rgb+ir] |
| :-: | :-: | :-: | :-: | :-: | :-: |
| SegNet-rgb | 4.57% | - | - | - | - | 
| SegNet-mix | 6.80% | 6.59% | 7.39% | 6.22% | 5.90% |
| MFNet | 6.77% | 6.01% | 6.93% | 6.02% | 5.43% |
| DeepLabV3-rgb | 18.84% | - | - | - | - |
| DeepLabV3-mix | 14.72% | 14.67% | 20.04% | 19.79% | 15.03% |

----

在MIFGSM攻击, 参数为eps = 8/255, alpha = 2/255, steps = 10, decay = 1.0. 下表中不同channel代表攻击的通道, 结果为acc/mIoU

| model | channel[0] | channel[1] | channel[2] | channel[ir] |
| :-: | :-: | :-: | :-: | :-: |
| SegNet-inf | - | - | - | 2.94% |
| SegNet-rgb | 17.43% | 10.99% | 17.09% | - |
| SegNet-mix | 28.82% | 12.09% | 23.06% | 16.73% |
| MFNet | 39.81% | 11.27% | 24.52% | 28.81% |
| DeepLabV3-inf | - | - | - | 5.93% |
| DeepLabV3-rgb | 33.97% | 23.34% | 43.00% | - |
| DeepLabV3-mix | 33.01% | 31.88% | 32.10% | 29.60% |


| model | channel[0+1] | channel[0+2] | channel[1+2] | channel[0+ir] | channel[1+ir] | channel[2+ir] |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| SegNet-rgb | 6.79% | 8.10% | 6.64% | - | - | - |
| SegNet-mix | 7.04% | 10.80% | 6.67% | 9.18% | 5.86% | 7.88% |
| MFNet | 7.42% | 12.00% | 6.79% | 15.61% | 5.11% | 8.21% |
| DeepLabV3-rgb | 13.83% | 17.82% | 14.80% | - | - | - |
| DeepLabV3-mix | 18.45% | 18.57% | 17.69% | 17.12% | 16.31% | 16.48% |


| model | channel[rgb] | channel[0+1+ir] | channel[0+2+ir] | channel[1+2+ir] | channel[rgb+ir] |
| :-: | :-: | :-: | :-: | :-: | :-: |
| SegNet-rgb | 5.45% | - | - | - | - |
| SegNet-mix | 4.70% | 4.44% | 5.27% | 4.30% | 3.90% |
| MFNet | 5.57% | 4.43% | 5.29% | 4.33% | 4.10% |
| DeepLabV3-rgb | 12.45% | - | - | - | - |
| DeepLabV3-mix | 12.40% | 12.10% | 12.02% | 11.97% | 11.19% |

----
by Armit
2024/3/12
