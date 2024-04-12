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

| name | train | val | test |
| :-: | :-: | :-: | :-: |
| Seg-rgb | 79.83%/58.84% | 52.85%/50.07% | 56.19%/49.67% |
| Seg-inf | 78.72%/51.24% | 36.94%/36.07% | 49.33%/38.22% |
| Seg-mix | 84.51%/62.37% | 54.62%/50.38% | 56.41%/50.61% |
| MF      | 84.45%/82.15% | 53.83%/62.54% | 60.79%/57.70% |

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


----
by Armit
2024/3/12
