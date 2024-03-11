### train

> 我们会跑很多对比试验，所以需要做版本隔离
> 你需要魔改 train.py 使得能从命令行额外传入 `--exp <name> --source [rgb|inf|mix]` 来运行下列实验，并使得结果分别输出在 weights/\<name\>/*.pth

train the following configuration:

- [ ] Seg-rgb: SegNet with rgb input (in_ch=3)
- [ ] Seg-inf: SegNet with inf input (in_ch=1)
- [ ] Seg-mix: SegNet with rgb+inf input (in_ch=4)
- [ ] MF: MFNet with rgb+inf input (in_ch=4)


### eval

> 同理，你需要魔改 test.py 使得能从命令行额外传入 `--exp <name> --split [train|val|test]` 来进行下面的实验

run infer on all dataset splits, gather metrcis to fill following table:

| name | train | val | test |
| :-: | :-: | :-: | :-: |
| Seg-rgb | <Acc/mIoU> |  |  |
| Seg-inf |            |  |  |
| Seg-mix |            |  |  |
| MF      |            |  |  |


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

⚪ FGSM

> **e** for eps; **e** need /255

| name | limit | eps=4 | eps=8 | eps=12 | eps=16 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Seg-rgb |   -    | <Acc/mIoU> |  |  |  |
| Seg-inf |   -    |            |  |  |  |
| Seg-mix |   -    |            |  |  |  |
| Seg-mix |  rgb   |            |  |  |  |
| Seg-mix |  inf   |            |  |  |  |
| MF      |  -     |            |  |  |  |
| MF      |  rgb   |            |  |  |  |
| MF      |  inf   |            |  |  |  |

⚪ PGD

> **e** for eps, **a** for alpha, **s** for steps; **e** and **a** need /255

| name | limit | e=4,a=1,s=10 | e=8,a=1,s=10 | e=8,a=1,s=20 | e=16,a=1,s=20 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Seg-rgb |   -    | <Acc/mIoU> |  |  |  | 
| Seg-inf |   -    |            |  |  |  | 
| Seg-mix |   -    |            |  |  |  | 
| Seg-mix |  rgb   |            |  |  |  | 
| Seg-mix |  inf   |            |  |  |  | 
| MF      |  -     |            |  |  |  | 
| MF      |  rgb   |            |  |  |  | 
| MF      |  inf   |            |  |  |  | 


----
by Armit
2024/3/12
