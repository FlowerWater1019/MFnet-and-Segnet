## 补充数据融合模型相关的实验

> 数据融合模型可以视作一种数据预处理，将 visible 和 infrared 数据提前融合成一张图(通道数待定)，下游模型的输入也就只有这一张图，故考虑只 SegNet 即可

### 实验步骤

- setup 3rd-party repos
  - run `repo\init_repos.cmd`
  - `MMIF-CDDFuse`: already contains ckpt under `models/` folder
  - `Dif-Fusion`: manually download ckpt follow the README.md
- generate fused data from MFNet dataset
  - run `python infer_MMIF-CDDFuse_IVF.py` (TODO: 👉 [任务一](#任务一跑通-infer_mmif-cddfuse_ivfpy获得融合数据))
  - run `python infer_Dif-Fusion.py` (TODO: 👉 [任务三](#任务三写胶水脚本-infer_dif-fusionpy重复整套融合-训练-测试过程))
- train `SegNet` model on the generated datasets (TODO: 👉 [任务二](#任务二改造-mf_dataset-类跑-segnet-训练))
  - `data/MF_CDDFuse_IVF`
  - `data/MF_Dif-Fusion`
- evaluate the trained model checkpoints normally and adversarially


### 任务一：跑通 infer_MMIF-CDDFuse_IVF.py，获得融合数据

- run `repo\init_repos.cmd`
- run `python infer_MMIF-CDDFuse_IVF.py`
- check generated files under `data/MF_CDDFuse_IVF`


### 任务二：改造 MF_dataset 类，跑 SegNet 训练-测试

ℹ `infer_*.py` 脚本产生的融合数据文件夹 `data/MF_*` 里只含图像，不含标注等等其他元信息，即仅仅等价于 `data/MF/images` 文件夹，所以需要改造 dataloader 相关的代码来复用那些元信息文件（而不要复制很多份其他元信息，去手搓一个满足原格式的数据集文件夹！）

> 基本的思想是重写 `MF_dataset` 类中的 `read_image` 函数即可

- 继承 `MF_dataset` 类，写一个子类 `MF_ext_dataset`
- 子类的构造函数 `__init__` 接受一个额外的参数 `images_dir:str`
- 重写父类中的 `read_image`，改变 `folder="images"` 时该方法的逻辑
- 相应地改造 `train.py` 和 `test.py` 中使用到 `MF_dataset` 的地方

⭐ 然后正常地训练和测试就行了


### 任务三：写胶水脚本 infer_Dif-Fusion.py，重复整套融合-训练-测试过程

- 这个任务学习如何在不拷贝第三方库代码到自己的仓库(以免污染代码环境)的情况下，调用第三方库获得所需的结果
  - 阅读理解 `infer_MMIF-CDDFuse_IVF.py` 脚本后仿照完成 ;)
- 重复任务一和任务二，只不过 `MMIF-CDDFuse` 换成 `Dif-Fusion`


----
by Armit
2024年6月10日
