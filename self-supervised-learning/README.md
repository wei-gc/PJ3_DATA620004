# 自监督预训练：MoCo
## 该项目的实现主要是来自[源项目](https://github.com/facebookresearch/moco)，实现了以下需求：
* 在ImageNet-32*32数据集上用MoCo完成自监督预训练；
* 在CIFAR100数据集上用Linear Classification Protocol检验自监督能力；
* 使用tensorboard可视化loss曲线、acc@1、acc@5曲线

## 环境配置
```
torch
torchvision
tqdm
pillow
```
需要多张卡，代码为分布式训练

## 数据集配置
下载 [ImageNet-32*32](https://image-net.org/download-images.php) 数据集，
更改 `imagenet_process.py` 文件中的path目录，预处理数据集，使得该数据集能直接被torchvision.datasets.ImageFolder读取.

## 训练模型
```
python main_moco.py -a resnet18 
--lr 0.02 --batch-size 512 --dist-url 'tcp://localhost:10001' 
--multiprocessing-distributed --world-size 1 --rank 0 
--schedule 10 25 40 --epochs 50  'your/Imagenet32/path'
```
* 注意将 'your/Imagenet32/path' 替换为你的数据集目录
* 训练得到的ckpt可以在[checkpoint_0050.pth.tar](https://drive.google.com/file/d/1EdZ4sCJY7fCy5rIO-LyGwDRAD2A4sfGH/view)中下载

## Linear Classification Protocol检验
```
python main_lincls.py -a resnet18 -j 8 --lr 3 --batch-size 1024 
--pretrained your/checkpoint/path --dist-url 'tcp://localhost:10001' 
--multiprocessing-distributed --world-size 1 --rank 0 '~'
```
* 注意将 'your/Imagenet32/path' 替换为你的checkpoint地址，

