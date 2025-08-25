# AFSC + ResCNN-ASP 声纹识别系统

本项目实现了基于自适应频率频谱系数（AFSC）特征提取和ResCNN-ASP深度网络的声纹识别系统。系统使用三元组损失函数进行训练，能够有效提取说话人的声纹特征并进行身份验证。

## 1. 主要特点

- 🎙️ 创新的自适应频率频谱系数(AFSC)特征提取方法。
-  📊 数据驱动的自适应尺度变换器(AST-Net)。
- 🧠 结合残差卷积网络(ResCNN)和注意力统计池化(ASP)的深度模型。
- 🔥 端到端的三元组损失训练框架。
- ⚡ 支持GPU加速训练和推理。

## 2. 项目介绍

1. `myconfig.py`: 全局配置文件。
2. `dataset`: 数据集的加载与处理。
3. `generate_csv.py`: 生成CSV数据。
4. `features_preprocessing.py`: 语音特征预处理。
5. `AST_net.py`:自适应尺度变换网络。
6. `ResCNN_ASP_net.py`: ResCNN-ASP网络模型。
7. `neural_net.py`: 对照组神经网络模型。
8. `evaluation.py`: 模型评估。
9. `ResCNN_ASP_evaluation.py`: ResCNN-ASP模型评估。
10. `specaug.py`: 频谱增强算法(选用)。
11. `AFSC+ResCNN.pt`: AFSC的预训练模型。
12. `MFCC+ResCNN.pt`: MFCC的预训练模型。

## 3. 安装环境
### 3.1 安装PyTorch
#### CUDA 11.3
```bash
pip install 
torch==1.13.1+cu113 
torchvision==0.13.1+cu113
-f https://download.pytorch.org/whl/torch_stable.html
```
#### CPU版本
```shell
pip install torch==1.13.1 
torchvision==0.13.1
```
### 3.2 安装音频处理库

```bash
pip install librosa==0.10.1 
soundfile==0.12.1 
python-speech-features==0.6
```

### 安装其他依赖

```bash
pip install -r requirements.txt
```


## 4. 准备数据

### 4.1 数据集介绍

本项目使用的是​**LibriSpeech corpus**​ 数据集，这是由约翰霍普金斯大学于2015年公开的大规模英语语音数据集，包含1000小时的16kHz朗读英语语音。

### 4.2 数据集下载

```bash
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
```

## 5. 训练与评估模型

### 5.1 训练阶段

调整`myconfig`中的数据路径、模型选择、模型参数、特征参数、训练次数设置等等，请看代码中的详细备注。

训练 AST 网络获得自适应参数。
训练Neural net和ResCNN-ASP网络参数。
```bash
python AST_net.py
python neural_net.py
python ResCNN_ASP_net
```
若使用预训练模型可以跳过训练模型阶段，选择并将预训练模型到`save/`目录接口。

### 5.2 评估模型

```bash
python evaluation.py
python ResCNN_ASP_evaluation.py
```

评估完成后会输出以下指标：

**训练loss图**: 记录每一次loss的
​**训练时间​**: 数值越小越好
​**EER (等错误率)​**: 数值越小越好
​**Accuracy**: 分类准确率
**DET曲线**: 保存为`paper/DET_curve.png`

## 6. 总结


本项目创新性地采用**自适应频率频谱系数(AFSC)**替代传统MFCC特征提取方法，通过动态学习滤波器组参数实现说话人特征的自适应提取，显著提升了特征的表征能力。在模型架构方面，我们创造性地结合了**ResCNN残差连接与注意力统计池化(ASP)技术**，并引入帧级注意力机制，有效增强了模型对说话人特征的判别能力。整个系统构建了完整的端到端训练框架，支持多GPU并行训练和混合精度训练选项，通过优化的三元组损失函数实现高效稳定的模型训练。
