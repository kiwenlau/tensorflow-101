## TensorFlow入门教程

#### Docker相关命令

**构建镜像**

```bash
sudo docker build -t tensorflow .
```

**运行容器**

```bash
sudo docker run -i tensorflow python src/main.py
```

#### 运行结果

```
训练:
Epoch 000: Loss: 1.142, Accuracy: 29.167%
Epoch 050: Loss: 0.569, Accuracy: 78.333%
Epoch 100: Loss: 0.304, Accuracy: 95.833%
Epoch 150: Loss: 0.186, Accuracy: 97.500%
Epoch 200: Loss: 0.134, Accuracy: 98.333%

测试:
Test set accuracy: 96.667%

结果:
Example 0 prediction: Iris setosa
Example 1 prediction: Iris versicolor
Example 2 prediction: Iris virginica
```
