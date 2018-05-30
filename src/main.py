# -*- coding: utf-8 -*-

from train import train
from test import test
from predict import predict

# 使用数据训练模型
model = train()

# 测试模型的准确率
test(model)

# 使用模型进行预测
predict(model)
  