#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @IDE          :PyCharm
# @Project      :PytorchDemo_LetNet
# @FileName     :Trainer
# @CreatTime    :2022/5/9 21:29 
# @CreatUser    :DaneSun

'''
Trainer
'''
import datetime

import torch
from matplotlib import pyplot as plt

from PytorchTrainer.Exception.DataSetTypeError import DataSetTypeError


class Trainer:
    def __init__(self, model, train_loader, test_loader, classes, optimizer, loss_func = torch.nn.CrossEntropyLoss(), device = 'cpu'):
        self.model = model
        self.classes = classes
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device
        self.set_train_loader(train_loader)
        self.set_test_loader(test_loader)

    def set_train_loader(self,train_loader):
        if type(train_loader) is not torch.utils.data.DataLoader:
            raise DataSetTypeError('train_loader must be a torch.utils.data.DataLoader')
        else:
            self.train_loader = train_loader

    def set_test_loader(self,test_loader):
        if type(test_loader) is not torch.utils.data.DataLoader:
            raise DataSetTypeError('test_loader must be a torch.utils.data.DataLoader')
        else:
            # 将测试集加载器转换为迭代器
            self.test_loader = test_loader
            test_image, test_label = iter(test_loader).next()
            self.test_image, self.test_label = test_image.to(self.device), test_label.to(self.device)
    # 输出函数
    def __console_log__(self, epoch, train_loss, test_loss,acc):
        # 控制台红色字体
        print('\033[1;31m',end='')
        print('[{}] Epoch: {}, Train_Loss: {}, Test_Loss: {}, Acc: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),epoch, train_loss, test_loss, acc))
        print('\033[0m',end='')
    # 绘制损失折线图
    def draw_loss(self,train_loss,test_loss):
        plt.plot(train_loss,label='train_loss')
        plt.plot(test_loss,label='test_loss')
        plt.legend()
        plt.show()
    # 绘制准确率折线图
    def draw_accuracy(self,acc_list):
        plt.plot(acc_list,label='accuracy')
        plt.legend()
        plt.show()
    # 绘制全部图像
    def draw_all(self,train_loss,test_loss,acc_list):
        plt.figure(figsize=(10,10),dpi=100)
        plt.subplot(2,1,1)
        plt.plot(train_loss,label='train_loss')
        plt.plot(test_loss,label='test_loss')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(acc_list,label='accuracy')
        plt.legend()
        plt.show()

    # 训练函数
    def train(self, epochs, save_path = './models/', save_interval = None):
        print('\033[1;31m',end='')
        print('[{}] Start Training'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print('\033[0m',end='')

        train_loss = []
        test_loss = []
        acc_list = []
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader):
                self.model.train()

                # 获取输入数据
                inputs, labels = data
                # 转换设备
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # 梯度清零
                self.optimizer.zero_grad()
                # 预测
                outputs = self.model(inputs)
                # 计算损失
                loss = self.loss_func(outputs, labels)
                running_loss += loss.item()

                # 反向传播
                loss.backward()
                # 更新参数
                self.optimizer.step()
            with torch.no_grad():
                self.model.eval()
                test_outputs = self.model(self.test_image)
                predict_lable = torch.max(test_outputs, 1)[1]
                acc = (predict_lable == self.test_label).sum().item() / self.test_label.size(0) # 计算准确率
                acc_list.append(acc)
                test_loss.append(self.loss_func(test_outputs, self.test_label).item())


            # 输出结果
            self.__console_log__(epoch + 1, running_loss / len(self.train_loader), test_loss[-1], acc)

            train_loss.append(running_loss / len(self.train_loader))

            # 保存模型
            if save_interval is not None and epoch % save_interval == 0:
                torch.save(self.model.state_dict(), save_path +  'model_' + str(epoch) + '.pth')
        # 保存最后一个模型
        torch.save(self.model.state_dict(), save_path +  'model_final'+ '.pth')

        # 绘制全部图像
        self.draw_all(train_loss,test_loss,acc_list)

        print('[{}] Finished Training'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

if __name__ == '__main__':
    t = Trainer(1, 2, 3, 4, 5, 6)
    t.__console_log__(1, 2, 3)