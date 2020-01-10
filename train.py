import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.data_process import ImageAug, DeformAug, ScaleAug, CutOut, MyData, ToTensor
from utils.metric import compute_iou
import utils.configs as cf
import os
from Deeplabv3plus_model.model.deeplabv3plus import DeepLabv3Plus
from utils.loss import MySoftmaxCrossEntropyLoss
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val  # 每个batch的正确率
        self.sum += val * n  # 每个batch正确的个数
        self.count += n  # 总的个数
        self.avg = self.sum / self.count  # 总的正确率


def train(epoch, dataloader, model, criterion, optimizer):
    params = AverageMeter()
    model.train()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataloader)
    for step, sample in enumerate(dataprocess):
        inputs = sample['image']
        labels = sample['label']
        inputs = inputs.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs = inputs.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        total_mask_loss += loss
        loss.backward()
        optimizer.step()
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(loss.item()))


def valid(epoch, dataloader, model, criterion, optimizer):
    model.eval()
    total_mask_loss =0.0
    data_process = tqdm(dataloader)
    result = {"TP": {i: 0 for i in range(8)}, "TA": {i: 0 for i in range(8)}}
    for step, sample in enumerate(data_process):
        img = sample['image']
        mask = sample['label']
        img = img.type(torch.FloatTensor)
        if torch.cuda.is_available():
            img = img.to(device)
            mask = mask.to(device)
        output = model(img)
        loss = criterion(output, mask.long())
        total_mask_loss += loss.detach().item()
        pre = torch.argmax(func.softmax(output, dim=1), dim=1)
        result = compute_iou(pre, mask, result)
        data_process.set_description_str("epoch:{}".format(epoch))
        data_process.set_postfix_str("mask_loss:{:.4f}".format(loss.item()))


# def test():


def main_gen():
    # 1. 数据生成器
    train_dataset = MyData(cf.params['root_dir'], cf.params['train_csv'], transforms=transforms.Compose([ImageAug(),
                                                                        ScaleAug(), CutOut(64, 0.5), ToTensor()]))
    val_dataset = MyData(cf.params['root_dir'], cf.params['val_csv'], transforms=transforms.Compose([ToTensor()]))
    test_dataset = MyData(cf.params['root_dir'], cf.params['test_csv'], transforms=transforms.Compose([ToTensor()]))

    #  2. 数据加载器
    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    kwargs = {}
    train_data_batch = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=1)
    val_data_batch = DataLoader(val_dataset, batch_size=1)
    test_data_batch = DataLoader(test_dataset, batch_size=1)

    #   3. 模型加载
    #lane_config = Config()
    model = DeepLabv3Plus()

    #   4. 损失函数、优化器
    criterion = MySoftmaxCrossEntropyLoss(cf.params['num_class'])
    optimizer = torch.optim.SGD(model.parameters(), lr=cf.params['learning_rate'], momentum=0.9)

    # GPU
    if torch.cuda.is_available():
        model = model.to(device)
        criterion = criterion.to(device)

    for epoch in range(cf.params['epochs']):
        train(epoch, train_data_batch, model, criterion, optimizer)
        # print('loss:\t', loss)
        valid(epoch, val_data_batch, model, criterion, optimizer)
        # test()
        # if epoch % 10 == 0:
        #    torch.save({'state_dict': model.state_dict()}, os.path.join(cf.params['model_save_path'],
        #                                                               'logs','laneNet{}.pth.tar'.format(epoch)))
    torch.save({'state_dict': model.state_dict()}, os.path.join(cf.params['model_save_path'], 'finalNet.path'))


if __name__ == '__main__':
    main_gen()


