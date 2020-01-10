import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.data_process import ImageAug, DeformAug, ScaleAug, CutOut, MyData,ToTensor
import utils.configs as cf


def main_gen():
    dataset = MyData(cf.params['root_dir'], cf.params['train_csv'], transforms=transforms.Compose([ImageAug(),
                                                                    ScaleAug(), CutOut(32, 0.5)]))
    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    kwargs = {}
    train_data_batch = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True, num_workers=1)
    for i in range(1,2):
        image = dataset[i]['image']
        plt.figure('image')
        plt.imshow(image)
        plt.show()
'''
    for step, sample in enumerate(train_data_batch):
        img = sample['image']
        mask = sample['label']
        print(step, img.size())
        print(step, mask.size())
        if step == 10:
            break
'''

if __name__ == '__main__':
    main_gen()


