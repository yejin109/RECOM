import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data: pd.DataFrame, item_num, device, negative_samples, target_transform=None, transform=None):
        # I
        total_item = np.arange(item_num)

        # D_S
        uij = []

        for user in np.unique(data[:, 0]):
            user_ui = data[data[:, 0] == user]

            # I_{u}^{+}
            positive_data = user_ui[user_ui[:, 2] == 1, 1]

            # I\I_{u}^{+}
            negative_data = np.setdiff1d(total_item, positive_data)
            negative_data = negative_data[np.random.choice(negative_data.shape[0], negative_samples, replace=False)]

            user_uij = np.zeros((len(positive_data)*len(negative_data), 3))
            user_uij[:, 0] = user
            user_uij[:, 1] = np.repeat(positive_data, len(negative_data))
            user_uij[:, 2] = np.tile(negative_data, len(positive_data))
            uij.extend(user_uij)
        uij = np.array(uij)

        self.data = torch.LongTensor(uij).to(device)
        self.targets = torch.LongTensor(uij[:, 2]).to(device)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.targets.size()[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class DataLoad:
    def __init__(self, user_num, item_num, is_train=True):
        super(DataLoad, self).__init__()
        self.is_train = is_train
        self.user_num = user_num
        self.item_num = item_num

    def load_data(self, fold):
        if self.is_train:
            data_type = 'base'
        else:
            data_type = 'test'

        raw_data = pd.read_csv(f'data/ml-100k/u{fold}.{data_type}', '\t',
                               names=['user_id', 'item_id', 'rating', 'timestamp'],
                               engine='python')

        raw_data[['user_id', 'item_id']] -= 1
        raw_data['label'] = np.where(raw_data['rating'] == 5, 1, -1)
        raw_data.drop(['timestamp', 'rating'], axis=1,  inplace=True)

        return raw_data.values


