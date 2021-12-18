import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from RECOM.utils.model import BPR
from RECOM.utils.eval_func import Evaluator
from RECOM.utils.data_manage import DataLoad, CustomDataset

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
recom_num = 25

user_num = 943
item_num = 1682
embedding_dim = 32

batch_size = 8192
learning_rate = 1e-3
epochs = 20
regularization = {'user': 1e-2, 'positive': 1e-2, 'negative': 1e-2}
negative_samples = 5

data_manager = DataLoad(user_num, item_num)

model = BPR(user_num, item_num, embedding_dim, regularization).to(device)
optimizer = Adam(params=model.parameters(), lr=learning_rate)

fold_loss = []
for fold in range(1, 6):
    # Train
    data_manager.is_train = True
    train_raw_data = data_manager.load_data(fold)
    train_dataset = CustomDataset(train_raw_data, item_num, device, negative_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_loss_per_epoch = []
    for epoch in tqdm(range(epochs)):
        train_loss_per_iter = []

        for train_data, _ in train_dataloader:
            optimizer.zero_grad()
            loss = model(train_data)
            loss.backward()
            optimizer.step()
            train_loss_per_iter.append(loss.detach().cpu().numpy())
        train_loss_per_epoch.append(np.mean(train_loss_per_iter))
        if epoch+1 % 5 == 0:
            print(f'Epoch:{epoch}, Loss : {np.mean(train_loss_per_iter)}')
    fold_loss.extend(train_loss_per_epoch)

    # Evaluation
    data_manager.is_train = False
    val_raw_data = data_manager.load_data(fold)
    evaluator = Evaluator(val_raw_data[val_raw_data[:, 2] == 1])

    with torch.no_grad():
        r_hat = model.get_r_hat()
        user_oh = one_hot(torch.LongTensor(train_raw_data[:, 0]), num_classes=user_num).float().to(device)
        item_oh = one_hot(torch.LongTensor(train_raw_data[:, 1]), num_classes=item_num).float().to(device)
        indices = user_oh.T @ item_oh
        r_hat[indices == 1] = - math.inf

    recom_score, recom_label = torch.topk(r_hat, recom_num, sorted=True)
    evaluator.precision_recall(recom_label.detach().cpu().numpy())
    evaluator.ndcg(recom_label.detach().cpu().numpy())

plt.figure()
plt.plot(fold_loss)
plt.title('Train Loss')
plt.show()
