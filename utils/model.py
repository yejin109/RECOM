import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class BPR(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, regularization):
        super(BPR, self).__init__()
        self.regularization = regularization

        self.W = nn.Embedding(user_num, embedding_dim)
        nn.init.xavier_normal_(self.W.weight)
        self.H = nn.Embedding(item_num, embedding_dim)
        nn.init.xavier_normal_(self.H.weight)

        self.log_sig = nn.LogSigmoid()

    def get_r_hat(self):
        user_vec = self.W.weight.clone()
        item_vec = self.H.weight.clone()
        r_hat = user_vec @ item_vec.T
        return r_hat

    def forward(self, data):
        user_id = data[:, 0]
        positive_id = data[:, 1]
        negative_id = data[:, 2]

        user_vec = self.W(user_id)
        positive_vec = self.H(positive_id)
        negative_vec = self.H(negative_id)

        x_ui_hat = torch.sum(user_vec * positive_vec, dim=1)
        x_uj_hat = torch.sum(user_vec * negative_vec, dim=1)
        x_uij_hat = x_ui_hat - x_uj_hat

        bpr = - self.log_sig(x_uij_hat).sum()
        regularization = self.regularization['user'] * (torch.norm(user_vec, dim=1) ** 2).sum() \
                         + self.regularization['positive'] * (torch.norm(positive_vec, dim=1) ** 2).sum() \
                         + self.regularization['negative'] * (torch.norm(negative_vec, dim=1) ** 2).sum()
        loss = bpr + regularization
        return loss
