import numpy as np


class Evaluator:
    def __init__(self, ground_truth):
        super(Evaluator, self).__init__()
        # Te_{u}^{+}
        self.ground_truth = dict()
        self.target_user = np.unique(ground_truth[:, 0])

        for user_idx in self.target_user:
            user_item = ground_truth[:, 0] == user_idx
            self.ground_truth[user_idx] = ground_truth[user_item, 1]

    def precision_recall(self, recoms):
        precisions = []
        recalls = []
        for user_idx in self.target_user:
            pred = recoms[user_idx]
            accurate_preds = np.intersect1d(self.ground_truth[user_idx], pred)
            precisions.append(len(accurate_preds)/len(pred))
            recalls.append(len(accurate_preds) / len(self.ground_truth[user_idx]))
        print(f'Precision : {np.mean(precisions):.5f}')
        print(f'Recall : {np.mean(recalls):.5f}')

    def ndcg(self, recoms):
        score = np.arange(1, recoms.shape[1]+1)
        score = 1/np.log2(score+1)
        ndcg = []
        for user_idx in self.target_user:
            # i_{k}
            pred = recoms[user_idx]

            # i_{k} \in Te_{u}^{+}
            filt = np.where(np.isin(pred, self.ground_truth[user_idx]), 1, 0)

            ideal_num = np.min([len(self.ground_truth[user_idx]), len(pred)])

            dcg = np.sum(score*filt)
            idcg = np.sum(score[:ideal_num])

            ndcg.append(dcg/idcg)

        print(f'NDCG : {np.mean(ndcg): .5f}')
