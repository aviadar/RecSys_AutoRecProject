import numpy as np
import math


class Metrics:
    THRESHOLD = 3

    @staticmethod
    def RMSE(y_true, y_pred, lower_bound=1, upper_bound=5):
        y_pred[y_pred < lower_bound] = lower_bound
        y_pred[y_pred > upper_bound] = upper_bound
        rel_preds = (y_true >= lower_bound) & (y_true <= upper_bound)
        return np.sqrt(((rel_preds * (y_true - y_pred)) ** 2).sum() / rel_preds.sum())

    @staticmethod
    def MRR(df_true, df_pred, lower_bound=1, upper_bound=5, top_n=5, threshold_val=THRESHOLD):
        # please use MRR_for_user
        user_mrr = []
        for i in range(df_true.shape[0]):
            val = Metrics.MRR_for_user(df_true[i, :], df_pred[i, :], lower_bound, upper_bound, top_n, threshold_val)
            if val is not None:
                user_mrr.append(val)
        return np.sum(user_mrr) / len(user_mrr)

    @staticmethod
    def MRR_for_user(user_true, user_pred, lower_bound, upper_bound, top_n, threshold_val):
        rel_preds = (user_true >= lower_bound) & (user_true <= upper_bound)
        if rel_preds.sum() == 0:
            return None
        screened_user_true = user_true[rel_preds]
        screened_user_pred = user_pred[rel_preds]

        top_n_pred_indicies = screened_user_pred.argsort()[::-1][:top_n]
        for i, (pred, actual) in enumerate(
                zip(screened_user_pred[top_n_pred_indicies], screened_user_true[top_n_pred_indicies])):
            if actual >= threshold_val:
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def NDCG(df_true, df_pred, lower_bound=1, upper_bound=5, top_n=5):
        # please use NDCG_for_user
        user_ndcg = []
        for i in range(df_true.shape[0]):
            user_ndcg.append(Metrics.NDCG_for_user(df_true[i], df_pred[i], lower_bound, upper_bound, top_n))

        user_ndcg = [x for x in user_ndcg if math.isnan(x) == False]

        return np.sum(user_ndcg) / len(user_ndcg)

    @staticmethod
    def NDCG_for_user(user_true, user_pred, lower_bound=1, upper_bound=5, top_n=5):
        # please use DCG function
        rel_preds = (user_true >= lower_bound) & (user_true <= upper_bound)
        if rel_preds.astype(bool).sum() < top_n:
            return np.nan
        screened_user_true = user_true[rel_preds]
        screened_user_pred = user_pred[rel_preds]
        top_n_pred_indicies = screened_user_pred.argsort()[::-1][:top_n]
        user_dcg = Metrics.DCG(screened_user_true[top_n_pred_indicies])
        user_idcg = Metrics.DCG(np.sort(screened_user_true[top_n_pred_indicies])[::-1])
        return user_dcg / user_idcg

    @staticmethod
    def DCG(rel):
        # please implement the DCG formula
        dcg = rel[0]
        for i in range(1, len(rel)):
            dcg += rel[i] / np.log2(i + 2)

        return dcg

    @staticmethod
    def get_evaluation(df_true, df_pred):
        return {'RMSE': Metrics.RMSE(df_true, df_pred, ),
                'MRR_5': Metrics.MRR(df_true, df_pred, top_n=5),
                'MRR_10': Metrics.MRR(df_true, df_pred, top_n=10),
                'NDCG_5': Metrics.NDCG(df_true, df_pred, top_n=5),
                'NDCG_10': Metrics.NDCG(df_true, df_pred, top_n=10),
                'NDCG_100': Metrics.NDCG(df_true, df_pred, top_n=100)}
