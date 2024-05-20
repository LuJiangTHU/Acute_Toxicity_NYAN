from Algorithm.Algorithm import Algorithm
import torch
from utils import calculate_RMSE,calculate_R2

class MultiTaskLearning(Algorithm):
    def __init__(self, opt):
        # super(MultiTaskLearning, self).__init__(opt)
        Algorithm.__init__(self, opt)
        self.keep_best_model_metric_name = 'R2'
        self.max_best = True

    def allocate_tensors(self):
        self.tensor = {}
        self.tensor['feature'] = torch.FloatTensor()
        self.tensor['target'] = torch.FloatTensor()
        self.tensor['target_mask'] = torch.LongTensor()

    def set_tensors(self, batch):
        feature, target, target_mask = batch
        self.tensor['feature'].resize_(feature.size()).copy_(feature)
        self.tensor['target'].resize_(target.size()).copy_(target)
        self.tensor['target_mask'].resize_(target_mask.size()).copy_(target_mask)


    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)


    def process_batch(self, batch, do_train=True):

        self.set_tensors(batch)
        feature = self.tensor['feature']  # [batchsize, feature_dim]
        target = self.tensor['target']     # [batchsize, num_tasks]
        target_mask = self.tensor['target_mask']  # [batchsize, num_tasks]

        DNN = self.learners['DNN']
        Regressor = self.learners['Regressor']

        if do_train:
            self.optimizers['DNN'].zero_grad()
            self.optimizers['Regressor'].zero_grad()

        record = {}

        # forward
        final_fea = DNN(feature)
        pred = Regressor(final_fea)   # (batchsize, num_tasks)

        if do_train:
            loss = Regressor.regression_loss(pred=pred, target=target, target_mask=target_mask)
            record['MSE'] = loss.cpu().item()
            loss.backward()
            self.optimizers['DNN'].step()
            self.optimizers['Regressor'].step()


        if not do_train:
            _, RMSE_avg = calculate_RMSE(pred=pred, target=target, target_mask=target_mask)
            _, R2_avg = calculate_R2(pred=pred, target=target, target_mask=target_mask)
            record['RMSE'] = RMSE_avg.cpu().item()
            record['R2'] = R2_avg.cpu().item()


        return record











