'''
This script is used to evaluate the 5 cross validation R2 using single NYAN or using
Avalon + NYAN as input features
Run on sever lyt
'''

import argparse
import importlib as imp
import csv

import numpy as np

from utils import *
from MultiTaskLearning import MultiTaskLearning


df = pd.read_csv('./data/dataset.txt')
endpoint_name_list = list(df.columns.values[2:])


parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, default=2,
                    help = 'number of data loading workers')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--disp_step', type=int, default=200, help = 'display step during training')
args_opt = parser.parse_args()

num_fold = 5
num_NYAN_kind = 5


RMSE_5CV_list = []
RMSE_avg_5CV_list =[]
R2_5CV_list = []
R2_avg_5CV_list =[]


for fold in range(num_fold):

    # # Single NYAN
    # exp_config_file = [os.path.join('.','config','cfg_NYAN_fold{0}_lat{1}.py'.format(fold, i)) for i in range(num_NYAN_kind)]
    # exp_directory = [os.path.join('.','experiments', 'cfg_NYAN_fold{0}_lat{1}'.format(fold, i)) for i in range(num_NYAN_kind)]

    ## Avalon + NYAN
    exp_config_file = [os.path.join('.','config','cfg_Avalon+NYAN_fold{0}_lat{1}.py'.format(fold,i)) for i in range(num_NYAN_kind)]
    exp_directory = [os.path.join('.','experiments', 'cfg_Avalon+NYAN_fold{0}_lat{1}'.format(fold,i)) for i in range(num_NYAN_kind)]

    pred = []
    for i in range(0,num_NYAN_kind):  # 5 kinds of different NYAN_latent

        config = imp.machinery.SourceFileLoader('', exp_config_file[i]).load_module().config
        # config['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored
        config['disp_step'] = args_opt.disp_step
        config['exp_dir'] = exp_directory[i] # the place where logs, models, and other stuff will be stored


        feature_name = config['feature_name']
        print('本次实验选用{0}分子特征'.format(feature_name))

        # Set the train and test datasets, and their corresponding data loaders
        data_test_opt = config['data_test_opt']
        test_fold_file = config['test_fold_file']

        dataset_test = ToxicityDataset(phase='test',
                                    dataset_file=test_fold_file,
                                    feature_name=feature_name)

        Dataloader_test = ToxicityDataloader(
            dataset=dataset_test,
            batch_size=data_test_opt['batch_size'],
            num_workers=args_opt.num_workers,
            epoch_size=data_test_opt['epoch_size']
        )

        fea_tst = dataset_test[0:][0].cuda()
        tar_tst = dataset_test[0:][1].cuda()
        tar_mask_tst = dataset_test[0:][2].cuda()


        Alg = MultiTaskLearning(opt=config)

        if args_opt.cuda:
            Alg.load_to_gpu()

        Alg.load_checkpoint(epoch='*', train=False, suffix='.best')

        DNN = Alg.learners['DNN']
        Regressor = Alg.learners['Regressor']
        DNN.eval()
        Regressor.eval()

        final_fea_tst = DNN(fea_tst)
        pred.append(Regressor(final_fea_tst))   # (batchsize, 59)

    avg_pred = torch.stack(pred).mean(0)

    RMSE, RMSE_avg = calculate_RMSE(pred=avg_pred, target=tar_tst, target_mask=tar_mask_tst)
    R2, R2_avg = calculate_R2(pred=avg_pred, target=tar_tst, target_mask=tar_mask_tst)

    RMSE_5CV_list.append(RMSE)
    RMSE_avg_5CV_list.append(RMSE_avg)
    R2_5CV_list.append(R2)
    R2_avg_5CV_list.append(R2_avg)



RMSE_5CV = torch.stack(RMSE_5CV_list).mean(0).cpu().data.tolist()
R2_5CV = torch.stack(R2_5CV_list).mean(0).cpu().data.tolist()
RMSE_avg_5CV = torch.stack(RMSE_avg_5CV_list).mean().cpu().item()
R2_avg_5CV = torch.stack(R2_avg_5CV_list).mean().cpu().item()


RMSE_std = torch.stack(RMSE_5CV_list).std(1).cpu().data.tolist()
R2_std = torch.stack(R2_5CV_list).std(1).cpu().data.tolist()

# save the 5CV results
with open('./table_results/5CV_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    header = ['Metric', 'Fold_0', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Avg.']
    writer.writerow(header)
    writer.writerow(['RMSE'] + [v.cpu().item() for v in RMSE_avg_5CV_list] + [RMSE_avg_5CV] )
    writer.writerow(['RMSE_std'] + RMSE_std + [np.array(RMSE_std).mean()])

    writer.writerow(['R2'] +   [v.cpu().item() for v in R2_avg_5CV_list] + [R2_avg_5CV])
    writer.writerow(['R2_std'] + R2_std + [np.array(R2_std).mean()])
print('The 5CV avg. results have been saved to ./table_results/5CV_results.csv')


print('Avg_RMSE, Avg_R2:', RMSE_avg_5CV, R2_avg_5CV)



# save the performance into .csv  Avalon + NYAN
with open('./table_results/MT-NYAN.csv', 'w', newline='') as file:

    writer = csv.writer(file)
    header = ['Task', 'RMSE', 'R2']
    writer.writerow(header)
    for i,endpoint in enumerate(endpoint_name_list):
        writer.writerow([endpoint]+[RMSE_5CV[i], R2_5CV[i]])
    writer.writerow(['Avg.', RMSE_avg_5CV, R2_avg_5CV])

print('The 59 task results have been saved to ./table_results/MT-NYAN.csv')


