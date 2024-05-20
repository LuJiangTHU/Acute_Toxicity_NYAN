import os.path
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import pandas as pd
import torchnet as tnt
from scipy.stats import pearsonr

TARGET_PATH = './data/dataset.txt'
FEATURE_PATH = './data/avalon_bits.txt'
ALL_FEATURE_PATH = './data/all_descriptors.txt'
TRN_SPLIT_PATH = './data/random split/train_fold_0.txt'
TST_SPLIT_PATH = './data/random split/test_fold_0.txt'
SPLIT_DATASET_PATH = './data/random split'
MERGE_DATASET_PATH = os.path.join('./data', SPLIT_DATASET_PATH.split('/')[-1]+'-merge')
if (not os.path.isdir(MERGE_DATASET_PATH)):
    os.mkdir(MERGE_DATASET_PATH)
# FEATURE_NAME_LIST = ['Avalon', 'Morgan', 'AtomPair', 'NYAN', 'Avalon + NYAN']



def split_train_test_data(all_feature_file_name = ALL_FEATURE_PATH,
                          feature_name = 'Avalon',
                          target_file_name =TARGET_PATH,
                          trn_split_file_name = TRN_SPLIT_PATH,
                          tst_split_file_name = TST_SPLIT_PATH
                         ):


    fea_dir = os.path.join(MERGE_DATASET_PATH, feature_name)
    if (not os.path.isdir(fea_dir)):
        os.mkdir(fea_dir)

    out_trn_split_file_name = os.path.join(fea_dir,
                                           trn_split_file_name.split('/')[-1])
    out_tst_split_file_name = os.path.join(fea_dir,
                                           tst_split_file_name.split('/')[-1])


    label_df = pd.read_csv(target_file_name, dtype={'RTECS_ID': str})

    if feature_name in ['Avalon', 'Morgan', 'AtomPair']:
        all_feature_df = pd.read_csv(all_feature_file_name, dtype={'RTECS_ID': str})

        # 'Avalon', 'Morgan', 'AtomPair'
        column_name = ['RTECS_ID', ] + [feature_name + '_Bit ' + str(i) for i in range(1, 1025)]
        feature_df = all_feature_df[column_name]
    elif feature_name == 'Avalon + NYAN0':
        all_feature_df = pd.read_csv(all_feature_file_name, dtype={'RTECS_ID': str})
        column_name = ['RTECS_ID', ] + ['Avalon_Bit ' + str(i) for i in range(1, 1025)]
        avalon_feature_df = all_feature_df[column_name]
        NYAN_feature_df = pd.read_csv('./data/NYAN_latent0.txt', dtype={'RTECS_ID': str})
        feature_df = pd.merge(avalon_feature_df, NYAN_feature_df, on='RTECS_ID')   #合并Avalon和NYAN
    elif feature_name == 'Avalon + NYAN1':
        all_feature_df = pd.read_csv(all_feature_file_name, dtype={'RTECS_ID': str})
        column_name = ['RTECS_ID', ] + ['Avalon_Bit ' + str(i) for i in range(1, 1025)]
        avalon_feature_df = all_feature_df[column_name]
        NYAN_feature_df = pd.read_csv('./data/NYAN_latent1.txt', dtype={'RTECS_ID': str})
        feature_df = pd.merge(avalon_feature_df, NYAN_feature_df, on='RTECS_ID')  # 合并Avalon和NYAN
    elif feature_name == 'Avalon + NYAN2':
        all_feature_df = pd.read_csv(all_feature_file_name, dtype={'RTECS_ID': str})
        column_name = ['RTECS_ID', ] + ['Avalon_Bit ' + str(i) for i in range(1, 1025)]
        avalon_feature_df = all_feature_df[column_name]
        NYAN_feature_df = pd.read_csv('./data/NYAN_latent2.txt', dtype={'RTECS_ID': str})
        feature_df = pd.merge(avalon_feature_df, NYAN_feature_df, on='RTECS_ID')   #合并Avalon和NYAN
    elif feature_name == 'Avalon + NYAN3':
        all_feature_df = pd.read_csv(all_feature_file_name, dtype={'RTECS_ID': str})
        column_name = ['RTECS_ID', ] + ['Avalon_Bit ' + str(i) for i in range(1, 1025)]
        avalon_feature_df = all_feature_df[column_name]
        NYAN_feature_df = pd.read_csv('./data/NYAN_latent3.txt', dtype={'RTECS_ID': str})
        feature_df = pd.merge(avalon_feature_df, NYAN_feature_df, on='RTECS_ID')   #合并Avalon和NYAN
    elif feature_name == 'Avalon + NYAN4':
        all_feature_df = pd.read_csv(all_feature_file_name, dtype={'RTECS_ID': str})
        column_name = ['RTECS_ID', ] + ['Avalon_Bit ' + str(i) for i in range(1, 1025)]
        avalon_feature_df = all_feature_df[column_name]
        NYAN_feature_df = pd.read_csv('./data/NYAN_latent4.txt', dtype={'RTECS_ID': str})
        feature_df = pd.merge(avalon_feature_df, NYAN_feature_df, on='RTECS_ID')   #合并Avalon和NYAN
    else:
        feature_df = pd.read_csv('./data/' + feature_name +'.txt', dtype={'RTECS_ID': str})



    print('According to {0} and {1}, spliting the toxicity ground-truth data {2} and the feature {4} in molecular'
          'feature dataset {3}...'.format(
        trn_split_file_name,tst_split_file_name,target_file_name,all_feature_file_name,feature_name
    ))
    # 80081 * (1+1+59+1024+64)
    data_df = pd.merge(label_df, feature_df, on='RTECS_ID')
    trn_df = pd.read_csv(trn_split_file_name, dtype={'RTECS_ID': str})
    tst_df = pd.read_csv(tst_split_file_name, dtype={'RTECS_ID': str})

    # join with data
    trn_df = pd.merge(trn_df, data_df, on='RTECS_ID')  #(num_trn, 1+1+59+1024)
    tst_df = pd.merge(tst_df, data_df, on='RTECS_ID')  #(num_tst, 1+1+59+1024)



    trn_df.to_csv(out_trn_split_file_name, index=False)
    tst_df.to_csv(out_tst_split_file_name, index=False)

    print('Dataset preprocessing over!')
    print('The train set and test set combining toxicity ground-truth value and molecular features {2} '
          'have been saved into {0} and {1}。'.format(
        out_trn_split_file_name, out_tst_split_file_name, feature_name))


def endpoint_decomposition(target_file_name = TARGET_PATH):
    df = pd.read_csv(target_file_name)
    endpoints = list(df.columns.values[2:])
    endpoints_split = np.array([s.split('_')[:3] for s in endpoints])
    attribute={}
    attribute['species'] = set(endpoints_split[:,0])
    num_species = len(attribute['species'])
    attribute['routes'] = set(endpoints_split[:,1])
    num_routes = len(attribute['routes'])
    attribute['records'] = set(endpoints_split[:, 2])
    num_records = len(attribute['records'])
    print(f'--- {len(endpoints)} kinds of End Points ---')
    print(f'--- {num_species} kinds of species:',attribute['species'])
    print(f'--- {num_routes} kinds of routes:', attribute['routes'])
    print(f'--- {num_records} kinds of records:', attribute['records'])
    return endpoints_split



class ToxicityDataset(Dataset):
    def __init__(self,
                 phase='train',
                 dataset_file='train_fold_0.txt',
                 feature_name= 'Avalon'
                 ):

        assert (phase == 'train' or phase == 'test')
        assert (phase == dataset_file.split('_')[0])

        self.phase = phase
        self.name = dataset_file
        self.feature_name = feature_name
        self.pearson_threshold = 0.2
        self.num_shared_threshold = 10

        dataset_file_path = os.path.join(MERGE_DATASET_PATH, self.feature_name, dataset_file)

        print(f'Checking if {self.phase} dataset exists...')
        if not os.path.exists(dataset_file_path):
            if self.phase == 'train':
                trnset_path = os.path.join(SPLIT_DATASET_PATH, dataset_file)
                tstset_path = os.path.join(SPLIT_DATASET_PATH, dataset_file.replace('train','test'))
                split_train_test_data( feature_name = self.feature_name,
                                       trn_split_file_name = trnset_path,
                                       tst_split_file_name = tstset_path)
            else:
                trnset_path = os.path.join(SPLIT_DATASET_PATH, dataset_file.replace('test','train'))
                tstset_path = os.path.join(SPLIT_DATASET_PATH, dataset_file)
                split_train_test_data( feature_name = self.feature_name,
                                       trn_split_file_name = trnset_path,
                                       tst_split_file_name = tstset_path)
        else:
            print(f'{self.phase}dataset is already!')

        print(f'loading it into memory...')
        df = pd.read_csv(dataset_file_path, dtype={'RTECS_ID': str})

        self.endpoint_name_list = list(df.columns.values[2:61])
        self.dataset = df.values

        target_nan = self.dataset[:, 2:61].astype('float32')  # missing value is nan

        # input feature，[num_samples, 1024+64], float
        self.feature = torch.tensor(self.dataset[:, 61:].astype('float32'),requires_grad=False)

        # nan->0  [num_samples, 59], float
        self.target = torch.tensor(np.nan_to_num(target_nan), requires_grad=False)

        # #mask matrix，[num_samples, 59]，int
        self.target_mask = torch.tensor(1 - np.isnan(target_nan), requires_grad=False)
        print(f'Loading Over!')




    def __getitem__(self, index):
        return self.feature[index], self.target[index], self.target_mask[index]

    def __len__(self):
        return len(self.target)


class ToxicityDataloader():
    def __init__(self, dataset, batch_size = 32, num_workers = 0, epoch_size = 80):
        super(ToxicityDataloader, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch_size =epoch_size
        self.phase = self.dataset.phase
        self.num_workers = num_workers

    def get_iterator(self, epoch):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        dataloader = DataLoader(dataset = self.dataset,
                                batch_size = self.batch_size,
                                shuffle = (True if self.phase == 'train' else False),
                                num_workers =(0 if self.phase == 'test' else self.num_workers),
                                drop_last = False
                                )
        return dataloader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size





class FastConfusionMeter(object):
    def __init__(self, k, normalized=False):
        # super(FastConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, output, target):
        output = output.cpu().squeeze().numpy()
        target = target.cpu().squeeze().numpy()

        if np.ndim(output) == 1:
            output = output[None]

        onehot = np.ndim(target) != 1
        assert output.shape[0] == target.shape[0], \
            'number of targets and outputs do not match'
        assert output.shape[1] == self.conf.shape[0], \
            'number of outputs does not match size of confusion matrix'
        assert not onehot or target.shape[1] == output.shape[1], \
            'target should be 1D Tensor or have size of output (one-hot)'
        if onehot:
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'

        target = target.argmax(1) if onehot else target
        pred = output.argmax(1)

        target = target.astype(np.int32)
        pred = pred.astype(np.int32)
        conf_this = np.bincount(target * self.conf.shape[0] + pred, minlength=np.prod(self.conf.shape))
        conf_this = conf_this.astype(self.conf.dtype).reshape(self.conf.shape)
        self.conf += conf_this

    def value(self):
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


def getConfMatrixResults(matrix):
    assert (len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1])

    count_correct = np.diag(matrix)
    count_preds = matrix.sum(1)
    count_gts = matrix.sum(0)
    epsilon = np.finfo(np.float32).eps
    accuracies = count_correct / (count_gts + epsilon)
    IoUs = count_correct / (count_gts + count_preds - count_correct + epsilon)
    totAccuracy = count_correct.sum() / (matrix.sum() + epsilon)

    num_valid = (count_gts > 0).sum()
    meanAccuracy = accuracies.sum() / (num_valid + epsilon)
    meanIoU = IoUs.sum() / (num_valid + epsilon)

    result = {'totAccuracy': round(totAccuracy, 4), 'meanAccuracy': round(meanAccuracy, 4),
              'meanIoU': round(meanIoU, 4)}
    if num_valid == 2:
        result['IoUs_bg'] = round(IoUs[0], 4)
        result['IoUs_fg'] = round(IoUs[1], 4)

    return result


class AverageConfMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = np.asarray(0, dtype=np.float64)
        self.avg = np.asarray(0, dtype=np.float64)
        self.sum = np.asarray(0, dtype=np.float64)
        self.count = 0

    def update(self, val):
        self.val = val
        if self.count == 0:
            self.sum = val.copy().astype(np.float64)
        else:
            self.sum += val.astype(np.float64)

        self.count += 1
        self.avg = getConfMatrixResults(self.sum)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += float(val * n)
        self.count += n
        self.avg = round(self.sum / self.count,4)


class LAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = []
        self.avg = []
        self.sum = []
        self.count = 0

    def update(self, val):
        self.val = val
        self.count += 1
        if len(self.sum) == 0:
            assert(self.count == 1)
            self.sum = [v for v in val]
            self.avg = [round(v,4) for v in val]
        else:
            assert(len(self.sum) == len(val))
            for i, v in enumerate(val):
                self.sum[i] += v
                self.avg[i] = round(self.sum[i] / self.count,4)


class DAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = {}

    def update(self, values):
        assert (isinstance(values, dict))
        for key, val in values.items():
            if isinstance(val, (float, int)):
                if not (key in self.values):
                    self.values[key] = AverageMeter()
                self.values[key].update(val)
            elif isinstance(val, (tnt.meter.ConfusionMeter, FastConfusionMeter)):
                if not (key in self.values):
                    self.values[key] = AverageConfMeter()
                self.values[key].update(val.value())
            elif isinstance(val, AverageConfMeter):
                if not (key in self.values):
                    self.values[key] = AverageConfMeter()
                self.values[key].update(val.sum)
            elif isinstance(val, dict):
                if not (key in self.values):
                    self.values[key] = DAverageMeter()
                self.values[key].update(val)
            elif isinstance(val, list):
                if not (key in self.values):
                    self.values[key] = LAverageMeter()
                self.values[key].update(val)

    def average(self):
        average = {}
        for key, val in self.values.items():
            if isinstance(val, type(self)):
                average[key] = val.average()
            else:
                average[key] = val.avg

        return average

    def __str__(self):
        ave_stats = self.average()
        return ave_stats.__str__()



def regression_loss(pred, target, target_mask):
    '''
    :param pred: (batch_size, num_tasks), float
    :param target: (batch_size, num_tasks), float
    :param target_mask: (batch_size, num_tasks), 0 or 1
    :return: MSE averaged on batch
    '''
    mse = torch.mul((pred - target).pow(2), target_mask)
    loss = mse.sum()/target_mask.sum()
    return loss


def calculate_RMSE( pred, target, target_mask):
    mse = torch.mul((pred - target).pow(2), target_mask)  # (num_samples, num_tasks)
    RMSE = torch.sqrt(mse.sum(0)/target_mask.sum(0))
    return RMSE, RMSE.mean()



def calculate_R2(pred, target, target_mask):
    mse = torch.mul((pred - target).pow(2), target_mask)  #(num_samples, num_tasks)
    target_avg = target.sum(0)/target_mask.sum(0)   #(num_tasks,)
    mse_avg = torch.mul((target - target_avg).pow(2), target_mask)  #(num_samples, num_tasks)

    R2 = 1 - mse.sum(0)/mse_avg.sum(0)
    return R2, R2.mean()
