import os
import os.path
import glob
from tqdm import tqdm
import importlib as imp
import numpy as np

import torch.optim

from utils import *
import datetime
import logging


class Algorithm():
    def __init__(self, opt):
        self.set_experiment_dir(opt['exp_dir'])
        self.set_log_file_handler()
        self.logger.info('Algorithm options %s' % opt)
        self.opt = opt
        self.init_all_learners()
        self.allocate_tensors()
        self.curr_epoch = 0
        self.optimizers = {}
        self.keep_best_model_metric_name = None
        self.max_best = None


    def set_experiment_dir(self, directory_path):
        self.exp_dir = directory_path
        if (not os.path.isdir(self.exp_dir)):
            os.makedirs(self.exp_dir)

        self.vis_dir = os.path.join(directory_path, 'visuals')
        if (not os.path.isdir(self.vis_dir)):
            os.makedirs(self.vis_dir)

        self.preds_dir = os.path.join(directory_path,'preds')
        if (not os.path.isdir(self.preds_dir)):
            os.makedirs(self.preds_dir)


    def set_log_file_handler(self):
        self.logger = logging.getLogger(__name__)

        strHandler = logging.StreamHandler()
        formatter = logging.Formatter(
                '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        self.logger.addHandler(strHandler)
        self.logger.setLevel(logging.INFO)

        log_dir = os.path.join(self.exp_dir, 'logs')
        if (not os.path.isdir(log_dir)):
            os.makedirs(log_dir)

        now_str = datetime.datetime.now().__str__().replace(' ','_')

        self.log_file = os.path.join(log_dir, 'LOG_INFO_'+now_str+'.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        self.logger.addHandler(self.log_fileHandler)


    def init_all_learners(self):
        learners_defs = self.opt['learners']
        self.learners = {}
        self.optim_params = {}

        for key, value in learners_defs.items():
            self.logger.info('Set learner %s' % key)
            def_file = value['def_file']
            learner_opt = value['opt']
            self.optim_params[key] = (value['optim_params'] if ('optim_params' in value) else None)
            pretrained_path = value['pretrained'] if ('pretrained' in value) else None
            self.learners[key] = self.init_learner(def_file, learner_opt, pretrained_path, key)


    def init_learner(self, learner_def_file, learner_opt, pretrained_path, key):
        self.logger.info('==> Initialize learner %s from file %s with opts: %s' %
                         (key, learner_def_file, learner_opt))
        if (not os.path.isfile(learner_def_file)):
            raise ValueError('No existing file: {0}'.format(learner_def_file))


        learner = imp.machinery.SourceFileLoader('', learner_def_file).load_module().create_model(learner_opt)
        if pretrained_path != None:
            self.load_pretrained(learner, pretrained_path)

        return learner

    def load_pretrained(self, learner, pretrained_path):
        all_possible_files = glob.glob(pretrained_path)
        if len(all_possible_files) == 0:
            raise ValueError('{0}: no such file'.format(pretrained_path))
        else:
            pretrained_path = all_possible_files[-1]
        self.logger.info('==> Load pretrained parameters from file %s:' %
                         pretrained_path)

        assert(os.path.isfile(pretrained_path))
        pretrained_model = torch.load(pretrained_path)
        if pretrained_model['learner'].keys() == learner.state_dict().keys():
            learner.load_state_dict(pretrained_model['learner'])
        else:
            self.logger.info('==> WARNING: learner parameters in pre-trained file'
                             ' %s do not strictly match' % (pretrained_path))
            for pname, param in learner.named_parameters():
                if pname in pretrained_model['learner']:
                    self.logger.info('==> Copying parameter %s from file %s' %
                                     (pname, pretrained_path))
                    param.data.copy_(pretrained_model['learner'][pname])

    def init_all_optimizers(self):
        self.optimizers = {}
        for key, oparams in self.optim_params.items():
            self.optimizers[key] = None
            if oparams != None:
                self.optimizers[key] = self.init_optimizer(self.learners[key], oparams, key)

    def init_optimizer(self, learner, optim_opts, key):
        optim_type = optim_opts['optim_type']
        learning_rate = optim_opts['lr']
        optimizer = None
        parameters = filter(lambda p: p.requires_grad, learner.parameters())
        self.logger.info('Initialize optimizer: %s with params: %s for learner: %s'
                         % (optim_type, optim_opts, key))
        if optim_type == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=learning_rate, betas=optim_opts['beta'])
        elif optim_type == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=learning_rate,
                momentum=optim_opts['momentum'],
                nesterov=optim_opts['nesterov'] if ('nesterov' in optim_opts) else False,
                weight_decay=optim_opts['weight_decay'])
        else:
            raise ValueError('Not supported or recognized optim_type', optim_type)

        return optimizer


    def allocate_tensors(self):
        """(Optional) allocate torch tensors that could potentially be used in
            in the train_step() or evaluation_step() functions. If the
            load_to_gpu() function is called then those tensors will be moved to
            the gpu device.
        """
        self.tensor = {}


    def load_to_gpu(self):
        for key, learner in self.learners.items():
            self.learners[key] = learner.cuda()

        for key, tensor in self.tensor.items():
            self.tensor[key] = tensor.cuda()


    def save_checkpoint(self, epoch, suffix='', metric=None):
        for key, learner in self.learners.items():
            if self.optimizers[key] == None: continue
            self.save_learner(key, epoch, suffix=suffix, metric=metric)
            self.save_optimizer(key, epoch, suffix=suffix)


    def save_learner(self, learner_key, epoch, suffix='', metric=None):
        assert(learner_key in self.learners)
        filename = self._get_learner_checkpoint_filename(learner_key, epoch)+suffix
        state = {
            'epoch': epoch,
            'learner': self.learners[learner_key].state_dict(),
            'metric': metric
        }
        torch.save(state, filename)


    def save_optimizer(self, learner_key, epoch, suffix=''):
        assert(learner_key in self.optimizers)
        filename = self._get_optim_checkpoint_filename(learner_key, epoch)+suffix
        state = {
            'epoch': epoch,
            'optimizer': self.optimizers[learner_key].state_dict()
        }
        torch.save(state, filename)


    def _get_learner_checkpoint_filename(self, learner_key, epoch):
        return os.path.join(self.exp_dir, learner_key+'_learner_epoch'+str(epoch))

    def _get_optim_checkpoint_filename(self, learner_key, epoch):
        return os.path.join(self.exp_dir, learner_key+'_optim_epoch'+str(epoch))


    def find_most_recent_epoch(self, key, suffix):
        search_patern = self._get_learner_checkpoint_filename(key, '*') + suffix
        all_files = glob.glob(search_patern)
        if len(all_files) == 0:
            raise ValueError('%s: no such file.' % (search_patern))

        substrings = search_patern.split('*')
        assert(len(substrings) == 2)
        start, end = substrings
        all_epochs = [fname.replace(start,'').replace(end,'') for fname in all_files]
        all_epochs = [int(epoch) for epoch in all_epochs if epoch.isdigit()]
        assert(len(all_epochs) > 0)
        all_epochs = sorted(all_epochs)
        most_recent_epoch = int(all_epochs[-1])
        self.logger.info('Load checkpoint of most recent epoch %s' %
                         str(most_recent_epoch))
        return most_recent_epoch


    def load_checkpoint(self, epoch, train=True, suffix=''):
        self.logger.info('Load checkpoint of epoch %s' % (str(epoch)))

        for key, learner in self.learners.items(): # Load learners
            if self.optim_params[key] == None: continue
            if epoch == '*':
                epoch = self.find_most_recent_epoch(key, suffix)
            self.load_learner(key, epoch, suffix)

        if train: # initialize and load optimizers
            self.init_all_optimizers()
            for key, learner in self.learners.items():
                if self.optim_params[key] == None: continue
                self.load_optimizer(key, epoch, suffix)

        self.curr_epoch = epoch

    def load_learner(self, learner_key, epoch,suffix=''):
        assert(learner_key in self.learners)
        filename = self._get_learner_checkpoint_filename(learner_key, epoch)+suffix
        self.logger.info('Loading {0} for learner {1}'.format(filename, learner_key))
        assert(os.path.isfile(filename))
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.learners[learner_key].load_state_dict(checkpoint['learner'])

    def load_optimizer(self, learner_key, epoch,suffix=''):
        assert(learner_key in self.optimizers)
        filename = self._get_optim_checkpoint_filename(learner_key, epoch)+suffix
        assert(os.path.isfile(filename))
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.optimizers[learner_key].load_state_dict(checkpoint['optimizer'])


    def delete_checkpoint(self, epoch, suffix=''):
        for key, learner in self.learners.items():
            if self.optimizers[key] == None: continue

            filename_learner = self._get_learner_checkpoint_filename(key, epoch)+suffix
            if os.path.isfile(filename_learner): os.remove(filename_learner)

            filename_optim = self._get_optim_checkpoint_filename(key, epoch)+suffix
            if os.path.isfile(filename_optim): os.remove(filename_optim)

    def solve(self, data_loader_train, data_loader_test=None):
        self.max_num_epochs = self.opt['max_num_epochs']
        start_epoch = self.curr_epoch
        if len(self.optimizers) == 0:
            self.init_all_optimizers()

        # eval_stats  = {}
        # train_stats = {}
        self.init_record_of_best_model()
        for self.curr_epoch in range(start_epoch, self.max_num_epochs):
            self.logger.info('Training epoch [%3d / %3d]' % (self.curr_epoch+1, self.max_num_epochs))
            self.adjust_learning_rates(self.curr_epoch)
            train_stats = self.run_train_epoch(data_loader_train, self.curr_epoch)
            self.logger.info('==> Training stats: %s' % (train_stats))

            self.save_checkpoint(self.curr_epoch+1) # create a checkpoint in the current epoch
            if start_epoch != self.curr_epoch: # delete the checkpoint of the previous epoch
                self.delete_checkpoint(self.curr_epoch)

            if data_loader_test is not None:
                eval_stats = self.evaluate(data_loader_test)
                self.logger.info('==> Evaluation stats: %s' % (eval_stats))
                self.keep_record_of_best_model(eval_stats, self.curr_epoch)

        self.print_eval_stats_of_best_model()


    def init_record_of_best_model(self):
        self.optimal_metric_val = None
        self.best_stats = None
        self.best_epoch = None


    def adjust_learning_rates(self, epoch):
        # filter out the learners that are not trainable and that do
        # not have a learning rate Look Up Table (LUT_lr) in their optim_params
        optim_params_filtered = {k:v for k,v in self.optim_params.items()
            if (v != None and ('LUT_lr' in v))}

        for key, oparams in optim_params_filtered.items():
            LUT = oparams['LUT_lr']
            lr = next((lr for (max_epoch, lr) in LUT if max_epoch>epoch), LUT[-1][1])
            self.logger.info('==> Set to %s optimizer lr = %.10f' % (key, lr))
            for param_group in self.optimizers[key].param_groups:
                param_group['lr'] = lr

    def run_train_epoch(self, data_loader, epoch):
        self.logger.info('Training: %s' % os.path.basename(self.exp_dir))
        # self.dloader = data_loader
        # self.dataset_train = data_loader.dataset

        for key, learner in self.learners.items():
            if self.optimizers[key] == None: learner.eval()
            else: learner.train()

        disp_step = self.opt['disp_step'] if ('disp_step' in self.opt) else 50
        train_stats = DAverageMeter()
        self.bnumber = len(data_loader())
        for idx, batch in enumerate(tqdm(data_loader(epoch))):
            self.biter = idx # batch iteration.
            # self.global_iter = self.curr_epoch * self.bnumber + self.biter
            train_stats_this = self.train_step(batch)
            train_stats.update(train_stats_this)
            if (idx+1) % disp_step == 0:
                self.logger.info('==> Iteration [%3d][%4d / %4d]: %s' %
                                 (epoch+1, idx+1, self.bnumber, train_stats.average()))

        return train_stats.average()

    def evaluate(self, dloader):
        self.logger.info('Evaluating: %s' % os.path.basename(self.exp_dir))

        # self.dloader = dloader
        # self.dataset_eval = dloader.dataset
        # self.logger.info('==> Dataset: %s [%d batches]' % (dloader.dataset.name, len(dloader)))
        for key, learner in self.learners.items():
            learner.eval()

        eval_stats = DAverageMeter()
        self.bnumber = len(dloader())
        for idx, batch in enumerate(tqdm(dloader())):
            self.biter = idx
            eval_stats_this = self.evaluation_step(batch)
            eval_stats.update(eval_stats_this)

        self.logger.info('==> Results: %s' % eval_stats.average())

        return eval_stats.average()


    def keep_record_of_best_model(self, eval_stats, current_epoch):
        if self.keep_best_model_metric_name is not None:
            metric_name = self.keep_best_model_metric_name
            if (metric_name not in eval_stats):
                raise ValueError('The provided metric {0} for keeping the best '
                                 'model is not computed by the evaluation routine.'
                                 .format(metric_name))
            metric_val = eval_stats[metric_name]

            if self.max_best:
                if self.optimal_metric_val is None or metric_val > self.optimal_metric_val:
                    self.optimal_metric_val = metric_val
                    self.best_stats = eval_stats
                    self.save_checkpoint(
                        self.curr_epoch+1, suffix='.best', metric=self.optimal_metric_val)
                    if self.best_epoch is not None:
                        self.delete_checkpoint(self.best_epoch+1, suffix='.best')
                    self.best_epoch = current_epoch
                    self.print_eval_stats_of_best_model()
            else:
                if self.optimal_metric_val is None or metric_val < self.optimal_metric_val:
                    self.optimal_metric_val = metric_val
                    self.best_stats = eval_stats
                    self.save_checkpoint(
                        self.curr_epoch+1, suffix='.best', metric=self.optimal_metric_val)
                    if self.best_epoch is not None:
                        self.delete_checkpoint(self.best_epoch+1, suffix='.best')
                    self.best_epoch = current_epoch
                    self.print_eval_stats_of_best_model()

    def print_eval_stats_of_best_model(self):
        if self.best_stats is not None:
            metric_name = self.keep_best_model_metric_name
            self.logger.info('==> Best results w.r.t. %s metric: epoch: %d - %s'
                                     % (metric_name, self.best_epoch + 1, self.best_stats))

    # FROM HERE ON ARE ABSTRACT FUNCTIONS THAT MUST BE IMPLEMENTED BY THE CLASS
    # THAT INHERITS THE Algorithms CLASS
    def train_step(self, batch):
        """Implements a training step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es)
            * Backward propagation through the networks
            * Apply optimization step(s)
            * Return a dictionary with the computed losses and any other desired
                stats. The key names on the dictionary can be arbitrary.
        """
        pass

    def evaluation_step(self, batch):
        """Implements an evaluation step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es) or any other evaluation metrics.
            * Return a dictionary with the computed losses the evaluation
                metrics for that batch. The key names on the dictionary can be
                arbitrary.
        """
        pass