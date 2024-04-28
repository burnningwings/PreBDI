"""
Written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

import logging

import numpy as np

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")
import os
import sys
import time
import pickle
import json

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Project modules
from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.PreBDI import model_factory
from models.loss import get_loss_module
from optimizers import get_optimizer


def main(config):

    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config['data_class']]
    my_data = data_class(config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    feat_dim = my_data.feature_df.shape[-1]  # dimensionality of data features

    # Split dataset
    test_data = my_data
    val_data = my_data
    if config['test_pattern']:
        test_data = data_class(config['data_dir'], pattern=config['test_pattern'], n_proc=-1, config=config)
    if config['val_pattern']:
        val_data = data_class(config['data_dir'], pattern=config['val_pattern'], n_proc=-1, config=config)

    logger.info("{} samples may be used for training".format(len(my_data.feature_df)))
    logger.info("{} samples will be used for validation".format(len(val_data.feature_df)))
    logger.info("{} samples will be used for testing".format(len(test_data.feature_df)))

    # Pre-process features
    normalizer = None
    if config['norm_from']:
        with open(config['norm_from'], 'rb') as f:
            norm_dict = pickle.load(f)
        normalizer = Normalizer(**norm_dict)
    elif config['normalization'] is not None and config['normalization'] != 'None':
        normalizer = Normalizer(config['normalization'])
        my_data.feature_df[..., 0] = normalizer.normalize(my_data.feature_df[..., 0])
        if not config['normalization'].startswith('per_sample'):
            # get normalizing values from training set and store for future use
            norm_dict = normalizer.__dict__
            with open(os.path.join(config['output_dir'], 'normalization.pickle'), 'wb') as f:
                pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
    if normalizer is not None:
        if len(val_data.feature_df):
            val_data.feature_df[..., 0] = normalizer.normalize(val_data.feature_df[..., 0])
        if len(test_data.feature_df):
            test_data.feature_df[..., 0] = normalizer.normalize(test_data.feature_df[..., 0])

    # Create model
    logger.info("Creating model ...")
    model = model_factory(config, my_data)

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))


    # Initialize optimizer

    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=7)

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config['lr']  # current learning step
    # Load model and optimizer state
    if args.load_model:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
    model.to(device)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model, device_ids=[0, 1])

    loss_module = get_loss_module(config)

    if config['test_only'] == 'testset':  # Only evaluate and skip training
        dataset_class, collate_fn, runner_class = pipeline_factory(config)
        # test_dataset = dataset_class(test_data, test_indices)
        test_dataset = dataset_class(test_data)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False,
                                 num_workers=config['num_workers'],
                                 pin_memory=True,
                                 collate_fn=lambda x: collate_fn(x, max_len=my_data.max_seq_len),drop_last=True)
        test_evaluator = runner_class(model, test_loader, device, loss_module,
                                            print_interval=config['print_interval'], console=config['console'])

        with torch.no_grad():
            aggr_metrics_test, per_batch_test = test_evaluator.evaluate(keep_all=True)
        print_str = 'Test Summary: '
        for k, v in aggr_metrics_test.items():
            if k == "epoch" and v is None:
                continue
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)

        return
    
    # Initialize data generators
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    # val_dataset = dataset_class(val_data, val_indices)
    val_dataset = dataset_class(val_data)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: collate_fn(x, max_len=my_data.max_seq_len), drop_last=True)

    # train_dataset = dataset_class(my_data, train_indices)
    train_dataset = dataset_class(my_data)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=True,
                              collate_fn=lambda x: collate_fn(x, max_len=my_data.max_seq_len), drop_last=True)

    trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                                 print_interval=config['print_interval'], console=config['console'])
    # val_evaluator = runner_class(model, val_loader, device, loss_module,
    #                                    print_interval=config['print_interval'], console=config['console'])

    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])
    val_evaluator = runner_class(model, val_loader, device, loss_module,
                                 print_interval=config['print_interval'], console=config['console'], tod=config['tod'], tbw=tensorboard_writer)



    best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}

    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                          best_value, epoch=0)
    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))

    logger.info('Starting training...')
    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False):
        mark = epoch if config['save_all'] else 'last'
        epoch_start_time = time.time()
        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        epoch_runtime = time.time() - epoch_start_time
        print()
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))
        total_epoch_time += epoch_runtime
        avg_epoch_time = total_epoch_time / (epoch - start_epoch)
        avg_batch_time = avg_epoch_time / len(train_loader)
        avg_sample_time = avg_epoch_time / len(train_dataset)
        logger.info("Avg epoch train. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_epoch_time)))
        logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))
        logger.info("Avg sample train. time: {} seconds".format(avg_sample_time))

        # evaluate if first or last epoch or at specified interval
        if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
            aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                                  best_metrics, best_value, epoch)
            if (config["task"] == "regression"):
                logger.info("Current lr: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
                dmae = aggr_metrics_val['val_mae']
                scheduler.step(dmae)

            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))

        # utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(mark)), epoch, model.module, optimizer)
        utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(mark)), epoch, model, optimizer)

        # Learning rate scheduling
        if epoch == config['lr_step'][lr_step]:
            # utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model.module, optimizer)
            utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model,
                             optimizer)
            lr = lr * config['lr_factor'][lr_step]
            if lr_step < len(config['lr_step']) - 1:  # so that this index does not get out of bounds
                lr_step += 1
            logger.info('Learning rate updated to: ', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Difficulty scheduling
        if config['harden'] and check_progress(epoch):
            train_loader.dataset.update()
            val_loader.dataset.update()

    # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + ".xls")
    book = utils.export_performance_metrics(metrics_filepath, metrics, header, sheet_name="metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"],
                          best_metrics, aggr_metrics_val, comment=config['comment'])

    logger.info('Best {} was {}. Other metrics: {}'.format(config['key_metric'], best_value, best_metrics))
    logger.info('All Done!')

    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))

    return best_value


if __name__ == '__main__':

    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)
