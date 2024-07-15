#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob

# torch
import torch
# import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(import_str):
    """
    Import a class from a string.

    Args:
        import_str (str): A string of the form 'module_name.ClassName'

    Returns:
        class: The class object

    Raises:
        ImportError: If the class cannot be found
    """
    # Split the string into module name and class name
    mod_str, _sep, class_str = import_str.rpartition('.')

    # Import the module
    __import__(mod_str)

    # Get the class from the module
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        # If the class cannot be found, raise an ImportError
        raise ImportError(
            'Class %s cannot be found (%s)' %
            (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        """
        Forward pass of the label smoothing loss function.

        Args:
            x (torch.Tensor): The input tensor, typically the output of a model.
            target (torch.Tensor): The target tensor, typically one-hot encoded
                labels.

        Returns:
            loss (torch.Tensor): The label smoothing loss as a scalar.
        """

        # The confidence in the ground truth labels. If the smoothing is 0.1, then
        # the confidence is 0.9.
        confidence = 1. - self.smoothing

        # Compute the log probabilities of the input tensor. The log probabilities
        # are computed along the last dimension.
        logprobs = F.log_softmax(x, dim=-1)

        # Compute the negative log likelihood loss. The loss is computed by
        # taking the log probability of the target class and summing over all
        # samples in the batch.
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))

        # Squeeze the loss tensor to remove the extra dimension.
        nll_loss = nll_loss.squeeze(1)

        # Compute the smoothing loss. The smoothing loss is the mean of the
        # log probabilities over all classes.
        smooth_loss = -logprobs.mean(dim=-1)

        # The total loss is the weighted sum of the negative log likelihood loss
        # and the smoothing loss.
        loss = confidence * nll_loss + self.smoothing * smooth_loss

        # Return the mean of the loss over the batch.
        return loss.mean()


def get_parser():
    """
    Define the command line arguments for the training script. The priority of the arguments is as follows:
    command line > config file > default value.
    """

    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')

    # Define the command line arguments
    # Work directory
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',  # Default work directory
        help='the work folder for storing results')

    # Model saved name
    parser.add_argument('-model_saved_name', default='')

    # Config file path
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',  # Default config file path
        help='path to the configuration file')

    # Processor arguments
    # Phase
    parser.add_argument(
        '--phase', default='train', help='must be train or test')

    # Save score
    parser.add_argument(
        '--save-score',
        type=str2bool,  # Boolean type argument
        default=False,  # Default value
        help='if true, the classification score will be stored')

    # Visualize and debug arguments
    # Random seed
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')

    # Log interval
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')

    # Save interval
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')

    # Save epoch
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')

    # Evaluation interval
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')

    # Print log
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')

    # Show top k accuracy
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # Data loader arguments
    # Data feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')

    # Number of worker
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')

    # Training data loader arguments
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')

    # Testing data loader arguments
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # Model arguments
    # Model
    parser.add_argument('--model', default=None, help='the model will be used')

    # Model arguments
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')

    # Weights
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')

    # Ignored weights
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # Optimizer arguments
    # Base learning rate
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')

    # Step for learning rate decay
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')

    # Device
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')

    # Optimizer
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')

    # Use Nesterov or not
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')

    # Batch size
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')

    # Test batch size
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')

    # Start epoch
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')

    # Number of epochs
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')

    # Weight decay
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')

    # Learning rate decay ratio
    parser.add_argument(
        '--lr-ratio',
        type=float,
        default=0.001,
        help='decay rate for learning rate')

    # Learning rate decay rate
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')

    # Warm up epochs
    parser.add_argument(
        '--warm_up_epoch', type=int, default=0)

    # Loss type
    parser.add_argument('--loss-type', type=str, default='CE')

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def load_data(self):
        # Import the feeder class based on the argument
        Feeder = import_class(self.arg.feeder)
        
        # Initialize an empty dictionary to store the data loaders
        self.data_loader = dict()
        
        # If the phase is 'train', load the train data
        if self.arg.phase == 'train':
            # Create a DataLoader for the train data
            self.data_loader['train'] = torch.utils.data.DataLoader(
                # Use the feeder class to create the dataset
                dataset=Feeder(**self.arg.train_feeder_args),
                # Set the batch size
                batch_size=self.arg.batch_size,
                # Shuffle the data
                shuffle=True,
                # Number of worker processes
                num_workers=self.arg.num_worker,
                # Drop the last batch if it's not full
                drop_last=True,
                # Initialize the seed for each worker
                worker_init_fn=init_seed)
        
        # If the phase is not 'train', load the test data
        else:
            # Create a DataLoader for the test data
            self.data_loader['test'] = torch.utils.data.DataLoader(
                # Use the feeder class to create the dataset
                dataset=Feeder(**self.arg.test_feeder_args),
                # Set the batch size
                batch_size=self.arg.test_batch_size,
                # Do not shuffle the data
                shuffle=False,
                # Number of worker processes
                num_workers=self.arg.num_worker,
                # Do not drop the last batch
                drop_last=False,
                # Initialize the seed for each worker
                worker_init_fn=init_seed)

# Define a function to load the model
    def load_model(self):
        # Determine the output device based on the argument
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device

        # Import the model class based on the argument
        Model = import_class(self.arg.model)
        
        # Copy the model file to the working directory
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        
        # Initialize the model using the model class and its arguments
        self.model = Model(**self.arg.model_args)
        
        # Choose the appropriate loss function based on the loss type
        if self.arg.loss_type == 'CE':
            self.loss = nn.CrossEntropyLoss().cuda(output_device)
        else:
            self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(output_device)

        # Load weights if specified in the arguments
        if self.arg.weights:
            # Extract the global step from the weights file name
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            
            # Load the weights from the file
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            # Move the weights to the output device and convert to OrderedDict
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            # Get the list of keys from the weights
            keys = list(weights.keys())
            
            # Remove specified weights to ignore
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                # Load the model state dict with the weights
                self.model.load_state_dict(weights)
            except Exception:
                # Handle exceptions by updating the model state dict
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        """
        This function loads the optimizer used for training the model.
        It supports two optimizers: SGD and Adam.
        The optimizer is set based on the value of the argument 'optimizer' in the argument dictionary.
        The learning rate, momentum, nesterov, and weight decay are set based on the values of the corresponding arguments.
        If the optimizer is SGD, the momentum and nesterov are set to 0.9 and True respectively.
        If the optimizer is Adam, the weight decay is set to the value of the argument 'weight_decay'.
        If the optimizer is neither SGD nor Adam, a ValueError is raised.
        Finally, a log message is printed indicating that the warm up epoch is set to the value of the argument 'warm_up_epoch'.
        """
        # Check the optimizer and set the optimizer accordingly
        if self.arg.optimizer == 'SGD':
            # If the optimizer is SGD, set the learning rate, momentum, nesterov, and weight decay
            self.optimizer = optim.SGD(
                self.model.parameters(),  # Set the parameters of the model as the parameters of the optimizer
                lr=self.arg.base_lr,  # Set the learning rate to the value of the argument 'base_lr'
                momentum=0.9,  # Set the momentum to 0.9
                nesterov=self.arg.nesterov,  # Set the nesterov flag to the value of the argument 'nesterov'
                weight_decay=self.arg.weight_decay  # Set the weight decay to the value of the argument 'weight_decay'
            )
        elif self.arg.optimizer == 'Adam':
            # If the optimizer is Adam, set the learning rate and weight decay
            self.optimizer = optim.Adam(
                self.model.parameters(),  # Set the parameters of the model as the parameters of the optimizer
                lr=self.arg.base_lr,  # Set the learning rate to the value of the argument 'base_lr'
                weight_decay=self.arg.weight_decay  # Set the weight decay to the value of the argument 'weight_decay'
            )
        else:
            # If the optimizer is neither SGD nor Adam, raise a ValueError
            raise ValueError()

        # Print a log message indicating that the warm up epoch is set to the value of the argument 'warm_up_epoch'
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # This function saves the arguments used for the run to a YAML file in the work directory.
        # The file is named "config.yaml" and contains the arguments used for the run.

        # Get the arguments as a dictionary
        arg_dict = vars(self.arg)

        # Check if the work directory exists. If not, create it.
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

        # Open the config.yaml file in the work directory for writing.
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            # Write a comment to the file indicating the command line used to run the script.
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")

            # Dump the argument dictionary to the file in YAML format.
            # This will create a human-readable YAML file with the arguments used for the run.
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch, idx):
        """
        Adjust the learning rate based on the current epoch and index.

        The learning rate is adjusted based on the following schedule:

        - For the first `warm_up_epoch` epochs, the learning rate is linearly
        increased from 0 to `base_lr`.
        - For the remaining epochs, the learning rate is cosine-annealed from
        `base_lr` to `base_lr * lr_ratio` over the course of the remaining
        epochs.

        The learning rate is updated for the optimizer, and the updated value
        is returned.

        Args:
            epoch: The current epoch number.
            idx: The current index in the epoch.

        Returns:
            The updated learning rate.
        """
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            # If we're in the warm-up phase, linearly increase the learning rate
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                # Otherwise, compute the learning rate based on the cosine annealing schedule
                T_max = len(self.data_loader['train']) * (self.arg.num_epoch - self.arg.warm_up_epoch)
                T_cur = len(self.data_loader['train']) * (epoch - self.arg.warm_up_epoch) + idx

                # Compute the minimum learning rate (eta_min) for the cosine annealing schedule
                eta_min = self.arg.base_lr * self.arg.lr_ratio

                # Compute the learning rate using the cosine annealing schedule
                lr = eta_min + 0.5 * (self.arg.base_lr - eta_min) * (1 + np.cos((T_cur / T_max) * np.pi))

            # Update the learning rate for the optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        """
        Perform training for a single epoch.

        Args:
            epoch (int): The current epoch number.
            save_model (bool, optional): Whether to save the model weights. Defaults to False.
        """
        
        # Set the model to training mode
        self.model.train()

        # Print the current epoch number
        self.print_log('Training epoch: {}'.format(epoch + 1))

        # Get the training data loader
        loader = self.data_loader['train']

        # Initialize empty lists to store the loss and accuracy values
        loss_value = []
        acc_value = []

        # Add the current epoch number to the tensorboard writer
        self.train_writer.add_scalar('epoch', epoch, self.global_step)

        # Record the current time
        self.record_time()

        # Initialize a dictionary to track the time spent in different stages
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        # Create a progress bar iterator for the training data loader
        process = tqdm(loader)

        # Iterate over the training data batch by batch
        for batch_idx, (data, label, index) in enumerate(process):

            # Adjust the learning rate based on the current epoch and batch index
            self.adjust_learning_rate(epoch, batch_idx)

            # Increment the global step counter
            self.global_step += 1

            # Move the data and labels to the appropriate device (GPU)
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)

            # Update the timer for data loading
            timer['dataloader'] += self.split_time()

            # Perform forward pass
            output = self.model(data)
            loss = self.loss(output, label)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Perform backward pass and update the model parameters
            loss.backward()
            self.optimizer.step()

            # Append the loss value to the list
            loss_value.append(loss.data.item())

            # Update the timer for model training
            timer['model'] += self.split_time()

            # Calculate the accuracy
            _, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())

            # Append the accuracy value to the list
            acc_value.append(acc.data.item())

            # Add the accuracy and loss values to the tensorboard writer
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # Update the learning rate
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)

            # Update the timer for statistics
            timer['statistics'] += self.split_time()

        # Calculate the proportion of time spent in each stage
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        # Print the mean training loss and accuracy
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))

        # Print the current learning rate
        self.print_log('\tLearning Rate: {:.4f}'.format(self.lr))

        # Print the time consumption in different stages
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        # Save the model weights if specified
        if save_model:
            # Get the state dictionary of the model
            state_dict = self.model.state_dict()

            # Create an ordered dictionary with the model weights
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            # Save the model weights to a file
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        # Open the files for writing if specified
        if wrong_file is not None:
            # Open the file for writing the wrong predictions
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            # Open the file for writing the prediction results
            f_r = open(result_file, 'w')
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Print the evaluation epoch
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        
        # Iterate over the specified loader names
        for ln in loader_name:
            # Initialize the lists for storing loss values and scores
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            
            # Initialize the progress bar for the current loader
            process = tqdm(self.data_loader[ln])
            
            # Iterate over the batches in the loader
            for batch_idx, (data, label, index) in enumerate(process):
                # Append the labels to the list
                label_list.append(label)
                
                # Evaluate the model without gradient computation
                with torch.no_grad():
                    # Move the data and labels to the appropriate device
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    
                    # Get the model output
                    output = self.model(data)
                    
                    # Calculate the loss
                    loss = self.loss(output, label)
                    
                    # Append the model output score to the list
                    score_frag.append(output.data.cpu().numpy())
                    
                    # Append the loss value to the list
                    loss_value.append(loss.data.item())
                    
                    # Get the predicted labels
                    _, predict_label = torch.max(output.data, 1)
                    
                    # Append the predicted labels to the list
                    pred_list.append(predict_label.data.cpu().numpy())
                    
                    # Increment the step counter
                    step += 1
                
                # If either the wrong file or result file is specified, write the predictions to the respective files
                if wrong_file is not None or result_file is not None:
                    # Get the predicted labels and true labels as lists
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    
                    # Iterate over the predictions
                    for i, x in enumerate(predict):
                        # If the result file is specified, write the prediction and true label to the file
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        
                        # If the prediction is incorrect and the wrong file is specified, write the index, prediction, and true label to the file
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            
            # Concatenate the scores from multiple batches to a single array
            score = np.concatenate(score_frag)
            
            # Calculate the mean loss value
            loss = np.mean(loss_value)
            
            # If the feeder is UCLA, update the sample names in the loader dataset
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            
            # Calculate the accuracy of the model on the current loader
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            
            # If the accuracy is better than the best accuracy, update the best accuracy and epoch
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
            
            # Print the accuracy and the model name
            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            
            # If the phase is training, add the loss and accuracy to the tensorboard writer
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)
            
            # Create a dictionary of scores for each sample
            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            
            # Print the mean loss value for the current loader
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            
            # Print the top-k accuracy for each specified value of k
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
            
            # If the save_score flag is set, save the score dictionary to a file
            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)
            
            # Calculate the accuracy for each class in the confusion matrix
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            
            # Save the accuracy for each class to a file
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self):
        # Check if the phase is 'train'
        if self.arg.phase == 'train':
            # Print the parameters
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            
            # Calculate the global step
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            
            # Function to count the number of parameters in the model
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Print the number of parameters in the model
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            
            # Loop through the epochs
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # Determine if the model should be saved
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch   
                
                # Train the model
                self.train(epoch, save_model=True)
                
                # Evaluate the model
                self.eval(epoch, save_score=True, loader_name=['test'])
            
            # Find the best model weights path
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            
            # Load the best model weights
            weights = torch.load(weights_path)
            
            # Check the device type
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            
            # Load the model state dictionary with the best weights
            self.model.load_state_dict(weights)
            
            # Generate paths for wrong and right files
            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            
            # Disable print log
            self.arg.print_log = False
            
            # Evaluate the model on the test dataset
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            
            # Enable print log
            self.arg.print_log = True
            
            # Calculate the total number of parameters in the model
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Print model evaluation results
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')
        
        # Check if the phase is 'test'
        elif self.arg.phase == 'test':
            # Generate paths for wrong and right files
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')
            
            # Check if weights are provided
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            
            # Disable print log
            self.arg.print_log = False
            
            # Print model and weights information
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            
            # Evaluate the model on the test dataset
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            
            # Enable print log
            self.arg.print_log = True
            
            # Print done message
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()