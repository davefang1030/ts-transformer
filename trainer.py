from tqdm import tqdm, tqdm_notebook
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from argparse import Namespace
import os
import pandas as pd


class ModelTrainer(object):
    """
    Common logic for model training
    """
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._is_notebook = False
        try:
            self._is_notebook = True if get_ipython() else False
        except NameError:
            pass

    def train(self, dataset, args, mask_index=-1):
        """

        :param dataset: data
        :param args: training args
        :param mask_index:
        :return:
        """
        train_state = self.make_train_state(args)
        if self._is_notebook:
            epoch_bar = tqdm_notebook(desc='training routine', total=args.num_epochs, position=0)
        else:
            epoch_bar = tqdm(desc='training routine', total=args.num_epochs, position=0)

        dataset.set_split('train')
        if self._is_notebook:
            train_bar = tqdm_notebook(desc='split=train', total=dataset.get_num_batches(args.batch_size),
                                      position=1, leave=False)
        else:
            train_bar = tqdm(desc='split=train', total=dataset.get_num_batches(args.batch_size),
                             position=1, leave=False)

        dataset.set_split('val')
        if self._is_notebook:
            val_bar = tqdm_notebook(desc='split=val', total=dataset.get_num_batches(args.batch_size),
                                    position=1, leave=False)
        else:
            val_bar = tqdm(desc='split=val', total=dataset.get_num_batches(args.batch_size),
                           position=1, leave=False)


        self.model.to(args.device)

        try:
            for epoch_index in range(args.num_epochs):
                train_state['epoch_index'] = epoch_index

                # training set
                dataset.set_split('train')
                batch_generator = dataset.generate_batches(dataset, batch_size=args.batch_size, device=args.device)
                running_loss = 0.0
                running_acc = 0.0
                self.model.train()
                for batch_index, batch_dict in enumerate(batch_generator):
                    # step 1. zero the gradients
                    self.optimizer.zero_grad()
                    # step 2. forward pass
                    y_pred = self.forward_pass(batch_dict)
                    # step 3. compute the loss
                    loss = self.calculate_loss(y_pred, batch_dict, mask_index)
                    loss.backward()
                    # step 5. gradient step
                    self.optimizer.step()
                    # compute running loss and accuracy
                    running_loss = (running_loss * batch_index + loss.item()) / (batch_index + 1)
                    acc_t = self.compute_accuracy(y_pred, batch_dict, mask_index)
                    running_acc = (running_acc * batch_index + acc_t) / (batch_index + 1)
                    # update bar
                    train_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
                    train_bar.update()

                train_state['train_loss'].append(running_loss)
                train_state['train_acc'].append(running_acc)

                # valuation set
                dataset.set_split('val')
                batch_generator = dataset.generate_batches(dataset, batch_size=args.batch_size, device=args.device)
                running_loss = 0.0
                running_acc = 0.0

                self.model.eval()
                for batch_index, batch_dict in enumerate(batch_generator):
                    y_pred = self.forward_pass(batch_dict)
                    loss = self.calculate_loss(y_pred, batch_dict, mask_index)
                    # compute running loss and accuracy
                    running_loss = (running_loss * batch_index + loss.item()) / (batch_index + 1)
                    acc_t = self.compute_accuracy(y_pred, batch_dict, mask_index)
                    running_acc = (running_acc * batch_index + acc_t) / (batch_index + 1)
                    # update bar
                    val_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
                    val_bar.update()

                train_state['val_loss'].append(running_loss)
                train_state['val_acc'].append(running_acc)

                # update train_state
                train_state = self.update_train_state(args=args, train_state=train_state)

                # update learning rate
                self.scheduler.step(train_state['val_loss'][-1])

                if train_state['stop_early']:
                    break

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

        except KeyboardInterrupt:
            print("Existing training")

        return train_state

    def eval(self, dataset, args, split='test', mask_index=-1):
        """
        evaluate performance
        :param dataset:
        :param args:
        :param split:
        :param mask_index:
        :return:
        """
        # valuation set
        dataset.set_split(split)
        batch_generator = dataset.generate_batches(dataset, batch_size=args.batch_size, device=args.device)
        running_loss = 0.0
        running_acc = 0.0

        self.model.eval()
        predictions = np.zeros([len(dataset), 1])
        predictions_probs = []
        starting_index = 0
        for batch_index, batch_dict in enumerate(batch_generator):
            y_pred = self.forward_pass(batch_dict)
            loss = self.calculate_loss(y_pred, batch_dict, mask_index)
            # compute running loss and accuracy
            running_loss = (running_loss * batch_index + loss.item()) / (batch_index + 1)
            acc_t = self.compute_accuracy(y_pred, batch_dict, mask_index)
            running_acc = (running_acc * batch_index + acc_t) / (batch_index + 1)
            # save prediction and probabilities
            prob = F.softmax(y_pred, dim=1).detach().cpu().numpy()
            _, pred = prob.max(dim=1)
            predictions_probs.append(prob)
            predictions[starting_index:starting_index+len(pred), :] = pred
            starting_index += len(pred)

        return running_loss, running_acc, predictions, predictions_probs

    def forward_pass(self, batch_dict):
        """
        Calculate forward pass of the network
        :param batch_dict:
        :return: prediction
        """
        raise NotImplementedError

    def calculate_loss(self, y_pred, batch_dict, mask_index):
        """
        Calculate loss for the batch
        :param y_pred:
        :param batch_dict:
        :param mask_index:
        :return: loss
        """
        raise NotImplementedError

    def compute_accuracy(self, y_pred, batch_dict, mask_index):
        """
        Calculate accuracy...
        :param y_pred:
        :param batch_dict:
        :param mask_index:
        :return:
        """
        raise NotImplementedError

    def update_train_state(self, args, train_state):
        """Handle the training state updates.
        Components:
         - Early Stopping: Prevent overfitting.
         - Model Checkpoint: Model is saved if the model is better
        :param args: main arguments
        :param train_state: a dictionary representing the training state values
        :returns: a new train_state
        """
        # Save one model at least
        if train_state['epoch_index'] == 0:
            torch.save(self.model.state_dict(), train_state['model_filename'])
            train_state['stop_early'] = False

        # Save model if performance improved
        elif train_state['epoch_index'] >= 1:
            loss_tm1, loss_t = train_state['val_loss'][-2:]

            # If loss worsened
            if loss_t >= loss_tm1:
                # Update step
                train_state['early_stopping_step'] += 1
            # Loss decreased
            else:
                # Save the best model
                if loss_t < train_state['early_stopping_best_val']:
                    torch.save(self.model.state_dict(), train_state['model_filename'])
                    train_state['early_stopping_best_val'] = loss_t

                # Reset early stopping step
                train_state['early_stopping_step'] = 0

            # Stop early ?
            train_state['stop_early'] = \
                train_state['early_stopping_step'] >= args.early_stopping_criteria

        return train_state

    def make_train_state(self, args):
        train_state = {'stop_early': False,
                       'early_stopping_step': 0,
                       'early_stopping_best_val': 1e8,
                       'learning_rate': args.learning_rate,
                       'epoch_index': 0,
                       'train_loss': [],
                       'train_acc': [],
                       'val_loss': [],
                       'val_acc': [],
                       'test_loss': -1,
                       'test_acc': -1,
                       'model_filename': args.model_state_file}
        return train_state

    @staticmethod
    def set_seed_everywhere(seed, cuda):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def handle_dirs(dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)


def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)


def normalize_sizes(y_pred, y_true):
    """Normalize tensor sizes

    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true


def calc_accuracy(y_pred, y_true, mask_index):
    """

    :param y_pred: softmax prediction. max indicates the prediction
    :param y_true:
    :param mask_index:
    :return:
    """
    """
    Args:
        y_pred () : 
        y_target: 
    """
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100

