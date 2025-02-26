import numpy as np
import torch
from torchvision.utils import make_grid

from base.base_trainer import BaseTrainer


class DefaultTrainer(BaseTrainer):
    """
        Trainer class

        Note:
            Inherited from BaseTrainer.
        """

    def __init__(self, config, model, loss, metrics, optimizer, lr_scheduler, resume, data_loader,
                 valid_data_loader=None, train_logger=None, **extra_args):
        super(DefaultTrainer, self).__init__(config, model, loss, metrics, optimizer, lr_scheduler, resume,
                                             train_logger)

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self._load_pretrained_weights(**extra_args)  # load the pretrained weights of each subnetwork

    def _load_pretrained_weights(self, **extra_args):
        subnetwork1_checkpoint_path = extra_args.get('subnetwork1_checkpoint_path')
        subnetwork2_checkpoint_path = extra_args.get('subnetwork2_checkpoint_path')
        if subnetwork1_checkpoint_path:
            subnetwork1_checkpoint = torch.load(subnetwork1_checkpoint_path)
            if self.data_parallel:
                self.model.module.Subnetwork1.load_state_dict(subnetwork1_checkpoint['model'])
            else:
                self.model.Subnetwork1.load_state_dict(subnetwork1_checkpoint['model'])
            print('load subnetwork1_checkpoint from {} ...'.format(subnetwork1_checkpoint_path))
        if subnetwork2_checkpoint_path:
            subnetwork2_checkpoint = torch.load(subnetwork2_checkpoint_path)
            if self.data_parallel:
                self.model.module.Subnetwork2.load_state_dict(subnetwork2_checkpoint['model'])
            else:
                self.model.Subnetwork2.load_state_dict(subnetwork2_checkpoint['model'])
            print('load subnetwork2_checkpoint from {} ...'.format(subnetwork2_checkpoint_path))

    def _eval_metrics(self, m_pred, m, I_pred, I):
        acc_metrics = np.zeros(len(self.metrics) * 2)
        for i, metric in enumerate(self.metrics):
            acc_metrics[i * 2] += metric(m_pred, m)
            self.writer.add_scalar('{}_m'.format(metric.__name__), acc_metrics[i * 2])
            acc_metrics[i * 2 + 1] += metric(I_pred, I)
            self.writer.add_scalar('{}_I'.format(metric.__name__), acc_metrics[i * 2 + 1])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        # set the model to train mode
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics) * 2)

        # start training
        for batch_idx, sample in enumerate(self.data_loader):
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            # get data and send them to GPU
            prior_i = sample['prior_i'].to(self.device)
            prior_p = sample['prior_p'].to(self.device)
            m_shadow = sample['m_shadow'].to(self.device)
            I_shadow = sample['I_shadow'].to(self.device)

            m = sample['m'].to(self.device)
            I = sample['I'].to(self.device)
            p = sample['p'].to(self.device)

            # get network output
            m_pred, I_pred = self.model(prior_i, prior_p, m_shadow, I_shadow)

            p_pred = torch.clamp(m_pred / (I_pred + 1e-7), min=0, max=1)

            # visualization
            with torch.no_grad():
                if batch_idx % 100 == 0:
                    # save images to tensorboardX
                    self.writer.add_image('prior_i', make_grid(prior_i))
                    self.writer.add_image('prior_p', make_grid(prior_p))
                    self.writer.add_image('m_shadow', make_grid(m_shadow / m_shadow.max()))
                    self.writer.add_image('I_shadow', make_grid(I_shadow / 2))

                    self.writer.add_image('m_pred', make_grid(m_pred / m_pred.max()))
                    self.writer.add_image('I_pred', make_grid(I_pred / 2))
                    self.writer.add_image('p_pred', make_grid(p_pred))

                    self.writer.add_image('m', make_grid(m / m.max()))
                    self.writer.add_image('I', make_grid(I / 2))
                    self.writer.add_image('p', make_grid(p))

            # train model
            self.optimizer.zero_grad()
            model_loss = self.loss(m_pred, m, I_pred, I)
            model_loss.backward()
            self.optimizer.step()

            # calculate total loss/metrics and add scalar to tensorboard
            self.writer.add_scalar('loss', model_loss.item())
            total_loss += model_loss.item()
            total_metrics += self._eval_metrics(m_pred / m_pred.max(), m / m.max(), I_pred / 2, I / 2)

            # show current training step info
            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        model_loss.item(),  # it's a tensor, so we call .item() method
                    )
                )

        # turn the learning rate
        self.lr_scheduler.step()

        # get batch average loss/metrics as log and do validation
        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        # set the model to validation mode
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics) * 3)

        # start validating
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')

                # get data and send them to GPU
                prior_i = sample['prior_i'].to(self.device)
                prior_p = sample['prior_p'].to(self.device)
                m_shadow = sample['m_shadow'].to(self.device)
                I_shadow = sample['I_shadow'].to(self.device)

                m = sample['m'].to(self.device)
                I = sample['I'].to(self.device)
                p = sample['p'].to(self.device)

                # infer and calculate the loss
                # (N, C, H, W) GPU tensor
                m_pred, I_pred = self.model(prior_i, prior_p, m_shadow, I_shadow)
                loss = self.loss(m_pred, m, I_pred, I)

                # calculate total loss/metrics and add scalar to tensorboardX
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(m_pred / m_pred.max(), m / m.max(), I_pred / 2, I / 2)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
