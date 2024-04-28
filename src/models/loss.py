import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_module(config):

    task = config['task']
    loss = config['loss']

    if loss == "NLLLOSS":
        return DomainAdaptLoss(reduction='none')
    if (task == "imputation") or (task == "transduction"):
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element

    if task == "classification":
        return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample
    if task == "regression":
        return nn.MSELoss(reduction='none')  # outputs loss for each batch sample
        # return BiggerMSELoss(reduction='none')
    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task))

def get_l1_loss():
    return MAELoss(reduction='none')

def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        # return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
        #                        ignore_index=self.ignore_index, reduction=self.reduction)
        return F.cross_entropy(inp, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)


class BiggerMSELoss(nn.Module):
    """ Bigger MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()


    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        delta = 2
        error = y_true - y_pred
        cond = error > 0
        big_loss = delta * torch.square(error)
        small_loss = torch.square(error)
        return torch.where(cond, big_loss, small_loss)

class DomainLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        delta = 2
        error = y_true - y_pred
        cond = error > 0
        big_loss = delta * torch.square(error)
        small_loss = torch.square(error)
        return torch.where(cond, big_loss, small_loss)

class MAELoss(nn.CrossEntropyLoss):
    """
    MAE LOSS
    """

    def forward(self, inp, target):
        return F.l1_loss(inp, target)


class DomainAdaptLoss(nn.CrossEntropyLoss):
    """
    DomainAdapt LOSS
    """

    def forward(self, class_pre, class_label):


        pre = torch.log(class_pre)
        # 计算分类损失
        loss_class = nn.NLLLoss()
        loss = loss_class(pre, class_label.long())
        # # 计算源域识别损失
        # source_pre = torch.index_select(domain_pre, dim=0, index=torch.nonzero(domain_label==0).squeeze())
        # source_label = torch.index_select(domain_label, dim=0, index=torch.nonzero(domain_label==0).squeeze())
        # loss_source_domain = nn.NLLLoss(source_pre, source_label)
        # # 计算目标域识别损失
        # target_pre = torch.index_select(domain_pre, dim=0, index=torch.nonzero(domain_label == 1).squeeze())
        # target_label = torch.index_select(domain_label, dim=0, index=torch.nonzero(domain_label == 1).squeeze())
        # loss_target_domain = nn.NLLLoss(target_pre, target_label)

        return loss
