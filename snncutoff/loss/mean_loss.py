
import torch
import torch.nn as nn


class MeanLoss(nn.Module):
    def __init__(self, criterion, *args, **kwargs):
        super(MeanLoss, self).__init__()
        self.criterion=criterion
        # self.li_update = LICell()

    def forward(self, x, y):
        # x = self.li_update(x)
        mean = x.mean(0)
        if isinstance(self.criterion, torch.nn.NLLLoss):
            log_softmax = torch.nn.LogSoftmax(dim=2)
            out = log_softmax(x)
        else:
            out = x

        loss = 0
        for t in range(x.shape[0]):
            loss = loss + self.criterion(out[t], y)
        # loss = self.criterion(mean,y)
        loss = loss/x.shape[0]
        # loss  = self.criterion(out.mean(0), y)
        # for t in range(x.shape[0]):
        #     if t>10:
        #         loss = loss + torch.softmax(out[t],dim=-1)
        # # loss = loss/x.shape[0]
        # loss = self.criterion(loss,y)


        return mean, loss