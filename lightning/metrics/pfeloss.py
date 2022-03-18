import torch.nn as nn 
import torch.nn.functional as F
import torch


class MLSLoss(nn.Module):

    def __init__(self, mean = False):
        super(MLSLoss, self).__init__()
        self.mean = mean
    def negMLS(self, mu_X, sigma_sq_X):
        if self.mean:
            XX = torch.mul(mu_X, mu_X).sum(dim=1, keepdim=True)
            YY = torch.mul(mu_X.T, mu_X.T).sum(dim=0, keepdim=True)
            XY = torch.mm(mu_X, mu_X.T)
            mu_diff = XX + YY - 2 * XY
            sig_sum = sigma_sq_X.mean(dim=1, keepdim=True) + sigma_sq_X.T.sum(dim=0, keepdim=True)
            diff    = mu_diff / (1e-8 + sig_sum) + mu_X.size(1) * torch.log(sig_sum)
            return diff
        else:
            mu_diff = mu_X.unsqueeze(1) - mu_X.unsqueeze(0)
            sig_sum = sigma_sq_X.unsqueeze(1) + sigma_sq_X.unsqueeze(0)
            diff    = torch.mul(mu_diff, mu_diff) / (1e-10 + sig_sum) + torch.log(sig_sum)  # BUG
            diff    = diff.sum(dim=2, keepdim=False)
            return diff
    def forward(self, mu_X, log_sigma_sq, gty):
        mu_X     = F.normalize(mu_X) # if mu_X was not normalized by l2
        non_diag_mask = (1 - torch.eye(mu_X.size(0))).int()
        if gty.device.type == 'cuda':
            non_diag_mask = non_diag_mask.cuda(0)      
        sig_X    = torch.exp(log_sigma_sq)
        loss_mat = self.negMLS(mu_X, sig_X)
        gty_mask = (torch.eq(gty[:, None], gty[None, :])).int()
        pos_mask = (non_diag_mask * gty_mask) > 0
        pos_loss = loss_mat[pos_mask].mean()
        return pos_loss
    
class UncertaintyHead(nn.Module):
    ''' Evaluate the log(sigma^2) '''
    
    def __init__(self, in_feat = 512):

        super(UncertaintyHead, self).__init__()
        self.fc1   = nn.Parameter(torch.Tensor(in_feat, in_feat))
        self.bn1   = nn.BatchNorm1d(in_feat, affine=True)
        self.relu  = nn.ReLU(in_feat)
        self.fc2   = nn.Parameter(torch.Tensor(in_feat, in_feat))
        self.bn2   = nn.BatchNorm1d(in_feat, affine=False)
        self.gamma = nn.Parameter(torch.Tensor([1.0]))
        self.beta  = nn.Parameter(torch.Tensor([0.0]))   # default = -7.0
        
        nn.init.kaiming_normal_(self.fc1)
        nn.init.kaiming_normal_(self.fc2)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn1(F.linear(x, F.normalize(self.fc1))))
        x = self.bn2(F.linear(x, F.normalize(self.fc2)))  # 2*log(sigma)
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))  # log(sigma^2)
        return x