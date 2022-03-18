import torch
import torch.nn as nn

class CrFiqaLoss(nn.Module):
    def __init__(self,
                 in_features, out_features, s=64.0, m=0.50, alpha=10.0):
        super(CrFiqaLoss, self).__init__()
        self.metric = CrFiqaMetric(in_features, out_features, s, m)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_qs= torch.nn.SmoothL1Loss(beta=0.5)
        self.alpha= alpha
    def forward(self, features, qs, label):
        thetas, std, ccs, nnccs = self.metric(features, label)
        loss_qs = self.criterion_qs(ccs/ nnccs,qs)
        loss_v = self.criterion(thetas, label) + self.alpha* loss_qs
        return loss_v, thetas

class OnTopQS(nn.Module):
    def __init__(self,
                 num_features=512):
        super(OnTopQS, self).__init__()
        self.qs=nn.Linear(num_features,1)

    def forward(self, x):
        return self.qs(x)

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class CrFiqaMetric(nn.Module):
    r"""Implement of ArcFace:
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(CrFiqaMetric, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        with torch.no_grad():
           distmat=cos_theta[index,label.view(-1)].detach().clone()
           max_negative_cloned=cos_theta.detach().clone()
           max_negative_cloned[index,label.view(-1)]= -1e-12
           max_negative, _=max_negative_cloned.max(dim=1)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta, 0 ,distmat[index,None],max_negative[index,None]