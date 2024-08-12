import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss(weight)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs, targets):
        return self.loss(self.softmax(outputs), targets)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1e-10
        logits = torch.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

class InvDiceLoss(nn.Module):
    def __init__(self):
        super(InvDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1e-10
        logits = torch.sigmoid(logits)
        iflat = 1 - logits.view(-1)
        tflat = 1 - targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

class CompleteDiceLoss(nn.Module):
    def __init__(self):
        super(CompleteDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1e-10
        logits = torch.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        inv_iflat = 1 - logits.view(-1)
        inv_tflat = 1 - targets.view(-1)
        inv_intersection = (inv_iflat * inv_tflat).sum()
            
        return 2 - ((2 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)) - ((2 * inv_intersection + smooth) / (inv_iflat.sum() + inv_tflat.sum() + smooth))

'''
class PixelWiseDIoULoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    def forward(self, m1, m2):
        # S = 1 - IoU
        I = m1 * m2
        UminusI = torch.abs(m1 - m2)
        S = UminusI/(UminusI+I)
        
        # Normalized distance between weighted centroids
        ind_i = torch.linspace(0, m1.shape[0], steps=m1.shape[0]) 
        ind_j = torch.linspace(0, m1.shape[1], steps=m1.shape[1])
        grid_i, grid_j = torch.FloatTensor(np.meshgrid(ind_i, ind_j, indexing='ij')).to(self.device)
        p_w = m1
        pgt_w = m2
        c_w1 = torch.sqrt(torch.abs(m1 - m2) * m1)
        c_w2 = torch.sqrt(torch.abs(m1 - m2) * m2)
        
        pi = torch.mean(p_w * grid_i)
        pj = torch.mean(p_w * grid_j)
        pgti = torch.mean(pgt_w * grid_i)
        pgtj = torch.mean(pgt_w * grid_j)
        
        ci1 = torch.mean(c_w1 * grid_i)
        cj1 = torch.mean(c_w1 * grid_j)
        ci2 = torch.mean(c_w2 * grid_i)
        cj2 = torch.mean(c_w2 * grid_j)
        
        D = ((pi - pgti) ** 2 + (pj - pgtj) ** 2)/((ci1 - ci2) ** 2 + (cj1 - cj2) ** 2)
        
        return S + D
'''  
''' 
class CircleDIoULoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def forward(self, x1, y1, r1, x2, y2, r2):
        d = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if r1 + r2 <= d:
            IoU = 0.
        elif r1 < r2 and d + r1 <= r2:
            IoU = (np.pi * r1 * r1)/(np.pi * r2 * r2)
        elif r2 < r1 and d + r2 <= r1:
            IoU = (np.pi * r2 * r2)/(np.pi * r1 * r1)
        else:   
            A = (r1 ** 2) * torch.acos((d ** 2 + r1 ** 2 - r2 ** 2)/(2 * d * r1))
            B = (r2 ** 2) * torch.acos((d ** 2 + r2 ** 2 - r1 ** 2)/(2 * d * r2))
            C = 0.5 * torch.sqrt((-d + r1 + r2)(d + r1 - r2)(d - r1 + r2)(d + r1 + r2))
            I = A + B - C
            U = np.pi * (r1 ** 2) + np.pi * (r2 ** 2) - I
            IoU = I/(U+0.0000001)
        
        S = torch.ones(I.shape).to(self.device) - 
        
        return torch.mean(S + d)

'''
class CircleIoULoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def forward(self, x1, y1, r1, x2, y2, r2):
        if r1 < r2:
            xs = x1
            ys = y1
            rs = r1
            xl = x2
            yl = y2
            rl = r2
        else:
            xs = x2
            ys = y2
            rs = r2
            xl = x1
            yl = y1
            rl = r1
        
        d = torch.sqrt((xs - xl) ** 2 + (ys - yl) ** 2)
        if rs + rl <= d:
            IoU = 0.
        elif d + rs <= rl:
            IoU = (np.pi * rs * rs)/(np.pi * rl * rl)
        else:
            ix = (d * d - rl * rl + rs * rs)/(2 * d)
            iy = torch.sqrt(rs * rs - ix * ix)
            theta_s = torch.asin(iy / rs)
            theta_l = torch.asin(iy / rl)
            phi_s = 2 * theta_s
            phi_l = 2 * theta_l
            
            peri_s = rs + rs + 2 * iy
            tarea_s = torch.sqrt(peri_s * (peri_s - rs) * (peri_s - rs) * (peri_s - 2 * iy))
            carea_s = ((2 * np.pi - phi_s)/(2 * np.pi)) * np.pi * rs * rs
            I_s = np.pi * rs * rs - (carea_s + tarea_s)
            
            peri_l = rl + rl + 2 * iy
            tarea_l = torch.sqrt(peri_l * (peri_l - rl) * (peri_l - rl) * (peri_l - 2 * iy))
            carea_l = ((2 * np.pi - phi_l)/(2 * np.pi)) * np.pi * rl * rl
            I_l = np.pi * rl * rl - (carea_l + tarea_l)
            
            I = I_s + I_l
            
            U = np.pi * rs * rs + np.pi * rl * rl - I
            
            IoU = I/(U + 0.0000001)
            
        S = torch.ones(IoU.shape) - IoU
        
        return torch.mean(S)

class CircleDIoULoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def forward(self, x1, y1, r1, x2, y2, r2):
        if r1 < r2:
            xs = x1
            ys = y1
            rs = r1
            xl = x2
            yl = y2
            rl = r2
        else:
            xs = x2
            ys = y2
            rs = r2
            xl = x1
            yl = y1
            rl = r1
        
        d = torch.sqrt((xs - xl) ** 2 + (ys - yl) ** 2)
        if rs + rl <= d:
            IoU = 0.
        elif d + rs <= rl:
            IoU = (np.pi * rs * rs)/(np.pi * rl * rl)
        else:
            ix = (d * d - rl * rl + rs * rs)/(2 * d)
            iy = torch.sqrt(rs * rs - ix * ix)
            theta_s = torch.asin(iy / rs)
            theta_l = torch.asin(iy / rl)
            phi_s = 2 * theta_s
            phi_l = 2 * theta_l
            
            peri_s = rs + rs + 2 * iy
            tarea_s = torch.sqrt(peri_s * (peri_s - rs) * (peri_s - rs) * (peri_s - 2 * iy))
            carea_s = ((2 * np.pi - phi_s)/(2 * np.pi)) * np.pi * rs * rs
            I_s = np.pi * rs * rs - (carea_s + tarea_s)
            
            peri_l = rl + rl + 2 * iy
            tarea_l = torch.sqrt(peri_l * (peri_l - rl) * (peri_l - rl) * (peri_l - 2 * iy))
            carea_l = ((2 * np.pi - phi_l)/(2 * np.pi)) * np.pi * rl * rl
            I_l = np.pi * rl * rl - (carea_l + tarea_l)
            
            I = I_s + I_l
            
            U = np.pi * rs * rs + np.pi * rl * rl - I
            
            IoU = I/(U + 0.0000001)
            
        S = torch.ones(IoU.shape) - IoU
        
        return torch.mean(S) + torch.mean(d / (rs + rl)) + torch.mean((rs - rl) ** 2)

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, target):
        hinge_loss = 1 - output * target
        hinge_loss[hinge_loss < 0] = 0
        return torch.mean(hinge_loss)
  
class PixelWiseIoULoss(nn.Module):
    
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def forward(self, m1_p, m2):
        #print(m1_p.shape)
        #print(m2.shape)
        # S = 1 - IoU
        m1 = torch.sigmoid(m1_p)
        I = m1 * m2
        #print(I)
        U = (m1 + m2) - I
        #print(U)
        I_sum = torch.sum(I, (2,3))
        U_sum = torch.sum(U, (2,3))
        IoU = I_sum/(U_sum+0.00000001)
        #print(I_sum.shape, U_sum.shape)
        S = 1.0 - torch.mean(IoU)
        
        return S

class PixelWiseDIoULoss(nn.Module):
    
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def forward(self, m1_p, m2):
        #print(m1_p.shape)
        #print(m2.shape)
        # S = 1 - IoU
        m1 = torch.sigmoid(m1_p)
        I = m1 * m2
        #print(I)
        U = (m1 + m2) - I
        #print(U)
        I_sum = torch.sum(I, (2,3))
        U_sum = torch.sum(U, (2,3))
        IoU = I_sum/(U_sum+0.00000001)
        #print(I_sum.shape, U_sum.shape)
        S = 1.0 - torch.mean(IoU)
        
        ind_i = torch.linspace(0, m2.shape[2], steps=m2.shape[2]) 
        ind_j = torch.linspace(0, m2.shape[3], steps=m2.shape[3])
        grid_i, grid_j = torch.FloatTensor(np.meshgrid(ind_i, ind_j, indexing='ij')).to(self.device)
        
        pi = m1 * grid_i
        pj = m1 * grid_j
        pgti = m2 * grid_i
        pgtj = m2 * grid_j
        
        c_i = torch.mean(pi)
        c_j = torch.mean(pj)
        g_c_i = torch.mean(pgti)
        g_c_j = torch.mean(pgtj)
        
        #print(pi.item(), pj.item(), pgti.item(), pgtj.item())
        
        D = ((c_i - g_c_i) ** 2 + (c_j - g_c_j) ** 2)/(m2.shape[2] ** 2 + m2.shape[3] ** 2)
        
        return S + D

''''
class DIoULossMask(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.softmax = nn.Softmax(dim=1)
    def forward(self, m1, tar_m):
        # S = 1 - IoU
        m1 = self.softmax(m1)
        tar_m = tar_m.reshape(-1, 1, tar_m.shape[1], tar_m.shape[2])
        inv_tar_m = torch.ones(tar_m.shape).to(self.device) - tar_m
        
        m2 = torch.cat([inv_tar_m, tar_m], axis=1).to(self.device)
        I = m1 * m2
        I_mean = torch.mean(I)
        #print(I)
        U = (m1 + m2) - I
        #print(U)
        U_mean = torch.mean(U)
        S = 1 - I_mean/(U_mean+0.0001)
        # Normalized distance between weighted centroids
        ind_i = torch.linspace(0, m1.shape[2], steps=m1.shape[2]) 
        ind_j = torch.linspace(0, m1.shape[3], steps=m1.shape[3])
        grid_i, grid_j = torch.FloatTensor(np.meshgrid(ind_i, ind_j, indexing='ij')).to(self.device)
        #print(m1.shape, m2.shape)
        #print(p_w.shape)
        #print(grid_i.shape)
        pi = torch.mean(m1 * grid_i, axis=0)
        pj = torch.mean(m1 * grid_j, axis=0)
        pgti = torch.mean(m2 * grid_i, axis=0)
        pgtj = torch.mean(m2 * grid_j, axis=0)
        
        #print(pi.shape, pj.shape, pgti.shape, pgtj.shape, ci1.shape, ci2.shape, cj1.shape, cj2.shape)
        
        D = ((pi - pgti) ** 2 + (pj - pgtj) ** 2)
        
        return S + torch.mean(D)
        
class DIoULoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, m1, m2):

        #m1: softmax outputs for the predicted masks
        #m2: target masks

        # S = 1 - IoU
        I = m1 * m2
        I_mean = torch.mean(I)
        #print(I)
        U = (m1 + m2) - I
        #print(U)
        U_mean = torch.mean(U)
        S = 1 - I_mean/(U_mean+0.0001)
        # Normalized distance between weighted centroids
        ind_i = torch.linspace(0, m1.shape[2], steps=m1.shape[2]) 
        ind_j = torch.linspace(0, m1.shape[3], steps=m1.shape[3])
        grid_i, grid_j = torch.FloatTensor(np.meshgrid(ind_i, ind_j, indexing='ij')).requires_grad_(False).to(self.device)
        #print(m1.shape, m2.shape)
        #print(p_w.shape)
        #print(grid_i.shape)
        pi = torch.mean(m1 * grid_i, axis=0)
        pj = torch.mean(m1 * grid_j, axis=0)
        pgti = torch.mean(m2 * grid_i, axis=0)
        pgtj = torch.mean(m2 * grid_j, axis=0)
        
        #print(pi.shape, pj.shape, pgti.shape, pgtj.shape, ci1.shape, ci2.shape, cj1.shape, cj2.shape)
        
        D = ((pi - pgti) ** 2 + (pj - pgtj) ** 2)/320
        
        return S + torch.mean(D)
      
'''
'''
def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)
'''
