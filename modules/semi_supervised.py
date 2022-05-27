import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Model
from .loss_coral import coral_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_pred_to_pseudo_parallel(targets_u, converter):
    blank = converter.dict[' ']
    unl_N, unl_len = targets_u.shape
    targets_u = targets_u.view(unl_N, -1)
    indexs = torch.arange(0, unl_len).repeat(unl_N, 1).to(device)
    indexs[targets_u > blank] += 10000
    eos_index = indexs.argmin(
        dim=-1)  # find smallest index of extend word(eos bos pad blank)
    eos_index[(eos_index == 0) & (targets_u[:, 0] > blank)] = unl_len - 1
    unl_length = eos_index + 1
    new_eos_index = eos_index.expand(unl_len, unl_N).permute(1, 0)
    indexs = torch.arange(0, unl_len).expand(unl_N, unl_len).to(device)
    pad_mask = (indexs - new_eos_index) > 0
    non_pad_mask = (pad_mask == False)

    cur_indexs = torch.arange(0, unl_N).to(device)
    targets_u[cur_indexs, eos_index] = converter.dict['[EOS]']
    targets_u[pad_mask] = converter.dict['[PAD]']

    # convert prediction to pseudo label
    pseudo = torch.cuda.LongTensor(unl_N, unl_len + 1)
    pseudo[:, 0] = converter.dict['[SOS]']
    pseudo[:, 1:] = targets_u
    return pseudo, non_pad_mask, unl_length


class CrossEntropyLoss(nn.Module):

    def __init__(self, opt, online_for_init_target, converter):
        super(CrossEntropyLoss, self).__init__()

        self.opt = opt
        self.converter = converter
        self.target_model = Model(opt)  # create the ema model
        self.target_model = torch.nn.DataParallel(self.target_model).to(device)
        self.target_model.train()
        self.ce_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=converter.dict['[PAD]']).to(device)
        if opt.Prediction == 'Attn':
            self.text_for_pred = torch.LongTensor(opt.batchSize).fill_(
                opt.sos_token_index).to(device)

        # copy online and init
        for param_t, param_s in zip(self.target_model.parameters(),
                                    online_for_init_target.parameters()):
            param_t.data.copy_(param_s.data)  # initialize
            param_t.requires_grad = False  # not update by gradient

        for buffer_t, buffer_s in zip(self.target_model.buffers(),
                                      online_for_init_target.buffers()):
            buffer_t.copy_(buffer_s)

    def _update_ema_variables(self, online, iteration, alpha=0.999):
        # Use the true average until the exponential average is more correct
        # alpha = min(1 - 1 / (iteration + 1), alpha)
        for param_t, param_s in zip(self.target_model.parameters(),
                                    online.parameters()):
            param_t.data = param_t.data * alpha + param_s.data * (1. - alpha)
        for buffer_t, buffer_s in zip(self.target_model.buffers(),
                                      online.buffers()):
            buffer_t.copy_(buffer_s)

    def forward(self,
                unl_img,
                aug_img,
                online,
                iteration,
                total_iter,
                l_local_feat=None,
                l_logit=None,
                l_text=None):
        loss_SemiSL = None
        l_da = None
        l_confident_ratio = 0
        self._update_ema_variables(online, iteration, self.opt.ema_alpha)
        self.target_model.eval()
        with torch.no_grad():
            unl_logit_weak = self.target_model(unl_img,
                                               text=self.text_for_pred,
                                               is_train=False)
        self.target_model.train()
        _, preds_index = unl_logit_weak.max(2)
        sequence_score, preds_index = unl_logit_weak.softmax(dim=-1).max(
            dim=-1)
        targets_u, non_pad_mask, unl_length = convert_pred_to_pseudo_parallel(
            preds_index, self.converter)
        sequence_score[non_pad_mask == False] = 10000
        batch_score = sequence_score.min(dim=-1)[0]
        mask = batch_score.ge(self.opt.confident_threshold)
        confident_ratio = mask.to(torch.float).mean()
        if iteration % 100 == 0:
            print('score', batch_score[:2])
            print('targets_u', targets_u[:2])
        if confident_ratio > 0:
            unl_logit_strong, unl_local_feats = online(aug_img,
                                                       targets_u[:, :-1],
                                                       use_project=True,
                                                       return_local_feat=True)
            unl_score, unl_index = unl_logit_strong.log_softmax(dim=-1).max(
                dim=-1)
            unl_pred, non_pad_mask, _ = convert_pred_to_pseudo_parallel(
                unl_index, self.converter)
            unl_pred = unl_pred[:, 1:]
            unl_score[non_pad_mask == False] = 0
            unl_prob = unl_score.sum(dim=-1).exp()
            unl_mask = unl_prob.ge(self.opt.confident_threshold)

            l_score, l_index = l_logit.log_softmax(dim=-1).max(dim=-1)
            l_pred, non_pad_mask, _ = convert_pred_to_pseudo_parallel(
                l_index, self.converter)
            l_pred = l_pred[:, 1:]
            l_score[non_pad_mask == False] = 0
            l_prob = l_score.sum(dim=-1).exp()
            l_mask = l_prob.ge(self.opt.l_confident_threshold)

            l_confident_ratio = l_mask.to(torch.float).mean()
            s_confident_ratio = unl_mask.to(torch.float).mean()
            if l_confident_ratio > 0 and s_confident_ratio > 0:
                l_da = coral_loss(l_local_feat[l_mask],
                                  l_pred[l_mask],
                                  unl_local_feats[unl_mask],
                                  unl_pred[unl_mask],
                                  BLANK=self.converter.dict[' '])
                if l_da is not None:
                    l_da = l_da * self.opt.lambda_mmd
            current_logit_strong = unl_logit_strong[mask].view(
                -1, unl_logit_strong.size(-1))
            current_target = targets_u[:, 1:][mask].view(-1)
            loss_SemiSL = confident_ratio * self.ce_criterion(
                current_logit_strong, current_target)
            loss_SemiSL = self.opt.lambda_cons * loss_SemiSL
        return loss_SemiSL, confident_ratio, l_da, l_confident_ratio


class KLDivLoss(nn.Module):

    def __init__(self, opt, online_for_init_target, converter):
        super(KLDivLoss, self).__init__()

        self.opt = opt
        self.converter = converter
        self.target_model = Model(opt)  # create the ema model
        self.target_model = torch.nn.DataParallel(self.target_model).to(device)
        self.target_model.train()
        self.kldiv_criterion = torch.nn.KLDivLoss(
            reduction='batchmean').to(device)
        self.text_for_pred = torch.LongTensor(opt.batchSize).fill_(
            opt.sos_token_index).to(device)

        # copy online and init
        for param_t, param_s in zip(self.target_model.parameters(),
                                    online_for_init_target.parameters()):
            param_t.data.copy_(param_s.data)  # initialize
            param_t.requires_grad = False  # not update by gradient

        for buffer_t, buffer_s in zip(self.target_model.buffers(),
                                      online_for_init_target.buffers()):
            buffer_t.copy_(buffer_s)

    def _update_ema_variables(self, online, iteration, alpha=0.999):
        # Use the true average until the exponential average is more correct
        # alpha = min(1 - 1 / (iteration + 1), alpha)
        for param_t, param_s in zip(self.target_model.parameters(),
                                    online.parameters()):
            param_t.data = param_t.data * alpha + param_s.data * (1. - alpha)
        for buffer_t, buffer_s in zip(self.target_model.buffers(),
                                      online.buffers()):
            buffer_t.copy_(buffer_s)

    def forward(self,
                unl_img,
                aug_img,
                online,
                iteration,
                total_iter,
                l_local_feat=None,
                l_logit=None,
                l_text=None):
        loss_SemiSL = None
        l_da = None
        l_confident_ratio = 0

        self._update_ema_variables(online, iteration, self.opt.ema_alpha)
        self.target_model.eval()
        with torch.no_grad():
            unl_logit = self.target_model(unl_img,
                                          text=self.text_for_pred,
                                          is_train=False)
        self.target_model.train()
        _, unl_len, nclass = unl_logit.size()

        sequence_score, preds_index = unl_logit.log_softmax(dim=-1).max(dim=-1)
        targets_u, non_pad_mask, unl_length = convert_pred_to_pseudo_parallel(
            preds_index, self.converter)
        sequence_score[non_pad_mask == False] = 0
        sample_prob = sequence_score.sum(dim=-1).exp()

        mask = sample_prob.ge(self.opt.confident_threshold)
        confident_mask = mask.view(-1, 1).repeat(1, unl_len)
        final_mask = (non_pad_mask & confident_mask)
        confident_ratio = mask.to(torch.float).mean()
        if iteration % 100 == 0:
            print('score', sample_prob[:2])
            print('targets_u', targets_u[:2])
        if confident_ratio > 0:
            unl_logit2, unl_local_feats = online(aug_img,
                                                 targets_u[:, :-1],
                                                 use_project=True,
                                                 return_local_feat=True)
            unl_score, unl_index = unl_logit2.log_softmax(dim=-1).max(dim=-1)
            unl_pred, non_pad_mask, _ = convert_pred_to_pseudo_parallel(
                unl_index, self.converter)
            unl_pred = unl_pred[:, 1:]
            unl_score[non_pad_mask == False] = 0
            unl_prob = unl_score.sum(dim=-1).exp()
            unl_mask = unl_prob.ge(self.opt.confident_threshold)

            l_score, l_index = l_logit.log_softmax(dim=-1).max(dim=-1)
            l_pred, non_pad_mask, _ = convert_pred_to_pseudo_parallel(
                l_index, self.converter)
            l_pred = l_pred[:, 1:]
            l_score[non_pad_mask == False] = 0
            l_prob = l_score.sum(dim=-1).exp()
            l_mask = l_prob.ge(self.opt.l_confident_threshold)

            l_confident_ratio = l_mask.to(torch.float).mean()
            s_confident_ratio = unl_mask.to(torch.float).mean()
            if l_confident_ratio > 0 and s_confident_ratio > 0:
                l_da = coral_loss(l_local_feat[l_mask],
                                  l_pred[l_mask],
                                  unl_local_feats[unl_mask],
                                  unl_pred[unl_mask],
                                  BLANK=self.converter.dict[' '])
                if l_da is not None:
                    l_da = l_da * self.opt.lambda_mmd
            uda_softmax_temp = self.opt.uda_softmax_temp if self.opt.uda_softmax_temp > 0 else 1
            unl_logit1 = (unl_logit.detach() /
                          uda_softmax_temp).softmax(dim=-1)
            unl_logit2 = F.log_softmax(unl_logit2, dim=-1)
            loss_SemiSL = confident_ratio * self.kldiv_criterion(
                unl_logit2[final_mask], unl_logit1[final_mask])
            loss_SemiSL = self.opt.lambda_cons * loss_SemiSL

        return loss_SemiSL, confident_ratio, l_da, l_confident_ratio
