import torch
import pdb
""" Adversarial Sequence-to-sequence Domain adaptation (https://github.com/AprilYapingZhang/Seq2SeqAdapt) """


def CORAL(source, target):
    d1 = source.data.shape[1]
    d2 = target.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d1 * d2)
    return loss


def coral_loss(source_context_history, source_pred_class,
               target_context_history, target_pred_class, BLANK):
    feature_dim = source_context_history.size()[-1]

    source_feature = source_context_history.reshape(-1, feature_dim)
    target_feature = target_context_history.reshape(-1, feature_dim)
    source_valid_char_index = (source_pred_class.reshape(-1, ) >
                               BLANK).nonzero().reshape(-1, )
    source_valid_char_feature = source_feature.reshape(
        -1, feature_dim).index_select(0, source_valid_char_index)
    target_valid_char_index = (target_pred_class.reshape(-1, ) >
                               BLANK).nonzero().reshape(-1, )
    target_valid_char_feature = target_feature.reshape(
        -1, feature_dim).index_select(0, target_valid_char_index)
    if len(source_valid_char_feature) > 0 and len(
            target_valid_char_feature) > 0:
        similarity_loss = CORAL(source_valid_char_feature,
                                target_valid_char_feature)
        return similarity_loss.mean()
    else:
        return None