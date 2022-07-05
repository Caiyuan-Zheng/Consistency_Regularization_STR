import os
import sys
import time
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import numpy as np
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from tools.dataset_test import hierarchical_dataset, AlignCollate
from tools.dataset_train import Batch_Mixed_Dataset
from model import Model
from test import validation, benchmark_all_eval
from modules.semi_supervised import CrossEntropyLoss, KLDivLoss
from utils import save_exp_config
from config_parser import parser_args
import pdb

torch.set_num_threads(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_base(opt,
               log,
               model,
               train_loader,
               valid_loader,
               lr,
               num_iter,
               criterion_SemiSL=None,
               unl_loader=None):
    # filter that only require gradient descent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print(f'Trainable params num: {sum(params_num)}')
    log.write(f'Trainable params num: {sum(params_num)}\n')

    projection_params = filter(
        lambda p: p.requires_grad,
        model.module.Prediction.projection_head.parameters())
    split_params = list(
        map(id, model.module.Prediction.projection_head.parameters()))
    base_params = filter(
        lambda p: id(p) not in split_params and p.requires_grad,
        model.parameters())

    # setup optimizer
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(filtered_parameters, lr=lr)
    elif opt.optimizer == 'adamw':
        optim_groups = [{
            'params': projection_params,
            'fix_lr': True
        }, {
            'params': base_params,
            'fix_lr': False
        }]
        optimizer = torch.optim.AdamW(optim_groups,
                                      lr=lr,
                                      weight_decay=opt.weight_decay)
    print("Optimizer:")
    print(optimizer)

    log.write(repr(optimizer) + '\n')

    if 'super' in opt.schedule:
        cycle_momentum = False

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            cycle_momentum=cycle_momentum,
            div_factor=20,
            final_div_factor=1000,
            total_steps=num_iter)
        print("Scheduler:")
        print(scheduler)
        log.write(repr(scheduler) + '\n')
    """ final options """
    # print(opt)
    opt_log = '------------ Options -------------\n'
    args = vars(opt)
    for k, v in args.items():
        if str(k) == 'character' and len(str(v)) > 500:
            opt_log += f'{str(k)}: So many characters to show all: number of characters: {len(str(v))}\n'
        else:
            opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '---------------------------------------\n'
    print(opt_log)
    log.write(opt_log)
    log.close()
    start_iter = 0
    # resum training
    if opt.saved_model != '':
        fine_tuning_flag = f'### resum model training from{opt.saved_model}\n'
        pretrained_state_dict = torch.load(opt.saved_model)
        online_dict = pretrained_state_dict['online_model']
        model.load_state_dict(online_dict, strict=False)

        if opt.semi in ['CrossEntropy', 'KLDiv']:
            target_dict = pretrained_state_dict['target_model']
            criterion_SemiSL.target_model.load_state_dict(target_dict,
                                                          strict=False)
        optimizer.load_state_dict(pretrained_state_dict['optimizer'])
        scheduler.load_state_dict(pretrained_state_dict['lr_scheduler'])
        start_iter = pretrained_state_dict['iteration']
        print(fine_tuning_flag)
    """ start training """
    start_time = time.time()
    best_score = -1

    # training loop
    ts = time.time()
    time_data = []
    for iteration in range(start_iter + 1, opt.num_iter + 1):
        t0 = time.time()
        data_dict = train_loader.get_batch()
        image = data_dict['img'].to(device)
        labels = data_dict['label']
        t1 = time.time()
        time_data.append(t1 - t0)
        labels_index, labels_length = converter.encode(
            labels, batch_max_length=opt.batch_max_length)
        # default recognition loss part
        if opt.semi in ['KLDiv', 'CrossEntropy']:
            preds, l_local_feat = model(
                image,
                labels_index[:, :-1],
                return_local_feat=True,
            )
        else:
            preds = model(
                image,
                labels_index[:, :-1],
            )

        target = labels_index[:, 1:]  # without [SOS] Symbol
        loss = criterion(preds.view(-1, preds.shape[-1]),
                         target.contiguous().view(-1))

        train_loss_avg.add(loss)
        # semi supervised part (SemiSL)
        if opt.semi == 'None':
            pass  # do nothing
        elif opt.semi in ['CrossEntropy']:
            data_dict = unl_loader.get_batch()
            unl_img_strong = data_dict['strong_img']
            unl_img_weak = data_dict['weak_img']
            unl_img_weak = unl_img_weak.to(device)
            unl_img_strong = unl_img_strong.to(device)
            loss_SemiSL, confident_ratio, l_da, l_confident_ratio = criterion_SemiSL(
                unl_img_weak,
                unl_img_strong,
                online=model,
                iteration=iteration,
                total_iter=num_iter,
                l_local_feat=l_local_feat,
                l_logit=preds,
                l_text=target)
            if confident_ratio > 0:
                loss = loss + loss_SemiSL
                semi_loss_avg.add(loss_SemiSL)
                confident_ratio_avg.add(confident_ratio)
            if l_confident_ratio > 0 and l_da is not None:
                loss = loss + l_da
                da_loss_avg.add(l_da)
                l_confident_ratio_avg.add(l_confident_ratio)
        elif opt.semi in ['KLDiv']:
            t0 = time.time()
            data_dict = unl_loader.get_batch()
            aug_img = data_dict['strong_img'].to(device)
            org_img = data_dict['weak_img'].to(device)
            t1 = time.time()
            time_data.append(t1 - t0)
            loss_SemiSL, confident_ratio, l_da, l_confident_ratio = criterion_SemiSL(
                org_img,
                aug_img,
                online=model,
                iteration=iteration,
                total_iter=num_iter,
                l_local_feat=l_local_feat,
                l_logit=preds,
                l_text=target)
            if confident_ratio > 0:
                loss = loss + loss_SemiSL
                semi_loss_avg.add(loss_SemiSL)
                confident_ratio_avg.add(confident_ratio)
            if l_confident_ratio > 0 and l_da is not None:
                loss = loss + l_da
                da_loss_avg.add(l_da)
                l_confident_ratio_avg.add(l_confident_ratio)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()
        if 'super' in opt.schedule:
            scheduler.step()
        if (iteration - 1) % opt.displayInterval == 0:
            save_dict = {
                'online_model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration
            }
            if opt.schedule in ['super']:
                save_dict['lr_sheduler'] = scheduler.state_dict()
            if opt.semi in ['KLDiv', 'CrossEntropy'
                            ] and criterion_SemiSL is not None:
                save_dict[
                    'target_model'] = criterion_SemiSL.target_model.state_dict(
                    )
            torch.save(save_dict,
                       f'{opt.checkpoint_root}/{opt.exp_name}/latest.pth')
        if iteration % 10 == 0:
            te = time.time()
            print(
                "iteration: %d/%d, train loss: %.6f; semi loss: %.6f; da loss: %.6f; l_confident_ratio: %.6f; confident ratio: %.6f; time: %.3f; time data:  %.3f;"
                % (iteration, num_iter, train_loss_avg.val(),
                   semi_loss_avg.val(), da_loss_avg.val(),
                   l_confident_ratio_avg.val(), confident_ratio_avg.val(),
                   te - ts, np.sum(time_data)))
            train_loss_avg.reset()
            semi_loss_avg.reset()
            da_loss_avg.reset()
            l_confident_ratio_avg.reset()
            confident_ratio_avg.reset()
            time_data.clear()
            ts = time.time()
        if (iteration - 1) % opt.val_interval == 0:
            # for validation log
            with open(f'{opt.checkpoint_root}/{opt.exp_name}/log_train.txt',
                      'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_score, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                if current_score > best_score:
                    best_score = current_score
                    torch.save(
                        model.state_dict(),
                        f'{opt.checkpoint_root}/{opt.exp_name}/best_score.pth')

                # validation log: loss, lr, score (accuracy or norm ED), time.
                lr = optimizer.param_groups[0]['lr']
                elapsed_time = time.time() - start_time
                valid_log = f'\n[{iteration}/{num_iter}] Train_loss: {train_loss_avg.val():0.5f}, Valid_loss: {valid_loss:0.5f}'
                valid_log += f', Semi_loss: {semi_loss_avg.val():0.5f}\n'
                valid_log += f'{"Student_current_score":17s}: {current_score:0.2f}, Current_lr: {lr:0.7f}\n'
                valid_log += f'{"Best_score":17s}: {best_score:0.2f}, Infer_time: {infer_time:0.1f}, Elapsed_time: {elapsed_time:0.1f}'

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5],
                                                confidence_score[:5]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[EOS]')]
                        pred = pred[:pred.find('[EOS]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                valid_log = f'{valid_log}\n{predicted_result_log}'
                print(valid_log)
                log.write(valid_log + '\n')
                train_loss_avg.reset()
                semi_loss_avg.reset()
    """ Evaluation at the end of training """
    print('Start evaluation on benchmark testset')
    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.makedirs(f'./evaluation_log', exist_ok=True)
    saved_best_model = f'{opt.checkpoint_root}/{opt.exp_name}/best_score.pth'
    model.load_state_dict(torch.load(f'{saved_best_model}'))

    model.eval()
    with torch.no_grad():
        total_accuracy, eval_data_list, accuracy_list = benchmark_all_eval(
            model, criterion, converter, opt)

    if opt.writer is not None:
        opt.writer.add_scalar('test/total_accuracy',
                              float(f'{total_accuracy:0.2f}'), iteration)
    for eval_data, accuracy in zip(eval_data_list, accuracy_list):
        accuracy = float(accuracy)
        if opt.writer is not None:
            opt.writer.add_scalar(f'test/{eval_data}',
                                  float(f'{accuracy:0.2f}'), iteration)

    print(
        f'finished the experiment: {opt.exp_name}, "CKLDiv_VISIBLE_DEVICES" was {opt.CKLDiv_VISIBLE_DEVICES}'
    )


def get_loaders(opt):
    unl_loader = None
    label_roots = [opt.train_1, opt.train_2]

    train_loader = Batch_Mixed_Dataset(opt, label_roots, opt.batchSize)

    if opt.semi != 'None':
        unlabel_roots = [opt.unl_train_1, opt.unl_train_2, opt.unl_train_3]
        unl_loader = Batch_Mixed_Dataset(opt,
                                         unlabel_roots,
                                         opt.unl_batchSize,
                                         learn_type='semi')

    AlignCollate_valid = AlignCollate(opt)
    valid_dataset, valid_dataset_log = hierarchical_dataset(
        root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=opt.batchSize,
                                               shuffle=False,
                                               num_workers=int(opt.workers),
                                               collate_fn=AlignCollate_valid,
                                               pin_memory=False)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    return train_loader, valid_loader, unl_loader


def test_base():
    """ Evaluation at the end of training """
    print('Start evaluation on benchmark testset')
    """ keep evaluation model and result logs """
    model.load_state_dict(torch.load(opt.saved_model),strict=False)

    model.eval()
    with torch.no_grad():
        total_accuracy, eval_data_list, accuracy_list = benchmark_all_eval(
            model, criterion, converter, opt)
    print("finish evalution")


if __name__ == "__main__":
    opt = parser_args()
    """ Seed and GPU setting """
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)  # if you are using multi-GPU.
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = True  # It fasten training.
    cudnn.deterministic = True

    opt.gpu_name = '_'.join(torch.cuda.get_device_name().split())
    if sys.platform == 'linux':
        opt.CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        opt.CUDA_VISIBLE_DEVICES = 0  # for convenience
    opt.num_gpu = torch.cuda.device_count()

    if sys.platform == 'win32':
        opt.workers = 0
    """ directory and log setting """
    if not opt.exp_name:
        opt.exp_name = f'Seed{opt.manual_seed}-{opt.model_name}'

    os.makedirs(f'{opt.checkpoint_root}/{opt.exp_name}', exist_ok=True)
    log = open(f'{opt.checkpoint_root}/{opt.exp_name}/log_train.txt', 'a')
    os.makedirs(f'./tensorboard', exist_ok=True)
    opt.writer = None
    # save_exp_config(opt.exp_name, opt.checkpoint_root)
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
        opt.sos_token_index = converter.dict['[SOS]']
        opt.eos_token_index = converter.dict['[EOS]']
    opt.num_class = len(converter.character)

    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial,
          opt.input_channel, opt.output_channel, opt.hidden_size,
          opt.num_class, opt.batch_max_length, opt.Transformation,
          opt.FeatureExtraction, opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    if opt.num_gpu > 0:
        model = torch.nn.DataParallel(model).to(device)
    model.train()

    # print("Model:")
    # print(model)
    log.write(repr(model) + '\n')
    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        # ignore [PAD] token
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=converter.dict['[PAD]']).to(device)
    assert opt.semi in ['CrossEntropy', 'KLDiv', 'None']
    if opt.semi in ['CrossEntropy']:
        criterion_SemiSL = CrossEntropyLoss(opt,
                                            online_for_init_target=model,
                                            converter=converter)
    elif opt.semi in ['KLDiv']:
        criterion_SemiSL = KLDivLoss(opt,
                                     online_for_init_target=model,
                                     converter=converter)
    else:
        criterion_SemiSL = None

    if opt.mode == "test":
        test_base()
        sys.exit()

    if opt.saved_model != '':
        fine_tuning_log = f'### loading pretrained model from {opt.saved_model}\n'
        pretrained_state_dict = torch.load(opt.saved_model)

        for name, param in model.named_parameters():
            try:
                param.data.copy_(pretrained_state_dict[name].data
                                 )  # load from pretrained model
            except:
                fine_tuning_log += f'non-pretrained layer: {name}\n'
        print(fine_tuning_log)
        log.write(fine_tuning_log + '\n')

    train_loader, valid_loader, unl_loader = get_loaders(opt)

    # loss averager
    train_loss_avg = Averager()
    semi_loss_avg = Averager()  # semi supervised loss avg
    confident_ratio_avg = Averager()
    da_loss_avg = Averager()
    l_confident_ratio_avg = Averager()
    train_base(opt,
               log,
               model,
               train_loader,
               valid_loader,
               opt.lr,
               num_iter=opt.num_iter,
               criterion_SemiSL=criterion_SemiSL,
               unl_loader=unl_loader)
