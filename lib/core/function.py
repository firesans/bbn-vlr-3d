import _init_paths
from core.evaluate import accuracy, AverageMeter, FusionMatrix

import numpy as np
import torch
import time


def train_model(
    trainLoader,
    model,
    epoch,
    epoch_number,
    optimizer,
    combiner,
    criterion,
    cfg,
    logger,
    **kwargs
):
    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()

    combiner.reset_epoch(epoch)

    if cfg.LOSS.LOSS_TYPE in ['LDAMLoss', 'CSCE']:
        criterion.reset_epoch(epoch)

    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    acc = AverageMeter()
    for i, (batch) in enumerate(trainLoader):
        s = batch[1].shape[0]
        point_clouds, labels = batch[0].view(s, -1, 3), batch[1]
        # point_clouds = point_clouds.to('cuda')
        # labels = labels.to('cuda').to(torch.long)
        meta = None

        # import pdb
        # pdb.set_trace()

        loss, now_acc = combiner.forward(model, criterion, point_clouds, labels, meta=meta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_loss.update(loss.data.item(), s)
        acc.update(now_acc, s)

        if i % cfg.SHOW_STEP == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100
            )
            logger.info(pbar_str)
        print(f"done step {i+1}/{len(trainLoader)}")
    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
    )
    logger.info(pbar_str)
    return acc.avg, all_loss.avg


def valid_model(
    dataLoader, epoch_number, model, cfg, criterion, logger, device, **kwargs
):
    model.eval()
    num_classes = dataLoader.dataset.get_num_classes()
    fusion_matrix = FusionMatrix(num_classes)

    with torch.no_grad():
        all_loss = AverageMeter()
        acc = AverageMeter()
        func = torch.nn.Softmax(dim=1)
        for i, (batch) in enumerate(dataLoader):
            point_clouds, labels = batch.pos.view(s, -1, 3), batch.y
            point_clouds = point_clouds.to('cuda')
            labels = labels.to('cuda').to(torch.long)

            feature = model(point_clouds, feature_flag=True)

            output = model(feature, classifier_flag=True)
            loss = criterion(output, labels)
            score_result = func(output)

            now_result = torch.argmax(score_result, 1)
            all_loss.update(loss.data.item(), labels.shape[0])
            fusion_matrix.update(now_result.cpu().numpy(), labels.cpu().numpy())
            now_acc, cnt = accuracy(now_result.cpu().numpy(), labels.cpu().numpy())
            acc.update(now_acc, cnt)

        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_Acc:{:>5.2f}%-------".format(
            epoch_number, all_loss.avg, acc.avg * 100
        )
        logger.info(pbar_str)
    return acc.avg, all_loss.avg
