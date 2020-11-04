import collections
import datetime
import logging
import os
import time
import torch
import torch.distributed as dist

from ssd.engine.inference import do_evaluation
from ssd.utils import dist_util
from ssd.utils.metric_logger import MetricLogger


def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = dist_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(cfg, model,
             data_loader,
             optimizer,
             scheduler,
             checkpointer,
             device,
             arguments,
             args):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()

    # 模型设置为train()模式，表示参数是可以进行更新的
    model.train()
    save_to_disk = dist_util.get_rank() == 0
    # 这个是关于模型训练过程中的过程记录
    if args.use_tensorboard and save_to_disk:
        import tensorboardX

        summary_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    # dataloader的大小，根据配置文件中的iteration进行训练
    # arguments = {"iteration": 0}，按照目前的理解是按照断点进行训练，这个表示的是当前的迭代次数这样
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    # 开始计时
    start_training_time = time.time()
    end = time.time()
    # 一次训练中，数据长度应该是dataloader的大小，也就是按照batchsize进行分割之后的大小
    # 数据集会返回图像和图像对应的标签，也就是(类别数目) （c+4）k，k个先验框、c个类别，然后加一个框的坐标位置
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        # print(iteration)
        # print(targets)
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = targets.to(device)
        # 把输入和目标输出传入模型，模型就会返回loss
        loss_dict = model(images, targets=targets)
        loss = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        # 这里是多GPU的操作，暂时先不用去理会
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(total_loss=losses_reduced, **loss_dict_reduced)

        # 这里是标准的反向传播的过程，传播就完事了
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 记录时间、写日志、写模型然后保存训练中的过程记录之类的，这里也基本是死的，主要找到模型就完事了
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)
        if iteration % args.log_step == 0:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                meters.delimiter.join([
                    "iter: {iter:06d}",
                    "lr: {lr:.5f}",
                    '{meters}',
                    "eta: {eta}",
                    'mem: {mem}M',
                ]).format(
                    iter=iteration,
                    lr=optimizer.param_groups[0]['lr'],
                    meters=str(meters),
                    eta=eta_string,
                    mem=round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                )
            )
            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                for loss_name, loss_item in loss_dict_reduced.items():
                    summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if iteration % args.save_step == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)

        # 目前问题主要存在这个部分，就是利用模型进行验证的过程中会报错，验证的文件有错误
        if args.eval_step > 0 and iteration % args.eval_step == 0 and not iteration == max_iter:
            eval_results = do_evaluation(cfg, model, distributed=args.distributed, iteration=iteration)
            if dist_util.get_rank() == 0 and summary_writer:
                for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                    write_metric(eval_result['metrics'], 'metrics/' + dataset, summary_writer, iteration)
            model.train()  # *IMPORTANT*: change to train mode after eval.

    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return model
