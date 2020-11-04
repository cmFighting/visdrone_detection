import argparse
import logging
import os

import torch
import torch.distributed as dist

from ssd.engine.inference import do_evaluation
from ssd.config import cfg
from ssd.data.build import make_data_loader
from ssd.engine.trainer import do_train
# 这里是比较关键的一步，通过这步来构建目标检测模型
from ssd.modeling.detector import build_detection_model
from ssd.solver.build import make_optimizer, make_lr_scheduler
from ssd.utils import dist_util, mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
from ssd.utils.misc import str2bool


# 训练主函数
def train(cfg, args):
    # 工厂模式，加载日志文件设置，这里暂时不同管
    logger = logging.getLogger('SSD.trainer')
    # 建立目标检测模型
    model = build_detection_model(cfg)
    # 设置Device并且把模型部署到设备上
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # 设置学习率、优化器还有学习率变化步长，可以理解为模拟退火这种，前面的步长比较大，后面的步长比较小
    lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    optimizer = make_optimizer(cfg, model, lr)

    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    arguments = {"iteration": 0}
    save_to_disk = dist_util.get_rank() == 0
    # **** 这里应该是从断点开始对模型进行训练 ****
    checkpointer = CheckPointer(model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    # Important 通过torch的形式去加载数据集
    # 关键在于如何加载数据集，模型的构建过程可以简单地看成是黑盒
    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
    train_loader = make_data_loader(cfg, is_train=True, distributed=args.distributed, max_iter=max_iter, start_iter=arguments['iteration'])

    # 正式开始训练, 暂时先不训练？
    # 不对，不训练也得加载数据集**** 暂时不训练就完事了 *** 直接看数据加载过程
    # model = do_train(cfg, model, train_loader, optimizer, scheduler, checkpointer, device, arguments, args)
    return model


def main():
    # 解析命令行 读取配置文件
    '''
    规定了模型的基本参数，训练的类，一共是20类加上背景所以是21
    模型的输入大小，为了不对原图造成影响，一般是填充为300*300的图像
    训练的文件夹路径2007和2012，测试的文件夹路径2007
    最大迭代次数为120000.学习率还有gamma的值，总之就是一系列的超参数
    输出的文件目录
    MODEL:
        NUM_CLASSES: 21
    INPUT:
        IMAGE_SIZE: 300
    DATASETS:
        TRAIN: ("voc_2007_trainval", "voc_2012_trainval")
        TEST: ("voc_2007_test", )
    SOLVER:
        MAX_ITER: 120000
        LR_STEPS: [80000, 100000]
        GAMMA: 0.1
        BATCH_SIZE: 32
        LR: 1e-3
    OUTPUT_DIR: 'outputs/vgg_ssd300_voc0712'
    Returns:
    '''
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config-file",
        default="configs/vgg_ssd300_voc0712.yaml",
        # default="configs/vgg_ssd300_visdrone0413.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    # 每2500步保存一次文件，并且验证一次文件，记录是每10次记录一次，然后如果不想看tensor的记录的话，可以关闭，使用的是tensorboardX
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=2500, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=2500, type=int, help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # 参数解析，可以使用多GPU进行训练
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    # 做一些启动前必要的检查
    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 创建模型输出文件夹
    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    # 使用logger来进行记录
    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    # 加载配置文件
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # 模型训练
    # model = train(cfg, args)
    model = train(cfg, args)

    # 开始进行验证
    if not args.skip_test:
        logger.info('Start evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(cfg, model, distributed=args.distributed)


if __name__ == '__main__':
    main()
