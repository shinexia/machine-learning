#!/usr/bin/env -S torchrun --standalone

import argparse
import json
import os
import shutil
import time
import logging
from enum import Enum
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

best_acc1 = 0

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)s - %(message)s",
    level=logging.INFO,
)


def main():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument(
        "--local_rank",
        default=int(os.environ["LOCAL_RANK"]),
        type=int,
        help="node rank for distributed training",
    )
    parser.add_argument(
        "--local_world_size",
        default=int(os.environ["LOCAL_WORLD_SIZE"]),
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--model_dir",
        metavar="DIR",
        default="models",
        help="path to export models dir (default: models)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        metavar="DIR",
        default="checkpoints",
        help="path to checkpoints dir (default: checkpoints)",
    )
    parser.add_argument(
        "--tensorboard_logdir",
        metavar="DIR",
        default="tensorboard_logs",
        help="path to tensorboard logs (default: tensorboard_logs)",
    )
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="efficientnet_b1",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) +
        " (default: efficientnet_b1)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=48,
        type=int,
        metavar="N",
        help="mini-batch size (default: 48)",
    )
    parser.add_argument(
        "-p",
        "--print_freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument("--datadir", default="ai-test")
    parser.add_argument("--limit", type=int, default=0)

    FLAGS = parser.parse_args()
    logging.info("FLAGS: %s", json.dumps(FLAGS.__dict__, ensure_ascii=False))

    if not torch.cuda.is_available():
        logging.fatal("cuda not available")

    # initialize the process group
    dist.init_process_group(backend="nccl")

    if FLAGS.gpu is None:
        FLAGS.gpu = FLAGS.local_rank

    with torch.cuda.device(FLAGS.gpu):
        worker_main(FLAGS)

    dist.destroy_process_group()


def worker_main(FLAGS):
    global best_acc1

    logging.info(
        "[%d] rank = %d, world_size = %d, local_rank = %d, local_world_size = %d",
        os.getpid(),
        dist.get_rank(),
        dist.get_world_size(),
        FLAGS.local_rank,
        FLAGS.local_world_size,
    )

    # create model
    logging.info("=> creating model %s", FLAGS.arch)
    model_creator = models.__dict__[FLAGS.arch]
    model: nn.Module = model_creator()
    model = model.cuda()

    # DistributedDataParallel will divide and allocate batch_size to all
    # available GPUs if device_ids are not set
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[FLAGS.gpu])

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                0.1,
                                momentum=0.9,
                                weight_decay=1e-4)
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if FLAGS.resume:
        checkpoint_file = FLAGS.resume
        if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            best_acc1 = checkpoint["best_acc1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            logging.info(
                "=> loaded checkpoint '%s' (epoch %d)",
                checkpoint_file,
                checkpoint["epoch"],
            )
        else:
            logging.warn("=> no checkpoint found at '%s'", checkpoint_file)

    cudnn.benchmark = True

    # Data loading code
    datadir = FLAGS.datadir
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(datadir, "train"),
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    )
    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(datadir, "val"),
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, shuffle=False, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=FLAGS.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=FLAGS.workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    tensorboard_writer: SummaryWriter = None

    if dist.get_rank() == 0:
        tensorboard_writer = SummaryWriter(log_dir=FLAGS.tensorboard_logdir)

    for epoch in range(FLAGS.epochs):
        train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,
              tensorboard_writer, FLAGS)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, FLAGS)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if dist.get_rank() == 0:
            # master node
            state_dict = {
                "epoch": epoch + 1,
                "arch": FLAGS.arch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            save_checkpoint(state_dict,
                            is_best,
                            checkpoint_dir=FLAGS.checkpoint_dir)
            tensorboard_writer.flush()

    # export model
    if dist.get_rank() == 0:
        tensorboard_writer.close()

        export_models(
            model,
            model_creator,
            FLAGS.model_dir,
        )


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    tensorbaord_writer: SummaryWriter,
    FLAGS,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        if tensorbaord_writer is not None:
            tensorbaord_writer.add_scalar("Loss/train", loss, epoch)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % FLAGS.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, FLAGS):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i

                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % FLAGS.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)

    num_datasets = len(val_loader.dataset)
    num_samples = len(val_loader.sampler) * dist.get_world_size()
    num_batches = len(val_loader) + (num_samples < num_datasets)

    progress = ProgressMeter(num_batches, [batch_time, losses, top1, top5],
                             prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    top1.all_reduce()
    top5.all_reduce()

    if num_samples < num_datasets:
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(num_samples, num_datasets))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=False,
            num_workers=FLAGS.workers,
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


def save_checkpoint(state,
                    is_best,
                    filename="checkpoint.pth.tar",
                    checkpoint_dir="checkpoints"):
    filename = os.path.join(checkpoint_dir, filename)
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        os.path.join(checkpoint_dir, "model_best.pth.tar"))


def export_models(model: nn.Module, model_creator, model_dir="models"):
    logging.info("=> exporting model")
    os.makedirs(model_dir, exist_ok=True)

    # remove `module.` prefix
    new_model: nn.Module = model_creator()
    new_state_dict = OrderedDict()
    len_prefix = len("module.")
    for k, v in model.state_dict().items():
        new_state_dict[k[len_prefix:]] = v
    new_model.load_state_dict(new_state_dict)
    new_model.eval()

    dummy_input = torch.randn(10, 3, 224, 224)

    model_pt_file = os.path.join(model_dir, "model.pt")
    ts_model = torch.jit.trace(new_model, dummy_input, check_trace=False)
    logging.info("save model to: %s", model_pt_file)
    ts_model.save(model_pt_file)

    model_onnx_file = os.path.join(model_dir, "model.onnx")
    logging.info("save onnx model to: %s", model_onnx_file)
    # Export the model
    torch.onnx.export(
        new_model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        model_onnx_file,  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=15,  # the ONNX version to export the model to
        do_constant_folding=True,
        input_names=["input__0"],  # the model's input names
        output_names=["output__0"],  # the model's output names
        dynamic_axes={
            "input__0": {
                0: "batch_size"
            },
            "output__0": {
                0: "batch_size"
            },
        },
    )


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count]).cuda()
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
