import argparse
import shutil
import os
import sys
import time
import warnings
from argparse import ArgumentParser
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import wandb
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import socket

from torch.utils.tensorboard import SummaryWriter
from snncutoff import data_loaders
from snncutoff.utils import set_seed, get_logger, load_config, dict_to_namespace, save_config
from snncutoff.ddp import reduce_mean, ProgressMeter, accuracy, AverageMeter

from snncutoff import SNNCASE
from snncutoff.API import get_model

def get_free_port():
    """Finds a free port dynamically."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def main_worker(local_rank, args):
    args.local_rank = local_rank
    if args.training.seed is not None:
        set_seed(args.training.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    best_acc = .0
    port = get_free_port()
    dist.init_process_group(backend='nccl',
                            init_method="tcp://localhost:"+str(port),
                            world_size=args.nprocs,
                            rank=args.local_rank)

    save_names = args.log+'/'+args.dataset.name + '.pth'
    checkpoint_names = args.log+'/'+args.dataset.name + '_checkpoint.pth'
    bestpoint_names = args.log+'/'+args.dataset.name + '_bestpoint.pth'

    model = get_model(args)
    if args.results.checkpoint_path != 'none':
        checkpoint = torch.load(args.results.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

    torch.cuda.set_device(local_rank)
    args.training.batch_size = int(args.training.batch_size / args.nprocs)

    # Data loading code
    train_dataset, val_dataset = data_loaders.get_data_loaders(path=args.dataset.dir, data=args.dataset.name,args=args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=False,
                                               batch_size=args.training.batch_size,
                                               num_workers=args.training.workers,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.training.batch_size,
                                             shuffle=False,
                                             num_workers=args.training.workers,
                                             pin_memory=True,
                                             sampler=val_sampler)



    model.cuda(local_rank)
    # for inputs, _ in val_dataset:
    #     inputs = val_dataset.unsqueeze(1)
    #     break  # Get only one batch
    inputs = val_dataset[0][0].unsqueeze(1)
    inputs = torch.ones_like(inputs)
    # inputs = torch.randn([1024,1,2])
    inputs = inputs.cuda(local_rank)
    output = model(inputs)
    model.cuda(local_rank)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(local_rank)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.lr)#,eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.training.epochs)
    

    if args.results.checkpoint_path != 'none':
        checkpoint = torch.load(args.results.checkpoint_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.training.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
    cudnn.benchmark = True


    # if args.evaluate:
    #     validate(val_loader, model, criterion, local_rank, args)
    #     return

    logger = get_logger(args.log+'/'+ args.dataset.name + '.log')
    logger.info('start training!')

    if args.local_rank == 0:
        if args.results.wandb_logging:
            wandb.init(config=args,name=args.log, project=args.dataset.name)
        if args.results.tensorboard_logging:
            writer = SummaryWriter(args.log)
    snncase = SNNCASE(net=model, criterion=criterion, args=args)
    for epoch in range(args.training.start_epoch, args.training.epochs):
        t1 = time.time()
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        # Create metric list
        training_metric_dic = {'Loss': [], 'Acc@1': [], 'Acc@5': []}
        custome_metric_dic  = {'cs_loss': []}
        training_metric_dic.update(custome_metric_dic)
        
        # train for one epoch
        training_metrics = train(train_loader, model, criterion, snncase, training_metric_dic, optimizer, epoch, local_rank, args)
        # evaluate on validation set
        test_metric_dic = {'Loss': [], 'Acc@1': [], 'Acc@5': []}
        custome_metric_dic  = {'cs_loss': []}
        test_metric_dic.update(custome_metric_dic)

        test_metrics = validate(val_loader, model, criterion,snncase, test_metric_dic, local_rank, args)
        scheduler.step()

        for name,training_metric in zip(list(training_metric_dic.keys()),training_metrics):
            training_metric_dic[name] = training_metric.avg

        for name,test_metric in zip(list(test_metric_dic.keys()),test_metrics):
            test_metric_dic[name] = test_metric.avg

        test_metric_dic['cs_loss'] = training_metric_dic['cs_loss']
        acc = test_metric_dic['Acc@1']
        is_best = acc >= best_acc
        best_acc = max(acc, best_acc)
        test_metric_dic['lr'] = scheduler.get_lr()[0]
        info_str = ', '.join(f"{key}: {value:.3f}" for key, value in test_metric_dic.items())
        logger.info('Epoch:[{}/{}]\t Best Acc={:.3f}\t'.format(epoch+1 , args.training.epochs, best_acc)+f"{info_str}")

        log_dic = {k:test_metric_dic[k] for k in ('Loss','Acc@1','cs_loss') if k in test_metric_dic}
        if args.local_rank == 0:
            if args.results.wandb_logging:
                wandb.log(log_dic)
            if args.results.tensorboard_logging:
                for key, value in log_dic.items():
                    writer.add_scalar('training/'+key, value, global_step=epoch)
        t2 = time.time()

        if is_best:
            if args.local_rank == 0:
                torch.save(model.module.state_dict(), save_names)

        if args.results.checkpoint_save:
            if args.local_rank == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_acc': best_acc,
                    }, is_best,checkpoint_names=checkpoint_names,bestpoint_names=bestpoint_names)


def train(train_loader, model, criterion, snncase, base_metrics, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    
    metrics = []
    # for metric in metric_list:
    for metric in list(base_metrics.keys()):
        metrics.append(AverageMeter(metric, ':.4e'))

    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time] + metrics,
                             prefix="Epoch: [{}]".format(epoch+1))
    # switch to train mode
    model.train()
    end = time.time()
    
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)
        regularization = args.regularizer.name != 'none'
        mean_out, loss = snncase.forward(images, target, regularization=regularization) 
        cs_loss = snncase.get_loss_reg()
        
        acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))

        torch.distributed.barrier()

        reduced_list = [loss,acc1,acc5]
        #custom value
        # if regularization:
        reduced_list.append(cs_loss)
        
        reduced_metrics = []
        for reduced_metric in reduced_list:
            reduced_metrics.append(reduce_mean(reduced_metric, args.nprocs))

        for metric,reduced_metric in zip(metrics, reduced_metrics):
            metric.update(reduced_metric.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # loss.backward(retain_graph=True)

        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if regularization:
            snncase.remove_hook()
            

        if i % args.training.print_freq == 0:
            progress.display(i)

    return metrics

def validate(val_loader, model, criterion, snncase, base_metrics, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')   
    metrics = []
    for metric in list(base_metrics.keys()):
        metrics.append(AverageMeter(metric, ':.4e'))
    progress = ProgressMeter(len(val_loader),
                             [batch_time] + metrics,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)
            # compute output
 
            mean_out,loss = snncase.forward(images, target,regularization=False)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_list = [loss,acc1,acc5]
            reduced_metrics = []
            for reduced_metric in reduced_list:
                reduced_metrics.append(reduce_mean(reduced_metric, args.nprocs))
            
            for metric,reduced_metric in zip(metrics, reduced_metrics):
                metric.update(reduced_metric.item(), images.size(0))

            # measure elapsed timecs1.avg, cs2.avg
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.training.print_freq == 0:
                progress.display(i)

    return metrics


def save_checkpoint(state, is_best, checkpoint_names='checkpoint.pth',bestpoint_names='best_model.pth'):
    torch.save(state, checkpoint_names)
    if is_best:
        shutil.copyfile(checkpoint_names, bestpoint_names)


def main(args):   
    if args.training.gpu_id != 'none':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.training.gpu_id
    args.nprocs = torch.cuda.device_count()

    if args.results.wandb_logging:
        wandb.login()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args,))

def update_nested_config(config, key, value):
    """Update nested config dict with dotted key like 'neuron.T'."""
    keys = key.split('.')
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Auto-cast value (int, float, bool)
    if value.lower() in ['true', 'false']:
        value = value.lower() == 'true'
    else:
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string
    d[keys[-1]] = value


if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('--override', action='append', default=[], help="Override config values like 'neuron.T=128'")

    args = parser.parse_args(sys.argv[1:])
    config = load_config(args.config)
    # Apply overrides
    for override in args.override:
        if '=' in override:
            key, value = override.split('=', 1)
            update_nested_config(config, key.strip(), value.strip())
        else:
            print(f"Invalid override format: {override}")
    args = dict_to_namespace(config)
    if config["results"]["save_dir"] == "none":
        args.results.save_dir = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    else:
        args.results.save_dir = args.dataset.name+'-'+args.training.model+'-'+args.neuron.name+'T'+str(args.neuron.T)+'G'+str(args.neuron.num_bit)+'-'+'Loss-'+args.loss.name+'-seed'+str(args.training.seed)

    args.log = args.results.save_dir_base+'/'+args.results.save_dir
    os.makedirs(args.log, exist_ok=True)
    save_config(config, args.log)
    args.neuron=config['neuron']
    args.architecture=config['architecture']
    main(args)