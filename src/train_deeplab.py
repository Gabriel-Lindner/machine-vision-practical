import torch.utils.data
import os
import rasterio
import torch
import numpy as np
import torch.nn.functional as F
import random
from math import cos,pi
from models.DeepLabV3_3D import DeepLabV3_3D

from SentinelDataset import SentinelDataset
from scipy.ndimage.filters import maximum_filter1d
from utils.metric import ConfusionMatrix
from utils.tracking import AverageValueMeter
import sys
from models.Unet3D import UNet
from models.SegmentationLosses import SegmentationLosses
from utils.ProgressBar import save, resume 
import wandb
import warnings

wandb.init(project="remote-sensing-image-segmentation")
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

root_path = '../data/munich480'  # root dir dataset
result_path = "../results"
resume_path = '' #'/kaggle/input/pretrained-sentinelunet/best_model.pth'
result_train = 'train_results_V3.txt'
result_validation = 'validation_results_V3.txt'

no_of_classes = 18 #5
workers = 8
batch_size = 8
h = w = 7
n_classes = 18 #400
LABEL_FILENAME = "y.tif"
best_test_acc = 0
loss = 'batch'
ottimizzatore = 'sgd'
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9
loss_weights = 'store_true'
ignore_index = 0
test_only = False
sample_duration = 30
n_epochs = 100

num_folds = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.config = {
  "learning_rate": learning_rate,
  "model": "DeepLabV3_3D",
  "epochs": n_epochs,
  "batch_size": batch_size
}


print("Starting loading Dataset...")

traindataset = SentinelDataset(root_path, tileids="tileids/train_fold0.tileids", seqlength=sample_duration)
traindataloader = torch.utils.data.DataLoader(
    traindataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last = True)
# How to iterate on a dataloader
for iteration, data in enumerate(traindataloader):
    input, target, target_ndvi, _, _ = data
    print('input temporal series with 30 images of size 13x48x48:', input.shape)
    print('target segmentation image (batchx48x48):', target.shape)
    print('target_ndvi containing 30 channels of size 48x48:', target_ndvi.shape)
    break

# Load test set
testdataset = SentinelDataset(root_path, tileids="tileids/test_fold0.tileids", seqlength=sample_duration)
testdataloader = torch.utils.data.DataLoader(
    testdataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last = True)
# Load validation set
validationdataset = SentinelDataset(root_path, tileids="tileids/eval.tileids", seqlength=sample_duration)
validationdataloader = torch.utils.data.DataLoader(
    validationdataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last = True)

numclasses = len(traindataset.classes)
labels = list(range(numclasses))


def adjust_classes_weights(cur_epoch, curr_iter, num_iter_x_epoch, tot_epochs, start_w, descending=True):
    current_iter = curr_iter + cur_epoch * num_iter_x_epoch
    max_iter = tot_epochs * num_iter_x_epoch
    # a = 0.75 - current_iter / (2 * max_iter)  # from 0.75 to 0.25
    a = 1 - current_iter / (max_iter)  # from 0 to 1
    if not descending:
        a = 1 - a    # from 0.25 to 0.75

    w = a * start_w + 1 - a
    return w


def adjust_learning_rate(cur_epoch, curr_iter, num_iter_x_epoch, tot_epochs, start_lr, lr_decay='cos'):
    current_iter = curr_iter + cur_epoch * num_iter_x_epoch
    max_iter = tot_epochs * num_iter_x_epoch

    if lr_decay == 'cos':
        lr = start_lr * (1 + cos(pi * (current_iter) / (max_iter))) / 2
    # elif lr_decay == 'step':
    #     lr = start_lr * (0.1 ** (cur_epoch // args.step))
    elif lr_decay == 'linear':
        lr = start_lr * (1 - (current_iter) / (max_iter))
    # elif lr_decay == 'schedule':
    #     count = sum([1 for s in args.schedule if s <= cur_epoch])
    #     lr = start_lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(lr_decay))

    return lr
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _remove_index(np_matrix, ignore_index=-100):
    if ignore_index == -100:
        return np_matrix
    else:
        m = np.delete(np_matrix, ignore_index, 0) # axes=0 -> delete row index
        m = np.delete(m, ignore_index, 1) # axes=1 -> delete col index
        return m


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x



def compute_train_weights(train_loader):
    beta = 0.9
    samples_per_cls = torch.zeros(n_classes)
    for batch_idx, data in enumerate(train_loader):
        inputs, targets, _, _, _ = data
        for cls in range(n_classes):
            samples_per_cls[cls] += torch.sum(targets == cls)
    max_occ = torch.max(samples_per_cls)
    weights = torch.FloatTensor(max_occ / samples_per_cls)
    # max_occ = torch.max(weights)
    # weights = torch.FloatTensor(weights / max_occ)
    if torch.cuda.is_available():
        weights = weights.cuda()

    return weights


def train_epoch(dataloader, network, optimizer, loss, ep, loss_cls, cls_weights=None):
    num_processed_samples = 0
    num_train_samples = len(dataloader.dataset)
    labels = list(range(numclasses))

    conf_mat_metrics = ConfusionMatrix(labels, ignore_class=ignore_index)
    num_cls = 18
    batch_size = 18
    labels = list(range(num_cls))
    conf_mat = ConfusionMatrix(labels, ignore_class=0)

    for e in range(5):
        target = torch.randint(num_cls, (batch_size,))
        output = torch.rand(batch_size, num_cls)
        pred = torch.argmax(output, dim=1)
        metrics_v, metrics_s = conf_mat(pred, target)
        for k, v in metrics_v.items():
            print(k, v)
        for k, v in metrics_s.items():
            print(k, v)
        print()
    loss_measure = AverageValueMeter()
    am = AverageValueMeter()
    am.add(np.array([1, 0.5, 0.2, 0.1, 0.8]))
    mean, std = am.value()
    print(mean, std, am.sum)

    am.add(np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
    mean, std = am.value()
    print(mean, std, am.sum)
    var_learning_rate = learning_rate
    for iteration, data in enumerate(dataloader):
        # with torch.no_grad():
        if optimizer == 'sgd':
            var_learning_rate = adjust_learning_rate(ep, iteration,  len(dataloader),
                                      n_epochs, learning_rate, lr_decay='cos')
            for param_group in optimizer.param_groups:
                param_group['lr'] = var_learning_rate
        if cls_weights is not None:
            loss_cls.weight = adjust_classes_weights(ep, iteration, len(dataloader),
                                      n_epochs, cls_weights, descending=False)

        optimizer.zero_grad()

        input, target, target_ndvi, _, _ = data
        num_processed_samples += len(input)
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            target_ndvi = target_ndvi.cuda()

        if loss == 'batch':
            output = network.forward(input)
            samples_per_cls = torch.stack([(target == x_u).sum() for x_u in range(n_classes)])
            l = loss(output, target, samples_per_cls)
        elif loss == 'ndvi':
            out_ndvi, output = network.forward(input)
            samples_per_cls = None
            if loss_weights:
                samples_per_cls = torch.stack([(target == x_u).sum() for x_u in range(n_classes)])
            l = loss(out_ndvi, output, target, target_ndvi, samples_per_cls, weight_mse=1.0)
        else:
            output = network.forward(input)
            l = loss(output, target)

        l.backward()
        optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(output, dim=1).squeeze(1)

            metrics_v, metrics_scalar = conf_mat_metrics(pred, target)
            str_metrics = ''.join(['%s| %f | ' % (key, value) for (key, value) in metrics_scalar.items()])
            loss_measure.add(l.item())
            str_metrics += 'loss| %f | ' % loss_measure.mean

            train_info = 'Train on | {} | Epoch| {} | [{}/{} ({:.0f}%)] | lr| {:.5f} | {}    '.format(
                dataloader.dataset.name, ep, num_processed_samples, num_train_samples,
                100. * (iteration + 1) / len(dataloader), var_learning_rate, str_metrics)
            sys.stdout.write('\r' + train_info)

    print()
    with open(os.path.join(result_path, result_train), 'a+') as f:
        f.write(train_info + '\n')


def test_epoch(dataloader, network, loss):
    num_processed_samples = 0
    num_test_samples = len(dataloader.dataset)
    labels = list(range(numclasses))
    conf_mat_metrics = ConfusionMatrix(labels, ignore_class=ignore_index)
    num_cls = 18
    batch_size = 18
    labels = list(range(num_cls))
    conf_mat = ConfusionMatrix(labels, ignore_class=0)

    for e in range(5):
        target = torch.randint(num_cls, (batch_size,))
        output = torch.rand(batch_size, num_cls)
        pred = torch.argmax(output, dim=1)
        metrics_v, metrics_s = conf_mat(pred, target)
        for k, v in metrics_v.items():
            print(k, v)
        for k, v in metrics_s.items():
            print(k, v)
        print()
    loss_measure = AverageValueMeter()
    am = AverageValueMeter()
    am.add(np.array([1, 0.5, 0.2, 0.1, 0.8]))
    mean, std = am.value()
    print(mean, std, am.sum)

    am.add(np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
    mean, std = am.value()
    print(mean, std, am.sum)

    with torch.no_grad():
        for iteration, data in enumerate(dataloader):

            input, target, target_ndvi, _, _ = data
            num_processed_samples += len(input)
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                target_ndvi = target_ndvi.cuda()

            if loss == 'batch':
                output = network.forward(input)
                samples_per_cls = torch.stack([(target == x_u).sum() for x_u in range(n_classes)])
                l = loss(output, target, samples_per_cls)
            elif loss == 'ndvi':
                out_ndvi, output = network.forward(input)
                samples_per_cls = None
                if loss_weights:
                    samples_per_cls = torch.stack([(target == x_u).sum() for x_u in range(n_classes)])
                l = loss(out_ndvi, output, target, target_ndvi, samples_per_cls)
            else:
                output = network.forward(input)
                l = loss(output, target)

            pred = torch.argmax(output, dim=1).squeeze(1)
            metrics_v, metrics_scalar = conf_mat_metrics(pred, target)
            str_metrics = ''.join(['%s| %f | ' % (key, value) for (key, value) in metrics_scalar.items()])
            loss_measure.add(l.item())
            str_metrics += 'loss| %f | ' % loss_measure.mean
            test_info = 'Test on | {} | Epoch | {} | [{}/{} ({:.0f}%)] | {}  '.format(
                dataloader.dataset.name, epoch, num_processed_samples,
                num_test_samples, 100. * (iteration + 1) / len(dataloader),
                str_metrics)
            sys.stdout.write('\r' + test_info)

        is_best = metrics_scalar['OA'] > best_test_acc
        best = '  **best result' if is_best else '         '
        test_info += best

        sys.stdout.write('\r' + test_info + '\n')
        with open(os.path.join(result_path, result_validation), 'a+') as f:
            f.write(test_info + '\n')

        if is_best:
            cls_names = np.array(traindataset.classes)[conf_mat_metrics.get_labels()]
            with open(os.path.join(result_path, "per_class_metricsDeepLab.txt"), 'a+') as f:
                f.write('classes:\n' + np.array2string(cls_names) + '\n')
                for k, v in metrics_v.items():
                    f.write(k + '\n')
                    if len(v.shape) == 1:
                        for ki, vi in zip(cls_names, v):
                            f.write("%.2f" % vi + '\t' + ki + '\n')
                    elif len(v.shape) == 2:  # confusion matrix
                        num_gt = np.sum(v, axis=1)
                        f.write('\n'.join(
                            [''.join(['{:10}'.format(item) for item in row]) + '  ' + lab + '(%d)' % tot
                             for row, lab, tot in zip(v, cls_names, num_gt)]))
                        f.write('\n')

        return metrics_scalar['OA']  # test_acc


def test_only(dataloader, network, loss, epoch, set_name):
    num_processed_samples = 0
    num_test_samples = len(dataloader.dataset)
    labels = list(range(numclasses))
    conf_mat_metrics = ConfusionMatrix(labels, ignore_class=ignore_index)
    num_cls = 18
    batch_size = 18
    labels = list(range(num_cls))
    conf_mat = ConfusionMatrix(labels, ignore_class=0)

    for e in range(5):
        target = torch.randint(num_cls, (batch_size,))
        output = torch.rand(batch_size, num_cls)
        pred = torch.argmax(output, dim=1)
        metrics_v, metrics_s = conf_mat(pred, target)
        for k, v in metrics_v.items():
            print(k, v)
        for k, v in metrics_s.items():
            print(k, v)
        print()
    loss_measure = AverageValueMeter()
    am = AverageValueMeter()
    am.add(np.array([1, 0.5, 0.2, 0.1, 0.8]))
    mean, std = am.value()
    print(mean, std, am.sum)

    am.add(np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
    mean, std = am.value()
    print(mean, std, am.sum)

    with torch.no_grad():
        for iteration, data in enumerate(dataloader):

            input, target, target_ndvi, dates, patch_id = data

            num_processed_samples += len(input)
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                target_ndvi = target_ndvi.cuda()

            if loss == 'batch':
                output = network.forward(input)
                samples_per_cls = torch.stack([(target == x_u).sum() for x_u in range(n_classes)])
                l = loss(output, target, samples_per_cls)
            elif loss == 'ndvi':
                out_ndvi, output = network.forward(input)
                samples_per_cls = None
                if loss_weights:
                    samples_per_cls = torch.stack([(target == x_u).sum() for x_u in range(n_classes)])
                l = loss(out_ndvi, output, target, target_ndvi, samples_per_cls)
                if iteration % 100 == 0:
                    write_signatures(model.module, target, output, target_ndvi, out_ndvi, dates, patch_id, set_name)
            else:
                output = network.forward(input)
                l = loss(output, target)

            pred = torch.argmax(output, dim=1).squeeze(1)
            metrics_v, metrics_scalar = conf_mat_metrics(pred, target)
            str_metrics = ''.join(['%s| %f | ' % (key, value) for (key, value) in metrics_scalar.items()])
            loss_measure.add(l.item())
            str_metrics += 'loss| %f | ' % loss_measure.mean
            test_info = 'Test on | {} | Epoch | {} | [{}/{} ({:.0f}%)] | {}  '.format(
                dataloader.dataset.name, epoch, num_processed_samples,
                num_test_samples, 100. * (iteration + 1) / len(dataloader),
                str_metrics)
            if ((100. * (iteration + 1))%100 == 0):
                sys.stdout.write('\r' + test_info)
            wandb.log({"loss": loss_measure.mean})
            

        cls_names = np.array(traindataset.classes)[conf_mat_metrics.get_labels()]
        if ((100. * (iteration + 1))%100 == 0):
            sys.stdout.write('\r' + test_info)
        with open(os.path.join(result_path, set_name + "_per_class_metricsDeepLab.txt"), 'w') as f:
            f.write('classes:\n' + np.array2string(cls_names) + '\n')
            sys.stdout.write('classes:\n' + np.array2string(cls_names) + '\n')
            for k, v in metrics_v.items():
                sys.stdout.write('\n' + k + '\n')
                f.write('\n' + k + '\n')
                if len(v.shape) == 1:
                    for ki, vi in zip(cls_names, v):
                        sys.stdout.write("%.2f" % vi + '\t' + ki + '\n')
                        f.write("%.2f" % vi + '\t' + ki + '\n')
                elif len(v.shape) == 2:  # confusion matrix
                    num_gt = np.sum(v, axis=1)
                    sys.stdout.write('\n'.join(
                        [''.join(['{:10}'.format(item) for item in row]) + '  ' + lab + '(%d)' % tot
                         for row, lab, tot in zip(v, cls_names, num_gt)]))
                    f.write('\n'.join(
                        [''.join(['{:10}'.format(item) for item in row]) + '  ' + lab + '(%d)' % tot
                         for row, lab, tot in zip(v, cls_names, num_gt)]))
                    sys.stdout.write('\n')
                    f.write('\n')

        if loss == 'ndvi':
            print("\nClass Activation Interval saved in:", result_path)

input_channels = 13          
resnet = 'resnet34_os8'    
last_activation = 'softmax' 

model = DeepLabV3_3D(num_classes = 18, input_channels = input_channels, resnet = resnet, last_activation = last_activation).cuda()
labels = list(range(numclasses))

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
#model.cuda()
if ottimizzatore == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)
elif ottimizzatore == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)

# Define Criterion
weights = None
if not (loss == 'batch' or loss == 'ndvi') and loss_weights:
    print('Computing weights per classes...')
    weights = compute_train_weights(traindataloader)
    weights = torch.sqrt(weights)
    print("weights per classes:", weights)

loss_cls = SegmentationLosses(weight=weights, cuda=True, ignore_index=ignore_index)
loss = loss_cls.build_loss(mode='ce')




start_epoch = 0
if resume_path:  # Empty strings are considered false
    print('trying to resume previous saved model...')
    state = resume(resume_path, model=model, optimizer=optimizer)

    if "epoch" in state.keys():
        start_epoch = state["epoch"]
        best_test_acc = state['best_test_acc']


for epoch in range(start_epoch, n_epochs):
    train_epoch(traindataloader, model, optimizer, loss, epoch, loss_cls, cls_weights=None)
    val_acc = test_epoch(validationdataloader, model, loss)
    wandb.log({"Val_Acc": val_acc})
    is_best = val_acc > best_test_acc
    if is_best:
        epochs_best_acc = epoch
        best_test_acc = val_acc
        if result_path:
            checkpoint_name = os.path.join(result_path, "best_model.pth")
            save(checkpoint_name, model, optimizer,
                epoch=epoch, best_test_acc=best_test_acc)
