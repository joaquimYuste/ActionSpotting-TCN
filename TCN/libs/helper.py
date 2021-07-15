import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from libs.config import Config
from libs.class_id_map import get_id2class_map
from libs.metric import AverageMeter, BoundaryScoreMeter, ScoreMeter
from libs.postprocess import PostProcessor
from libs.save_predictions import save_predictions

from tqdm import tqdm


def train(
    train_loader: DataLoader,
    model: nn.Module,
    criterion_cls: nn.Module,
    criterion_bound: nn.Module,
    lambda_bound_loss: float,
    optimizer: optim.Optimizer,
    epoch: int,
    device: str,
) -> float:
    losses = AverageMeter("Loss", ":.4e")

    # switch training mode
    model.train()

    for i, sample in enumerate(train_loader):
        x = sample["feature"]
        t = sample["label"]
        b = sample["boundary"]
        mask = sample["mask"]

        x = x.to(device)
        t = t.to(device)
        b = b.to(device)
        mask = mask.to(device)

        batch_size = x.shape[0]

        # compute output and loss
        output_cls, output_bound = model(x)

        loss = 0.0
        if isinstance(output_cls, list):
            n = len(output_cls)
            for out in output_cls:
                loss += criterion_cls(out, t, x) / n
        else:
            loss += criterion_cls(output_cls, t, x)

        if isinstance(output_bound, list):
            n = len(output_bound)
            for out in output_bound:
                loss += lambda_bound_loss * criterion_bound(out, b, mask) / n
        else:
            loss += lambda_bound_loss * criterion_bound(output_bound, b, mask)

        # record loss
        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg

def trainMSTCN(
    train_loader: DataLoader,
    model: nn.Module,
    criterion_cls: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    device: str,
) -> float:
    losses = AverageMeter("Loss", ":.4e")

    # switch training mode
    model.train()

    #for i, (feats, labels, targets) in tqdm(enumerate(train_loader):
    with tqdm(enumerate(train_loader), total=len(train_loader), ncols=160) as t:
        for i, (feats,targets) in t:
            #x = sample["feature"]
            #t = sample["label"]
            feats = feats.contiguous().view(feats.shape[0],feats.shape[2],feats.shape[1])
            feats = feats.to(device)
            targets = targets.to(device)

            batch_size, n_subclips, n_predictions, n_classes = targets.shape
            targets = targets.reshape(batch_size * n_subclips, n_predictions, n_classes)

            # compute output and loss
            output_cls = model(feats)

            loss = 0.0
            if isinstance(output_cls, list):
                n = len(output_cls)
                for out in output_cls:
                    out = out.reshape(batch_size * n_subclips, n_predictions, n_classes)
                    #loss += criterion_cls(out, targets, feats) / n
                    loss += criterion_cls(targets, out) / n
            else:
                #loss += criterion_cls(output_cls, targets, feats)
                loss += criterion_cls(targets, output_cls)

            # record loss
            losses.update(loss.item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses.avg


def validate(
    val_loader: DataLoader,
    model: nn.Module,
    criterion_cls: nn.Module,
    criterion_bound: nn.Module,
    lambda_bound_loss: float,
    device: str,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    boundary_th: float,
    tolerance: int,
) -> Tuple[float, float, float, float, float, float, float, float]:
    losses = AverageMeter("Loss", ":.4e")
    scores_cls = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )
    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in val_loader:
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)

            batch_size = x.shape[0]

            # compute output and loss
            output_cls, output_bound = model(x)

            loss = 0.0
            loss += criterion_cls(output_cls, t, x)
            loss += criterion_bound(output_bound, b, mask)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()

            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()

            # update score
            scores_cls.update(output_cls, t, output_bound, mask)
            scores_bound.update(output_bound, b, mask)

    cls_acc, edit_score, segment_f1s = scores_cls.get_scores()
    bound_acc, precision, recall, bound_f1s = scores_bound.get_scores()

    return (
        losses.avg,
        cls_acc,
        edit_score,
        segment_f1s,
        bound_acc,
        precision,
        recall,
        bound_f1s,
    )

def validateMSTCN(
    val_loader: DataLoader,
    model: nn.Module,
    criterion_cls: nn.Module,
    device: str,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
) -> Tuple[float, float, float, float, float, float, float, float]:
    losses = AverageMeter("Loss", ":.4e")
    #scores_cls = ScoreMeter(
    #    id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
    #    iou_thresholds=iou_thresholds,
    #)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad() and tqdm(enumerate(val_loader), total=len(val_loader), ncols=160) as t:
        all_outputs_cls = []
        all_targets = []
        for i, (feats, targets) in t:
            # x = sample["feature"]
            # t = sample["label"]

            #x = x.to(device)
            #t = t.to(device)

            feats = feats.contiguous().view(feats.shape[0], feats.shape[2], feats.shape[1])
            feats = feats.to(device)
            targets = targets.to(device)

            # compute output and loss
            out = model(feats)

            batch_size, n_subclips, n_predictions, n_classes = targets.shape
            targets = targets.reshape(batch_size * n_subclips, n_predictions, n_classes)
            out = out.reshape(batch_size * n_subclips, n_predictions, n_classes)

            loss = 0.0
            loss += criterion_cls(targets, out)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            # calcualte accuracy and f1 score
            out = out.to("cpu").data.numpy()

            targets = targets.to("cpu").data.numpy()

            # update metric
            #scores_cls.update(output_cls, targets)

    #cls_acc, edit_score, segment_f1s = scores_cls.get_scores()

    return (
        losses.avg,
        0, #cls_acc,
        0, #edit_score,
        [0,0,0] #segment_f1s,
    )


def evaluateASRF(
    val_loader: DataLoader,
    model: nn.Module,
    device: str,
    boundary_th: float,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    tolerance: float,
    result_path: str,
    config: Config,
    refinement_method: Optional[str] = None,
) -> None:
    postprocessor = PostProcessor(refinement_method, boundary_th)

    scores_before_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    scores_after_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for sample in val_loader:
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)

            # compute output and loss
            output_cls, output_bound = model(x)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()

            x = x.to("cpu").data.numpy()
            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()

            refined_output_cls = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )

            # update score
            scores_before_refinement.update(output_cls, t)
            scores_bound.update(output_bound, b, mask)
            scores_after_refinement.update(refined_output_cls, t)

    print("Before refinement:", scores_before_refinement.get_scores())
    print("Boundary scores:", scores_bound.get_scores())
    print("After refinement:", scores_after_refinement.get_scores())

    # save logs
    scores_before_refinement.save_scores(
        os.path.join(result_path, "test_as_before_refine.csv")
    )
    scores_before_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_before_refinement.csv")
    )

    scores_bound.save_scores(os.path.join(result_path, "test_br.csv"))

    scores_after_refinement.save_scores(
        os.path.join(result_path, "test_as_after_majority_vote.csv")
    )
    scores_after_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_after_majority_vote.csv")
    )

    model = ""
    if(config.model == "ActionSegmentRefinementFramework"):
        model = "asrf"
    elif(config.model == "ActionSegmentRefinementAttentionFramework"):
        model = "asraf"

    file_name = model+"_asb"+str(config.n_stages_asb)+"_brb"+str(config.n_stages_asb)+"_l"+str(config.n_layers)+"_"
    if(config.tmse):
        file_name += "tmse"
    else:
        file_name += "gmtse"
    file_name += "_split"+str(config.split)
    
    save_dir = os.path.join(os.path.dirname(result_path), "prediction_scores")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, model)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # save logs
    scores_after_refinement.save_scores(
        os.path.join(save_dir, "test_as_"+file_name+".csv")
    )
    scores_after_refinement.save_confusion_matrix(
        os.path.join(save_dir, "test_c_matrix_"+file_name+".csv")
    )

def evaluateMSTCN(
    val_loader: DataLoader,
    model: nn.Module,
    device: str,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    result_path: str,
    config: Config,
) -> None:
    #scores_cls = ScoreMeter(
    #    id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
    #    iou_thresholds=iou_thresholds,
    #)

    # switch to evaluate mode
    model.eval()

    model_name = ""
    if (config.model == "MultiStageTCN"):
        model_name = "mstcn"
    elif (config.model == "MultiStageAttentionTCN"):
        model_name = "msatcn"

    file_name = model_name + "_s" + str(config.n_stages) + "_l" + str(config.n_layers) + "_feats" + str(config.n_features)

    save_dir = os.path.join(os.path.dirname(result_path), "predictions")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    log = os.path.join(save_dir, "test_as_" + file_name + ".txt")
    f = open(log, 'w')
    with torch.no_grad() and tqdm(enumerate(val_loader), total=len(val_loader), ncols=160) as t:
        for i, (feats, targets) in t:
            # x = sample["feature"]
            # t = sample["label"]

            #x = x.to(device)
            #t = t.to(device)

            feats = feats.contiguous().view(feats.shape[0], feats.shape[2], feats.shape[1])
            feats = feats.to(device)

            # compute output and loss
            out = model(feats)

            batch_size, n_subclips, n_predictions, n_classes = targets.shape
            #targets = targets.reshape(batch_size * n_subclips, n_predictions, n_classes)
            #out = out.reshape(batch_size * n_subclips, n_predictions, n_classes)

            # calcualte accuracy and f1 score
            out = out.to("cpu").data.numpy()

            feats = feats.to("cpu").data.numpy()

            targets = parse_labels(targets, config)
            out = parse_labels(out, config)

            # save logs

            save_predictions(
                f,
                targets,
                out
            )
    f.close()
            # update score
            #scores.update(output_cls, t)

    #print("Scores:", scores.get_scores())

def parse_labels(labels, config):
    batch_size, n_subclips, n_predictions, n_classes = labels.shape
    n_frames = config.clip_length*2
    n_subclip_frames = int(n_frames/n_subclips)
    result = np.zeros((batch_size, n_subclips, n_subclip_frames))

    for clip in range(batch_size):
        for subclip in range(n_subclips):
            for pred in range(n_predictions):
                if(labels[clip][subclip][pred][0] >= 0.5):
                    subclip_frame = int(labels[clip][subclip][pred][1].item()) * n_subclip_frames
                    action = np.argmax(labels[clip][subclip][pred][2:]).item()

                    result[clip][subclip][subclip_frame] = action+1

    return result.reshape((batch_size, n_subclips*n_subclip_frames))
