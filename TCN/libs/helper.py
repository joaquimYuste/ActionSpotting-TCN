import os
from typing import Optional, Tuple

import zipfile
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from libs.config import Config
from libs.class_id_map import get_id2class_map
from libs.metric import AverageMeter, BoundaryScoreMeter, ScoreMeter
from libs.postprocess import PostProcessor
from libs.save_predictions import save_predictions, save_preds
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.ActionSpotting import evaluate

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
    test_loader: DataLoader,
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

    split = '_'.join(test_loader.dataset.split)
    output_folder = f"outputs_{split}"

    file_name = model_name + "_s" + str(config.n_stages) + "_l" + str(config.n_layers) + "_feats" + str(config.n_features)

    save_dir = os.path.join(os.path.dirname(result_path), "predictions")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    output_zip = os.path.join(save_dir, model_name, f"results_spotting_{split}.zip")
    metrics_result = os.path.join(save_dir, model_name, f"metrics_results_{split}.txt")
    metrics_csv = os.path.join(save_dir, model_name, f"metrics_results_{split}.csv")

    save_dir = os.path.join(save_dir, model_name, output_folder)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    #log = os.path.join(save_dir, "test_as_" + file_name + ".txt")
    #f = open(log, 'w')
    #with torch.no_grad() and tqdm(enumerate(val_loader), total=len(val_loader), ncols=160) as t:
        #for i, (feats, targets) in t:
    with tqdm(enumerate(test_loader), total=len(test_loader), ncols=120) as t:
        for i, (game_ID, feat_half1, feat_half2, label_half1, label_half2) in t:
            # x = sample["feature"]
            # t = sample["label"]

            #x = x.to(device)
            #t = t.to(device)
            game_ID = game_ID[0]
            feat_half1 = feat_half1.squeeze(0)
            feat_half1 = feat_half1.contiguous().view(feat_half1.shape[0], feat_half1.shape[2], feat_half1.shape[1]) # (B, C, N)

            feat_half2 = feat_half2.squeeze(0)
            feat_half2 = feat_half2.contiguous().view(feat_half2.shape[0], feat_half2.shape[2], feat_half2.shape[1]) # (B, C, N)

            feat_half1, feat_half2 = feat_half1.to(device), feat_half2.to(device)

            label_half1 = label_half1.float().squeeze(0)
            label_half2 = label_half2.float().squeeze(0)

            #feats = feats.contiguous().view(feats.shape[0], feats.shape[2], feats.shape[1])
            #feats = feats.to(device)

            # compute output and loss
            out_half1 = model(feat_half1).to("cpu").data.numpy()
            out_half2 = model(feat_half2).to("cpu").data.numpy()

            feat_half1 = feat_half1.to("cpu").data.numpy()
            feat_half2 = feat_half2.to("cpu").data.numpy()

            json_data = dict()
            json_data["UrlLocal"] = game_ID
            json_data["predictions"] = list()

            for half, predictions in enumerate([out_half1, out_half2]):
                json_data = get_preds_info(json_data, half, predictions, test_loader.dataset
                                           .window_size_frame, test_loader.dataset.framerate)

            os.makedirs(os.path.join(save_dir, game_ID), exist_ok=True)
            with open(os.path.join(save_dir, game_ID, "results_spotting.json"), 'w') as output_file:
                json.dump(json_data, output_file, indent=4)

    def zipResults(zip_path, target_dir, filename="results_spotting.json"):
        zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
        rootlen = len(target_dir) + 1
        for base, dirs, files in os.walk(target_dir):
            for file in files:
                if file == filename:
                    fn = os.path.join(base, file)
                    zipobj.write(fn, fn[rootlen:])

    zipResults(zip_path=output_zip,
               target_dir=save_dir,
               filename="results_spotting.json")

    results = evaluate(SoccerNet_path=test_loader.dataset.label_path,
                       Predictions_path=output_zip,
                       split="test",
                       prediction_file="results_spotting.json",
                       version=test_loader.dataset.version)

    a_mAP = results["a_mAP"]
    a_mAP_per_class = results["a_mAP_per_class"]
    a_mAP_visible = results["a_mAP_visible"]
    a_mAP_per_class_visible = results["a_mAP_per_class_visible"]
    a_mAP_unshown = results["a_mAP_unshown"]
    a_mAP_per_class_unshown = results["a_mAP_per_class_unshown"]

    with open(metrics_result, 'w') as f:
        f.write("Best Performance at end of training  \n")
        f.write("a_mAP visibility all: " + str(a_mAP) + "\n")
        f.write("a_mAP visibility all per class: " + str(a_mAP_per_class) + "\n")
        f.write("a_mAP visibility visible: " + str(a_mAP_visible) + "\n")
        f.write("a_mAP visibility visible per class: " + str(a_mAP_per_class_visible) + "\n")
        f.write("a_mAP visibility unshown: " + str(a_mAP_unshown) + "\n")
        f.write("a_mAP visibility unshown per class: " + str(a_mAP_per_class_unshown) + "\n")

        f.close()

    columns = ["SoccernetNet-v2","shown","unshown"]
    for c in EVENT_DICTIONARY_V2.keys():
        columns.append(c)

    log = pd.DataFrame(columns=columns)

    tmp = [round(a_mAP*100,2), round(a_mAP_visible*100,2), round(a_mAP_unshown*100,2)]
    for score in a_mAP_per_class:
        tmp.append(round(score*100,2))

    tmp_df = pd.Series(tmp, index=log.columns)

    log = log.append(tmp_df, ignore_index=True)
    log.to_csv(metrics_csv, index=False)

    return

            #targets = targets.reshape(batch_size * n_subclips, n_predictions, n_classes)
            #out = out.reshape(batch_size * n_subclips, n_predictions, n_classes)

            # calcualte accuracy and f1 score
            #out = out.to("cpu").data.numpy()

            #targets = parse_labels(targets, config)
            #out = parse_labels(out, config)

            # save logs

            #targets = targets.reshape((targets.shape[0], targets.shape[1], 1))
            #out = out.reshape((out.shape[0], out.shape[1], 1))

            #comparison = np.concatenate((targets, out), axis=-1)
            #save_preds(f, comparison)
    #f.close()
            # update score
            #scores.update(output_cls, t)

    #print("Scores:", scores.get_scores())

def get_preds_info(json_data, half, preds, clip_length, framerate):
    clips, n_subclips, n_predictions, n_classes = preds.shape
    n_subclip_frames = int(clip_length / n_subclips)
    #result = np.zeros((batch_size, n_subclips, n_subclip_frames))
    for clip in range(clips):
        for subclip in range(n_subclips):
            for pred in range(n_predictions):
                confidence = preds[clip][subclip][pred][0]

                #if(confidence>= 0.1):
                subclip_frame = int(preds[clip][subclip][pred][1].item() * n_subclip_frames)
                clip_frame = subclip_frame + subclip * n_subclip_frames
                game_frame = clip_frame + clip * clip_length

                seconds = int((game_frame // framerate) % 60)
                minutes = int((game_frame // framerate) // 60)

                action = np.argmax(preds[clip][subclip][pred][2:]).item()

                prediction_data = dict()
                prediction_data["gameTime"] = str(half + 1) + " - " + str(minutes) + ":" + str(seconds)
                prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[action]
                prediction_data["position"] = str(int((game_frame / framerate) * 1000))
                prediction_data["half"] = str(half + 1)
                prediction_data["confidence"] = str(confidence)
                json_data["predictions"].append(prediction_data)

    return json_data


def parse_labels_to_arrays(labels, config):
    batch_size, n_subclips, n_predictions, n_classes = labels.shape
    n_frames = config.clip_length*2
    n_subclip_frames = int(n_frames/n_subclips)
    result = np.zeros((batch_size, n_subclips, n_subclip_frames))

    for clip in range(batch_size):
        for subclip in range(n_subclips):
            for pred in range(n_predictions):
                if(labels[clip][subclip][pred][0] >= 0.3):
                    subclip_frame = int(labels[clip][subclip][pred][1].item() * n_subclip_frames)
                    action = np.argmax(labels[clip][subclip][pred][2:]).item()

                    result[clip][subclip][subclip_frame] = action+1

    return result.reshape((batch_size, n_subclips*n_subclip_frames))
