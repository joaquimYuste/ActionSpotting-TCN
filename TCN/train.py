import argparse
import os
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from libs import models
from libs.checkpoint import resume, save_checkpoint
from libs.class_id_map import get_n_classes
from libs.class_weight import get_class_weight, get_pos_weight, get_class_weight_soccernet
from libs.config import get_config
from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.helper import train, trainMSTCN, validate, validateMSTCN
from libs.loss_fn import ActionSegmentationLoss, BoundaryRegressionLoss
from libs.optimizer import get_optimizer
from libs.transformer import TempDownSamp, ToTensor
from libs.dataset_SoccerNet import SoccerNetClips, SoccerNetClipsTesting
from libs.Soccernet_loss import SpottingLoss

def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="train a network for action recognition"
    )
    parser.add_argument("--config", type=str, help="path of a config file")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="a number used to initialize a pseudorandom number generator.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )

    return parser.parse_args()


def main() -> None:
    # argparser
    args = get_arguments()

    # configuration
    config = get_config(args.config)

    result_path = os.path.dirname(args.config)

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # cpu or cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Dataloader
    # Temporal downsampling is applied to only videos in 50Salads
    downsamp_rate = 2 if config.dataset == "50salads" else 1

    train_data = SoccerNetClips(
        data_path=config.features_path,
        label_path=config.labels_path,
        features=config.features_name,
        window_size=config.clip_length,
        n_subclips=config.n_subclips,
        n_predictions=config.n_predictions
    )


    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers = config.num_workers,
        drop_last=True if config.batch_size > 1 else False,
    )

    # if you do validation to determine hyperparams
    if config.param_search:
        val_data = SoccerNetClips(
            data_path=config.features_path,
            label_path=config.labels_path,
            features=config.features_name,
            window_size=config.clip_length,
            n_subclips=config.n_subclips,
            n_predictions=config.n_predictions,
            split=["valid"]
        )

        val_loader = DataLoader(
            val_data,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

    # load model
    print("---------- Loading Model ----------")

    #n_classes = get_n_classes(config.dataset, dataset_dir=config.dataset_dir)
    n_classes = 17

    if (config.model == "ActionSegmentRefinementFramework"):
        model = models.ActionSegmentRefinementFramework(
            in_channel=config.in_channel,
            n_features=config.n_features,
            n_classes=n_classes,
            n_stages=config.n_stages,
            n_layers=config.n_layers,
            n_stages_asb=config.n_stages_asb,
            n_stages_brb=config.n_stages_brb,
        )
        log_name = "log_asrf.csv"

    elif (config.model == "ActionSegmentRefinementAttentionFramework"):
        model = models.ActionSegmentRefinementAttentionFramework(
            in_channel=config.in_channel,
            n_features=config.n_features,
            n_classes=n_classes,
            n_stages=config.n_stages,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            attn_kernel=config.attn_kernel,
            n_attn_layers=config.n_attn_layers,
            n_stages_asb=config.n_stages_asb,
            n_stages_brb=config.n_stages_brb,
        )
        log_name = "log_asraf.csv"

    elif (config.model == "MultiStageTCN"):
        model = models.MultiStageTCN(
            in_channel=config.in_channel,
            n_features=config.n_features,
            n_classes=n_classes,
            n_stages=config.n_stages,
            n_layers=config.n_layers,
        )
        log_name = "log_mstcn.csv"

    elif (config.model == "MultiStageAttentionTCN"):
        model = models.MultiStageAttentionTCN(
            in_channel=config.in_channel,
            n_features=config.n_features,
            n_classes=n_classes,
            n_predictions=config.n_predictions,
            n_subclips=config.n_subclips,
            n_stages=config.n_stages,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            attn_kernel=config.attn_kernel,
            n_attn_layers=config.n_attn_layers,
        )
        log_name = "log_msatcn.csv"

        # send the model to cuda/cpu
    model.to(device)

    optimizer = get_optimizer(
        config.optimizer,
        model,
        config.learning_rate,
        momentum=config.momentum,
        dampening=config.dampening,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )

    # resume if you want
    columns = ["epoch", "lr", "train_loss"]

    # if you do validation to determine hyperparams
    if config.param_search:
        columns += ["val_loss", "cls_acc", "edit"]
        columns += [
            "segment f1s@{}".format(config.iou_thresholds[i])
            for i in range(len(config.iou_thresholds))
        ]
        if config.model == "ActionSegmentRefinementFramework" or config.model == "ActionSegmentRefinementAttentionFramework":
            columns += ["bound_acc", "precision", "recall", "bound_f1s"]

    begin_epoch = 0
    best_loss = float("inf")
    log = pd.DataFrame(columns=columns)

    if args.resume:
        if os.path.exists(os.path.join(result_path, "checkpoint.pth")):
            checkpoint = resume(result_path, model, optimizer)
            begin_epoch, model, optimizer, best_loss = checkpoint
            log = pd.read_csv(os.path.join(result_path, log_name))
            print("training will start from {} epoch".format(begin_epoch))
        else:
            print("there is no checkpoint at the result folder")

    # criterion for loss
    if config.class_weight:
        class_weight = get_class_weight_soccernet(
            train_data.game_labels,
        )
        class_weight = class_weight.to(device)
    else:
        class_weight = None

    criterion_cls = SpottingLoss(config.lambda_coord, config.lambda_noobj)
    #criterion_cls = ActionSegmentationLoss(
        #ce=config.ce,
        #focal=config.focal,
        #tmse=config.tmse,
        #gstmse=config.gstmse,
        #weight=class_weight,
        #ignore_index=255,
        #ce_weight=config.ce_weight,
        #focal_weight=config.focal_weight,
        #        tmse_weight=config.tmse_weight,
        #    gstmse_weight=config.gstmse,
    #)

    #pos_weight = get_pos_weight(
    #    dataset=config.dataset,
    #    split=config.split,
    #    csv_dir=config.csv_dir,
    #    mode="training" if config.param_search else "trainval",
    #).to(device)

    #criterion_bound = BoundaryRegressionLoss(pos_weight=pos_weight)

    # train and validate model
    print("---------- Start training ----------")
    if(config.model == "ActionSegmentRefinementFramework" or config.model == "ActionSegmentRefinementAttentionFramework"):
        for epoch in range(begin_epoch, config.max_epoch):
            # training
            train_loss = train(
                train_loader,
                model,
                criterion_cls,
                criterion_bound,
                config.lambda_b,
                optimizer,
                epoch,
                device,
            )

            # if you do validation to determine hyperparams
            if config.param_search:
                (
                    val_loss,
                    cls_acc,
                    edit_score,
                    segment_f1s,
                    bound_acc,
                    precision,
                    recall,
                    bound_f1s,
                ) = validate(
                    val_loader,
                    model,
                    criterion_cls,
                    criterion_bound,
                    config.lambda_b,
                    device,
                    config.dataset,
                    config.dataset_dir,
                    config.iou_thresholds,
                    config.boundary_th,
                    config.tolerance,
                )

                # save a model if top1 acc is higher than ever
                if best_loss > val_loss:
                    best_loss = val_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(result_path, "best_loss_model.prm"),
                    )

            # save checkpoint every epoch
            save_checkpoint(result_path, epoch, model, optimizer, best_loss)

            # write logs to dataframe and csv file
            tmp = [epoch, optimizer.param_groups[0]["lr"], train_loss]

            # if you do validation to determine hyperparams
            if config.param_search:
                tmp += [
                    val_loss,
                    cls_acc,
                    edit_score,
                ]
                tmp += segment_f1s
                tmp += [
                    bound_acc,
                    precision,
                    recall,
                    bound_f1s,
                ]

            tmp_df = pd.Series(tmp, index=log.columns)

            log = log.append(tmp_df, ignore_index=True)

            if(config.model == "ActionSegmentRefinementFramework"):
                log.to_csv(os.path.join(result_path, log_name), index=False)
            elif(config.model == "ActionSegmentRefinementAttentionFramework"):
                log.to_csv(os.path.join(result_path, log_name), index=False)

            if config.param_search:
                # if you do validation to determine hyperparams
                print(
                    "epoch: {}\tlr: {:.4f}\ttrain loss: {:.4f}\tval loss: {:.4f}\tval_acc: {:.4f}\tedit: {:.4f}".format(
                        epoch,
                        optimizer.param_groups[0]["lr"],
                        train_loss,
                        val_loss,
                        cls_acc,
                        edit_score,
                    )
                )
            else:
                print(
                    "epoch: {}\tlr: {:.4f}\ttrain loss: {:.4f}".format(
                        epoch, optimizer.param_groups[0]["lr"], train_loss
                    )
                )
    elif(config.model == "MultiStageTCN" or config.model == "MultiStageAttentionTCN"):
        for epoch in range(begin_epoch, config.max_epoch):
            # training
            train_loss = trainMSTCN(
                train_loader,
                model,
                criterion_cls,
                optimizer,
                epoch,
                device,
            )

            # if you do validation to determine hyperparams
            if config.param_search:
                (
                    val_loss,
                    cls_acc,
                    edit_score,
                    segment_f1s,
                ) = validateMSTCN(
                    val_loader,
                    model,
                    criterion_cls,
                    device,
                    config.dataset,
                    config.dataset_dir,
                    config.iou_thresholds,
                )

                # save a model if top1 acc is higher than ever
                if best_loss > val_loss:
                    best_loss = val_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(result_path, "best_loss_model.prm"),
                    )

            if (epoch % 10 == 0):
                torch.save(
                    model.state_dict(),
                    os.path.join(result_path, "model_epoch_" + str(epoch) + ".prm"),
                )
                save_checkpoint(result_path, epoch, model, optimizer, best_loss, num_epoch=True)

            # save checkpoint every epoch
            save_checkpoint(result_path, epoch, model, optimizer, best_loss)

            # write logs to dataframe and csv file
            tmp = [epoch, optimizer.param_groups[0]["lr"], train_loss]

            # if you do validation to determine hyperparams
            if config.param_search:
                tmp += [
                    val_loss,
                    cls_acc,
                    edit_score,
                ]
                tmp += segment_f1s

            tmp_df = pd.Series(tmp, index=log.columns)

            log = log.append(tmp_df, ignore_index=True)

            if(config.model == "MultiStageTCN"):
                log.to_csv(os.path.join(result_path, log_name), index=False)
            else:
                log.to_csv(os.path.join(result_path, log_name), index=False)

            if config.param_search:
                # if you do validation to determine hyperparams
                print(
                    "epoch: {}\tlr: {:.4f}\ttrain loss: {:.4f}\tval loss: {:.4f}\tval_acc: {:.4f}\tedit: {:.4f}".format(
                        epoch,
                        optimizer.param_groups[0]["lr"],
                        train_loss,
                        val_loss,
                        cls_acc,
                        edit_score,
                    )
                )
            else:
                print(
                    "epoch: {}\tlr: {:.4f}\ttrain loss: {:.4f}".format(
                        epoch, optimizer.param_groups[0]["lr"], train_loss
                    )
                )


    # delete checkpoint
    os.remove(os.path.join(result_path, "checkpoint.pth"))

    # save models
    torch.save(model.state_dict(), os.path.join(result_path, "final_model.prm"))

    print("Done!")


if __name__ == "__main__":
    main()
