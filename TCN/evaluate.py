import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from libs import models
from libs.class_id_map import get_n_classes
from libs.config import get_config
from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.helper import evaluateASRF, evaluateMSTCN
from libs.transformer import TempDownSamp, ToTensor
from libs.dataset_SoccerNet import SoccerNetClips, SoccerNetClipsTesting
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2


def get_arguments():
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="evaluation for action segment refinement network."
    )
    parser.add_argument("config", type=str, help="path to a config file")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="""
            path to the trained model.
            If you do not specify, the trained model,
            'final_model.prm' in result directory will be used.
            """,
    )
    parser.add_argument(
        "--refinement_method",
        type=str,
        default="refinement_with_boundary",
        choices=["refinement_with_boundary", "relabeling", "smoothing"],
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Add --cpu option if you use cpu."
    )

    return parser.parse_args()


def main():
    args = get_arguments()

    # configuration
    config = get_config(args.config)

    result_path = os.path.dirname(args.config)

    # cpu or gpu
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.backends.cudnn.benchmark = True

    # Dataloader
    downsamp_rate = 2 if config.dataset == "50salads" else 1

    data = SoccerNetClipsTesting(
        data_path=config.features_path,
        label_path=config.labels_path,
        features=config.features_name,
        window_size=config.clip_length,
        n_subclips=config.n_subclips,
        n_predictions=config.n_predictions,
        split=["test"]
    )

    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # load model
    print("---------- Loading Model ----------")

    #n_classes = get_n_classes(config.dataset, dataset_dir=config.dataset_dir)
    n_classes = len(EVENT_DICTIONARY_V2)
    if(config.model == "ActionSegmentRefinementFramework"):
        model = models.ActionSegmentRefinementFramework(
            in_channel=config.in_channel,
            n_features=config.n_features,
            n_classes=n_classes,
            n_stages=config.n_stages,
            n_layers=config.n_layers,
            n_stages_asb=config.n_stages_asb,
            n_stages_brb=config.n_stages_brb,
        )
    
    elif(config.model == "ActionSegmentRefinementAttentionFramework"):
        model = models.ActionSegmentRefinementAttentionFramework(
            in_channel=config.in_channel,
            n_features=config.n_features,
            n_classes=n_classes,
            n_stages=config.n_stages,
            n_layers=config.n_layers,
            n_heads = config.n_heads,
            attn_kernel = config.attn_kernel,
            n_attn_layers = config.n_attn_layers,
            n_stages_asb=config.n_stages_asb,
            n_stages_brb=config.n_stages_brb,
        )

    elif(config.model == "MultiStageTCN"):
        model = models.MultiStageTCN(
            in_channel = config.in_channel,
            n_features = config.n_features,
            n_classes = n_classes,
            n_subclips=config.n_subclips,
            n_predictions=config.n_predictions,
            n_stages = config.n_stages,
            n_layers = config.n_layers,
            small_net = config.small_net,
        )

    elif(config.model == "MultiStageAttentionTCN"):
        model = models.MultiStageAttentionTCN(
            in_channel = config.in_channel,
            n_features = config.n_features,
            n_classes = n_classes,
            n_stages = config.n_stages,
            n_layers = config.n_layers,
            n_heads = config.n_heads,
            attn_kernel = config.attn_kernel,
            n_attn_layers = config.n_attn_layers,
            n_subclips = config.n_subclips,
            n_predictions = config.n_predictions
        )


    # send the model to cuda/cpu
    model.to(device)

    # load the state dict of the model
    if args.model is not None:
        state_dict = torch.load(os.path.join(result_path, args.model))
    else:
        state_dict = torch.load(os.path.join(result_path, "final_model.prm"))
    model.load_state_dict(state_dict)

    # train and validate model
    print("---------- Start testing ----------")

    results = None
    # evaluation
    if(config.model == "ActionSegmentRefinementFramework" or config.model == "ActionSegmentRefinementAttentionFramework"):
        evaluateASRF(
            loader,
            model,
            device,
            config.boundary_th,
            config.dataset,
            config.dataset_dir,
            config.iou_thresholds,
            config.tolerance,
            result_path,
            config,
            args.refinement_method,
        )
    else:
        evaluateMSTCN(
            loader,
            model,
            device,
            config.dataset,
            config.dataset_dir,
            config.iou_thresholds,
            result_path,
            config,
        )


    print("Done")


if __name__ == "__main__":
    main()
