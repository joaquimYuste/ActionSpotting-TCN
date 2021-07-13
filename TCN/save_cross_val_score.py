import argparse
import glob
import os

import numpy as np
import pandas as pd

from libs.config import get_config

def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(description="average cross validation results.")
    parser.add_argument(
        "config",
        type=str,
    )
    
    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # configuration
    config = get_config(args.config)
    
    save_dir = os.path.dirname(os.path.dirname(args.config))
    save_dir = os.path.join(save_dir, "prediction_scores")

    stages, branches = None, None
    if(config.model == "ActionSegmentRefinementFramework"):
        model = "asrf"
        branches = "_asb"+str(config.n_stages_asb)+"_brb"+str(config.n_stages_brb)
    elif(config.model == "ActionSegmentRefinementAttentionFramework"):
        model = "asraf"
        branches = "_asb"+str(config.n_stages_asb)+"_brb"+str(config.n_stages_brb)
    elif(config.model == "MultiStageTCN"):
        model = "mstcn"
        stages = "_"+"s"+str(config.n_stages)
    else:
        model = "msatcn"
        stages = "_"+"s"+str(config.n_stages)
    
    save_dir = os.path.join(save_dir, model)

    file_name = "test_as_"+model
    if(stages is not None):
        file_name += stages
    else:
        file_name += branches
    file_name += "_l"+str(config.n_layers)+"_"
    if(config.tmse):
        file_name += "tmse"
    else:
        file_name += "gmtse"
    file_name += "_split"

    score_files = glob.glob(save_dir+"/"+file_name+"*")
    print(save_dir+"/"+file_name)
    values = []
    for score_file in score_files:
        df = pd.read_csv(score_file)
        values.append(df.values.tolist())

    values = np.mean(values, axis=0)
    values = pd.DataFrame(values, columns=df.columns)
    values.to_csv(
        os.path.join(save_dir, "cross_validation_"+file_name[8:-6]+".csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
