# FIRST OF ALL, YOU NEED TO DOWNLOAD THE FEATURES OF THE THREE DATASET FROM:
# https://github.com/yabufarha/ms-tcn

# gtea dataset
# training test for each split
python3.7 train.py ./result/gtea/dataset-gtea_split-1/config.yaml
python3.7 evaluate.py ./result/gtea/dataset-gtea_split-1/config.yaml --refinement_method refinement_with_boundary

python3.7 train.py ./result/gtea/dataset-gtea_split-2/config.yaml
python3.7 evaluate.py ./result/gtea/dataset-gtea_split-2/config.yaml --refinement_method refinement_with_boundary

python3.7 train.py ./result/gtea/dataset-gtea_split-3/config.yaml
python3.7 evaluate.py ./result/gtea/dataset-gtea_split-3/config.yaml --refinement_method refinement_with_boundary

python3.7 train.py ./result/gtea/dataset-gtea_split-4/config.yaml
python3.7 evaluate.py ./result/gtea/dataset-gtea_split-4/config.yaml --refinement_method refinement_with_boundary

# average cross validation results.
python3.7 save_cross_val_score.py ./result/gtea/dataset-gtea_split-1/config.yaml

# average loss results.
# python3.7 ./utils/average_cv_results.py ./result/gtea/ log_mstcn.csv
python3.7 ./utils/average_cv_results.py ./result/gtea/ log_msatcn.csv
# python3.7 ./utils/average_cv_results.py ./result/gtea/ log_asraf.csv
# python3.7 ./utils/average_cv_results.py ./result/gtea/ log_asrf.csv


# Save predictions to numpy file
python3.7 save_pred.py ./result/gtea/dataset-gtea_split-2/config.yaml

# Save predictions as a png file
python3.7 utils/convert_arr2img.py ./result/gtea/dataset-gtea_split-2/predictions
