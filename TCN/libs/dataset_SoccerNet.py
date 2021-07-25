from torch.utils.data import Dataset

import numpy as np
import random
import os
import time


from tqdm import tqdm

import torch

import logging
import json

from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1



def feats2clip(feats, stride, clip_length, padding = "replicate_last", off=0):
    if padding =="zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
    # print(idx)
    return feats[idx,...]


def integer_enc(feat_half):
    label_half = np.ones((feat_half.shape[0], feat_half.shape[1])) #clips, frames
    label_half *= 17  # those are BG classes
    return label_half

def onehot_enc(feat_half, num_classes):
    label_half = np.zeros((feat_half.shape[0], feat_half.shape[1], num_classes+1)) #clips, frames, classes
    label_half[:, :, 17] = 1 # those are BG classes
    return label_half

def groundtruth_table(feats, n_subclips, n_predictions, num_classes):
    label_half = np.zeros((feats.shape[0], n_subclips, n_predictions, 2+num_classes))

    return label_half

def parse_groundtruth_table(window_size_frame, n_subclips, n_predictions, label_half, n_segment, frame, label):
    frames_per_subclip = int(window_size_frame/n_subclips)
    subclip = frame // frames_per_subclip
    subclip_frame = frame % frames_per_subclip # concrete subclip's frame
    offset = subclip_frame/frames_per_subclip # accordingly with the subclip

    assigned = False
    for i in range(n_predictions):
        if(label_half[n_segment][subclip][i][0] == 0 and not assigned): # if the confidence is 0 (no class) we assign the label
            label_half[n_segment][subclip][i][0] = 1
            label_half[n_segment][subclip][i][1] = offset
            label_half[n_segment][subclip][i][2+label] = 1 # onehot_class
            assigned = True

    if not assigned:
        raise ValueError("n_predictors is too low")

    return label_half

def parse_labels(self, labels):
    batch_size, n_subclips, n_predictions, n_classes = labels.shape
    n_frames = self.window_size_frame
    n_subclip_frames = int(n_frames / n_subclips)
    result = np.zeros((batch_size, n_subclips, n_subclip_frames))

    for clip in range(batch_size):
        for subclip in range(n_subclips):
            for pred in range(n_predictions):
                if (labels[clip][subclip][pred][0] >= 0.5):
                    subclip_frame = int(labels[clip][subclip][pred][1].item() * n_subclip_frames)
                    action = np.argmax(labels[clip][subclip][pred][2:]).item()

                    result[clip][subclip][subclip_frame] = action + 1

    return result.reshape((batch_size, n_subclips * n_subclip_frames))



def parse_integer_label(self, label_half, n_segment, left_segment, right_segment, left, right, label):
    label_half[n_segment][left:right] = label
    if (isinstance(left_segment, int)  and n_segment > 0):
        label_half[n_segment - 1][left_segment:] = label
    elif (isinstance(right_segment, int) and n_segment < label_half.shape[0] - 1):
        label_half[n_segment + 1][:right_segment] = label  # that's my class
    return label_half

def parse_onehot_label(self, label_half, n_segment, left_segment, right_segment, left, right, label):
    label_half[n_segment, left:right, 17] = 0     # not BG anymore
    label_half[n_segment, left:right, label] = 1  # action class
    if (isinstance(left_segment, int) and n_segment > 0):
        label_half[n_segment - 1, left_segment:, label] = 1
    elif (isinstance(right_segment, int) and n_segment < label_half.shape[0] - 1):
        label_half[n_segment + 1, :right_segment, label] = 1 # that's my class
    return label_half

class SoccerNetClips(Dataset):
    def __init__(self, data_path, label_path, features="ResNET_PCA512.npy", split=["train"], version=2,
                framerate=2, window_size=120, n_subclips=1, n_predictions=6):
        self.data_path = data_path
        self.label_path = label_path
        self.listGames = getListGames(split)
        self.features = features
        self.framerate = framerate
        self.window_size_frame = window_size*framerate
        self.version = version
        self.n_predictions = n_predictions
        self.n_subclips = n_subclips
        if version == 1:
            self.num_classes = 3
            self.labels="Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = len(EVENT_DICTIONARY_V2)
            self.labels="Labels-v2.json"

        self.class_count = np.zeros((self.num_classes))

        #logging.info("Checking/Download features and labels locally")
        #downloader = SoccerNetDownloader(path)
        #downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False,randomized=True)


        logging.info("Pre-compute clips")

        self.game_feats = list()
        self.game_labels = list()

        for game in tqdm(self.listGames):
            # Load features
            feat_half1 = np.load(os.path.join(self.data_path, game, "1_" + self.features))
            feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
            feat_half2 = np.load(os.path.join(self.data_path, game, "2_" + self.features))
            feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

            feat_half1 = feats2clip(torch.from_numpy(feat_half1), stride=self.window_size_frame, clip_length=self.window_size_frame)
            feat_half2 = feats2clip(torch.from_numpy(feat_half2), stride=self.window_size_frame, clip_length=self.window_size_frame)

            # Load labels
            labels = json.load(open(os.path.join(self.label_path, game, self.labels)))

            #if self.onehot:
            #    label_half1, label_half2 = self.onehot_enc(feat_half1), self.onehot_enc(feat_half2)
            #else:
            #    label_half1, label_half2 = self.integer_enc(feat_half1), self.integer_enc(feat_half2)

            label_half1 = groundtruth_table(feat_half1, n_subclips, n_predictions, self.num_classes)
            label_half2 = groundtruth_table(feat_half2, n_subclips, n_predictions, self.num_classes)

            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                time = seconds + 60 * minutes
                frame = framerate * time
                frame_segment = frame % self.window_size_frame
                n_segment = frame // self.window_size_frame

                if version == 1:
                    if "card" in event: label = 0
                    elif "subs" in event: label = 1
                    elif "soccer" in event: label = 2
                    else: continue
                elif version == 2:
                    if event not in self.dict_event:
                        continue
                    label = self.dict_event[event]

                # if label outside temporal of view
                if half == 1 and n_segment>=label_half1.shape[0]:
                    continue
                if half == 2 and n_segment>=label_half2.shape[0]:
                    continue

                self.class_count[label] += 1
                if half == 1:
                    label_half1 = parse_groundtruth_table(self.window_size_frame, self.n_subclips, self.n_predictions, label_half1, n_segment,  frame_segment, label)

                if half == 2:
                    label_half2 = parse_groundtruth_table(self.window_size_frame, self.n_subclips, self.n_predictions, label_half2, n_segment,  frame_segment, label)

            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)


        self.game_feats = np.concatenate(self.game_feats)
        self.game_labels = np.concatenate(self.game_labels)



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        return self.game_feats[index,:,:], self.game_labels[index,:]

    def __len__(self):
        return len(self.game_feats)


class SoccerNetClipsTesting(Dataset):
    def __init__(self, data_path, label_path, features="ResNET_PCA512.npy", split=["test"], version=2,
                framerate=2, window_size=120, n_subclips=1, n_predictions=6):
        self.data_path = data_path
        self.label_path = label_path
        self.listGames = getListGames(split)
        self.features = features
        self.window_size_frame = window_size*framerate
        self.framerate = framerate
        self.version = version
        self.split=split
        self.n_subclips = n_subclips
        self.n_predictions = n_predictions
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels="Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"

        #logging.info("Checking/Download features and labels locally")
        #downloader = SoccerNetDownloader(path)
        #for s in split:
         #   if s == "challenge":
          #      downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False,randomized=True)
         #   else:
          #      downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False,randomized=True)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        # Load features
        feat_half1 = np.load(os.path.join(self.data_path, self.listGames[index], "1_" + self.features))
        feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
        feat_half2 = np.load(os.path.join(self.data_path, self.listGames[index], "2_" + self.features))
        feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

        feat_half1 = feats2clip(torch.from_numpy(feat_half1),
                        stride=self.window_size_frame, #off=int(self.window_size_frame/2),
                        clip_length=self.window_size_frame)

        feat_half2 = feats2clip(torch.from_numpy(feat_half2),
                        stride=self.window_size_frame, #off=int(self.window_size_frame/2),
                        clip_length=self.window_size_frame)

        # Load labels
        label_half1 = groundtruth_table(feat_half1, self.n_subclips, self.n_predictions, self.num_classes)
        label_half2 = groundtruth_table(feat_half2, self.n_subclips, self.n_predictions, self.num_classes)

        # check if annoation exists
        if os.path.exists(os.path.join(self.label_path, self.listGames[index], self.labels)):
            labels = json.load(open(os.path.join(self.label_path, self.listGames[index], self.labels)))

            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = self.framerate * ( seconds + 60 * minutes )
                frame_segment = frame%self.window_size_frame
                n_segment = frame // self.window_size_frame

                if(n_segment>=label_half1.shape[0]):
                    continue

                if self.version == 1:
                    if "card" in event: label = 0
                    elif "subs" in event: label = 1
                    elif "soccer" in event: label = 2
                    else: continue
                elif self.version == 2:
                    if event not in self.dict_event:
                        continue
                    label = self.dict_event[event]

                if half == 1:
                    #frame = min(frame, feat_half1.shape[0]-1)
                    #label_half1[frame][label] = value
                    label_half1 = parse_groundtruth_table(self.window_size_frame, self.n_subclips, self.n_predictions,
                                                          label_half1, n_segment, frame_segment, label)


                if half == 2:
                    #frame = min(frame, feat_half2.shape[0]-1)
                    #label_half2[frame][label] = value
                    label_half2 = parse_groundtruth_table(self.window_size_frame, self.n_subclips, self.n_predictions,
                                                          label_half2, n_segment, frame_segment, label)

        
        return self.listGames[index], feat_half1, feat_half2, label_half1, label_half2

    def __len__(self):
        return len(self.listGames)

