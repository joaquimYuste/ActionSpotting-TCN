# Originally written by yabufarha
# https://github.com/yabufarha/ms-tcn/blob/master/model.py

from typing import Any, Optional, Tuple
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiStageTCN(nn.Module):
    """
    Y. Abu Farha and J. Gall.
    MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019

    parameters used in originl paper:
        n_features: 64
        n_stages: 4
        n_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        n_predictions: int,
        n_subclips: int,
        small_net: bool,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.n_predictions = n_predictions
        self.n_subclips = n_subclips
        self.small_net = small_net

        self.stage1 = SingleStageTCN(in_channel, n_features, n_features, n_layers)
        if self.small_net:
            self.pooling = Small1DCNN(n_features, n_features)
        self.tcn_predictor = nn.Conv1d(n_features, n_predictions * (2 + n_classes), 1)

        stages = [
            SingleStageTCN(n_features, n_features, n_features, n_layers)
            for _ in range(n_stages - 1)
        ]
        self.stages = nn.ModuleList(stages)
        predictors = [
            nn.Conv1d(n_features, n_predictions * (2 + n_classes), 1)
            for _ in range(n_stages - 1)
        ]
        self.predictors = nn.ModuleList(predictors)

        if n_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # for training
            outputs = []
            out = self.stage1(x)

            prob_out = self.generate_output(out, self.tcn_predictor)
            outputs.append(prob_out)

            for stage, predictor in zip(self.stages, self.predictors):
                out = stage(out)
                prob_out = self.generate_output(out, predictor)
                outputs.append(prob_out)
            return outputs
        else:
            # for evaluation
            out = self.stage1(x)
            prob_out = self.generate_output(out, self.tcn_predictor)
            for stage, predictor in zip(self.stages, self.predictors):
                out = stage(out)
                prob_out = self.generate_output(out, predictor)

            return prob_out

    def generate_output(self, x: torch.Tensor, predictor):
        batch, chann, length = x.shape

        if self.small_net:
            subclip_length = length // self.n_subclips

            out = torch.zeros(batch, chann, self.n_subclips)
            if x.is_cuda:
                out = out.to("cuda")

            for subclip in range(self.n_subclips):
                initial_frame = subclip*subclip_length
                final_frame = (subclip+1)*subclip_length
                out[:, :, subclip] = self.pooling(x[:, :, initial_frame:final_frame]).reshape((batch,chann))

        else:
            out = F.avg_pool1d(x, length//self.n_subclips, length//self.n_subclips) # out: [Batch_size, filters, n_subclips]

        out = predictor(out)

        batch, chann, length = out.shape
        out = out.contiguous().view(batch, length, chann)
        out = out.reshape((batch, length, self.n_predictions, self.n_classes + 2))

        out_prob_softmax = F.softmax(out[..., 2:], dim=-1)
        out_prob_sigmoid = F.sigmoid(out[...,:2])
        out_prob = torch.cat((out_prob_sigmoid, out_prob_softmax), dim=-1)

        return out_prob # out: [Batch_size, n_subclips, n_predictions, confidence + offset + n_classes)

class MultiStageAttentionTCN(nn.Module):

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        n_heads: int,
        attn_kernel: int,
        n_attn_layers: int,
        n_subclips: int,
        n_predictions: int,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.n_predictions = n_predictions
        self.n_subclips = n_subclips

        self.stage1 = SingleStageTCN(in_channel, n_features, n_features, n_layers)
        self.tcn_predictor = nn.Conv1d(n_features, n_predictions * (2 + n_classes), 1)

        stages = [
            SingleStageTCN(n_features, n_features, n_features, n_layers)
            for _ in range(n_stages - 1)
        ]
        self.stages = nn.ModuleList(stages)

        predictors = [
            nn.Conv1d(n_features, n_predictions * (2 + n_classes), 1)
            for _ in range(n_stages - 1)
        ]
        self.predictors = nn.ModuleList(predictors)

        # Attention Module
        self.attn = SingleStageTAN(n_features, n_features, n_features, attn_kernel, n_attn_layers, n_heads=n_heads)
        self.attn_predictor = nn.Conv1d(n_features, n_predictions * (2 + n_classes), 1)

        if n_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # for training
            outputs = []
            out = self.stage1(x)
            #outputs.append(out)

            prob_out = self.generate_output(out, self.tcn_predictor)
            outputs.append(prob_out)

            for stage, predictor in zip(self.stages, self.predictors):
                out = stage(out)
                prob_out = self.generate_output(out, predictor)
                outputs.append(prob_out)

            out = self.attn(out)

            prob_out = self.generate_output(out, self.attn_predictor)
            outputs.append(prob_out)

            return outputs

        else:
            # for evaluation
            out = self.stage1(x)

            for stage in self.stages:
                out = stage(out)

            out = self.attn(out)

            prob_out = self.generate_output(out, self.attn_predictor)
            return prob_out

    def generate_output(self, x: torch.Tensor, predictor):
        batch, chann, length = x.shape
        out = F.avg_pool1d(x, int(length / self.n_subclips), int(length / self.n_subclips))
        out = predictor(out)

        batch, chann, length = out.shape
        out = out.contiguous().view(batch, length, chann)
        out = out.reshape((batch, length, self.n_predictions, self.n_classes + 2))

        out_prob_softmax = F.softmax(out[..., 2:], dim=-1)
        out_prob_sigmoid = F.sigmoid(out[...,:2])
        out_prob = torch.cat((out_prob_sigmoid, out_prob_softmax), dim=-1)

        return out_prob


class SingleStageTCN(nn.Module):
    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_layers: int,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.conv_in = None
        if(in_channel != n_features):
            self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        layers = [
            DilatedResidualLayer(2 ** i, n_features, n_features)
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.conv_out = None
        if(n_features != n_classes):
            self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if(self.conv_in):
            out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        if(self.conv_out):
            out = self.conv_out(out)
        return out

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, in_channel: int, out_channels: int) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channel, out_channels, 3, padding=dilation, dilation=dilation,
        )
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)
        return x + out


class SingleStageTransformer(nn.Module):
    def __init__(
        self,
        in_channel: int,
        n_features: int,
        out_channels: int,
        n_layers: int,
        n_heads: int = 4,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)

        self.pos_enc = PositionalEncoding(n_features)
        layers = [
            TransformerLayer(n_features, n_features, n_heads)
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv1d(n_features, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)

        out = self.pos_enc(out) * math.sqrt(x.size(1))
        for layer in self.layers:
            out, weights = layer(out)

        out = self.conv_out(out)

        return (out, weights)

class TransformerLayer(nn.Module):
    def __init__(self, in_channel: int, out_channels: int, n_heads: int) -> None:
        super().__init__()
        assert out_channels % n_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.multihead_selfattn = nn.MultiheadAttention(out_channels, n_heads)

        #MLP part
        self.mlp = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1),
            nn.Dropout(0.3),
            NormalizedReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
        )

        self.norm2 = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch, chann, temp = x.size()

        qkv = x.permute(2,0,1)
        attn_out, weights = self.multihead_selfattn(qkv, qkv, qkv, need_weights=True)
        attn_out = attn_out.permute(1,2,0)

        out = attn_out + x

        out = self.norm1(out)

        mlp_out = self.mlp(out)

        out = mlp_out+out

        out = self.norm2(out)

        return out, weights

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        n_features: int,
        max_len: int = 4000,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout()

        pe = torch.zeros(max_len, n_features)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_features, 2).float() * (-math.log(10000.0)/n_features))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.permute(2,0,1)
        out = out + self.pe[:out.size(0), :]
        return self.dropout(out.permute(1,2,0))


class SingleStageTAN(nn.Module):
    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        kernel_size: int or list,
        n_layers: int,
        n_heads: int = 1,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.conv_in = None
        if(in_channel != n_features):
            self.conv_in = nn.Conv1d(in_channel, n_features, 1)

        # To use with different kernel sizes nor dilation
        if type(kernel_size) is list:
            assert len(kernel_size) == n_layers, "The number of kernels must be equal to the layers"

            layers = [
                ConvolutionalAttnLayer(n_features, n_features, kernel_size[i], groups=n_heads)
                for i in range(n_layers)
            ]
        # Equal kernel size for each layer and dilation mode activated
        else:
            layers = [
                ConvolutionalAttnLayer(n_features, n_features, kernel_size, dilation = 2 ** i, groups=n_heads)
                for i in range(n_layers)
            ]
        self.layers = nn.ModuleList(layers)

        self.conv_out = None
        if(n_features != n_classes):
            self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if(self.conv_in):
            out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        if(self.conv_out):
            out = self.conv_out(out)
        return out

class ConvolutionalAttnLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True
    ) -> None:
        super().__init__()

        assert kernel_size % 2 != 0, "kernel size must be an odd number. (example: kernel_size: 3)"
        assert out_channels % groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.out_channels = out_channels
        self.kernel_size = (dilation-1)*(kernel_size-1)+kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2
        self.groups = groups

        if self.dilation is not 1:
            self.dilation_mask = torch.FloatTensor([1 if i%self.dilation==0 else 0 for i in range(self.kernel_size)])

        self.rel_enc = nn.Parameter(torch.randn(out_channels, 1, kernel_size), requires_grad=True)

        # Dilated Positional Encoding
        self.padded_rel_enc = self.pad_within(self.rel_enc, self.kernel_size, self.dilation)

        self.query_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, groups=groups)#, bias=bias)
        self.key_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, groups=groups)#, bias=bias)
        self.value_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, groups=groups)#, bias=bias)

        self.mlp= nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1),
            nn.Dropout(0.3),
            NormalizedReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
        )
        self.dropout = nn.Dropout()

    def pad_within(self, x, kernel, dilation):
        out = torch.zeros(x.size(0),x.size(1),kernel)
        with torch.no_grad():
            for i in range(x.size(2)):
                out[:,:,i*dilation] = x[:,:,i]
        return out


    def forward(self, x):
        if(x.is_cuda):
            self.padded_rel_enc = self.padded_rel_enc.to("cuda")

        batch, channels, length = x.size()

        padded_x = F.pad(x, (self.padding, self.padding))       # We pad temporal length for each size

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        # Convolutional patches
        q_out = q_out.unfold(2, 1, self.stride)                 # Query: [B, C, L, 1]
        k_out = k_out.unfold(2, self.kernel_size, self.stride)  # Keys: [B, C, L, K]
        v_out = v_out.unfold(2, self.kernel_size, self.stride)  # Values: [B, C, L, K]

        # Positional encoding added
        k_out = k_out + self.padded_rel_enc

        if self.dilation is not 1:
            if(x.is_cuda):
                self.dilation_mask = self.dilation_mask.to("cuda")
            k_out = k_out*self.dilation_mask
            v_out = v_out*self.dilation_mask

        q_out = q_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, length, -1) # Query: [B, G, C//G, L, 1]
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, length, -1) # Keys: [B, G, C//G, L, K]
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, length, -1) # Values: [B, G, C//G, L, K]

        attn = (q_out * k_out).sum(dim=2, keepdims=True)
        attn = F.softmax(attn, dim=-1)
        out = (attn * v_out).sum(dim=-1).view(batch, -1, length) # Out: [B, C, T]

        out = F.relu(out)
        out = self.mlp(out)
        out = self.dropout(out)
        return out + x

class NormalizedReLU(nn.Module):
    """
    Normalized ReLU Activation prposed in the original TCN paper.
    the values are divided by the max computed per frame
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x /= x.max(dim=1, keepdim=True)[0] + self.eps

        return x


class EDTCN(nn.Module):
    """
    Encoder Decoder Temporal Convolutional Network
    """

    def __init__(
        self,
        in_channel: int,
        n_classes: int,
        kernel_size: int = 25,
        mid_channels: Tuple[int, int] = [128, 160],
        **kwargs: Any
    ) -> None:
        """
        Args:
            in_channel: int. the number of the channels of input feature
            n_classes: int. output classes
            kernel_size: int. 25 is proposed in the original paper
            mid_channels: list. the list of the number of the channels of the middle layer.
                        [96 + 32*1, 96 + 32*2] is proposed in the original paper
        Note that this implementation only supports n_layer=2
        """
        super().__init__()

        # encoder
        self.enc1 = nn.Conv1d(
            in_channel,
            mid_channels[0],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout1 = nn.Dropout(0.3)
        self.relu1 = NormalizedReLU()

        self.enc2 = nn.Conv1d(
            mid_channels[0],
            mid_channels[1],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout2 = nn.Dropout(0.3)
        self.relu2 = NormalizedReLU()

        # decoder
        self.dec1 = nn.Conv1d(
            mid_channels[1],
            mid_channels[1],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout3 = nn.Dropout(0.3)
        self.relu3 = NormalizedReLU()

        self.dec2 = nn.Conv1d(
            mid_channels[1],
            mid_channels[0],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout4 = nn.Dropout(0.3)
        self.relu4 = NormalizedReLU()

        self.conv_out = nn.Conv1d(mid_channels[0], n_classes, 1, bias=True)

        self.init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder 1
        x1 = self.relu1(self.dropout1(self.enc1(x)))
        t1 = x1.shape[2]
        x1 = F.max_pool1d(x1, 2)

        # encoder 2
        x2 = self.relu2(self.dropout2(self.enc2(x1)))
        t2 = x2.shape[2]
        x2 = F.max_pool1d(x2, 2)

        # decoder 1
        x3 = F.interpolate(x2, size=(t2,), mode="nearest")
        x3 = self.relu3(self.dropout3(self.dec1(x3)))

        # decoder 2
        x4 = F.interpolate(x3, size=(t1,), mode="nearest")
        x4 = self.relu4(self.dropout4(self.dec2(x4)))

        out = self.conv_out(x4)

        return out

    def init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class ActionSegmentRefinementFramework(nn.Module):
    """
    this model predicts both frame-level classes and boundaries.
    Args:
        in_channel: 2048
        n_feature: 64
        n_classes: the number of action classes
        n_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        n_stages_asb: Optional[int] = None,
        n_stages_brb: Optional[int] = None,
        **kwargs: Any
    ) -> None:

        if not isinstance(n_stages_asb, int):
            n_stages_asb = n_stages

        if not isinstance(n_stages_brb, int):
            n_stages_brb = n_stages

        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        shared_layers = [
            DilatedResidualLayer(2 ** i, n_features, n_features)
            for i in range(n_layers)
        ]
        self.shared_layers = nn.ModuleList(shared_layers)
        self.conv_cls = nn.Conv1d(n_features, n_classes, 1)
        self.conv_bound = nn.Conv1d(n_features, 1, 1)

        # action segmentation branch
        asb = [
            SingleStageTCN(n_classes, n_features, n_classes, n_layers)
            for _ in range(n_stages_asb - 1)
        ]

        # boundary regression branch
        brb = [
            SingleStageTCN(1, n_features, 1, n_layers) for _ in range(n_stages_brb - 1)
        ]
        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.conv_in(x)
        for layer in self.shared_layers:
            out = layer(out)

        out_cls = self.conv_cls(out)
        out_bound = self.conv_bound(out)

        if self.training:
            outputs_cls = [out_cls]
            outputs_bound = [out_bound]

            for as_stage in self.asb:
                out_cls = as_stage(self.activation_asb(out_cls))
                outputs_cls.append(out_cls)

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound))
                outputs_bound.append(out_bound)

            return (outputs_cls, outputs_bound)
        else:
            for as_stage in self.asb:
                out_cls = as_stage(self.activation_asb(out_cls))

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound))

            return (out_cls, out_bound)

class ActionSegmentRefinementAttentionFramework(nn.Module):

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        n_heads: int,
        attn_kernel: int,
        n_attn_layers: int,
        n_stages_asb: Optional[int] = None,
        n_stages_brb: Optional[int] = None,
        **kwargs: Any
    ) -> None:

        if not isinstance(n_stages_asb, int):
            n_stages_asb = n_stages

        if not isinstance(n_stages_brb, int):
            n_stages_brb = n_stages

        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        shared_layers = [
            DilatedResidualLayer(2 ** i, n_features, n_features)
            for i in range(n_layers)
        ]
        self.shared_layers = nn.ModuleList(shared_layers)

        self.conv_cls = nn.Conv1d(n_features, n_classes, 1)
        self.conv_bound = nn.Conv1d(n_features, 1, 1)

        self.attn_cls = SingleStageTAN(n_classes, n_features, n_classes, attn_kernel, n_attn_layers, n_heads=n_heads)

        # action segmentation branch
        asb = [
            SingleStageTCN(n_classes, n_features, n_classes, n_layers)
            for _ in range(n_stages_asb - 1)
        ]

        # boundary regression branch
        brb = [
            SingleStageTCN(1, n_features, 1, n_layers) for _ in range(n_stages_brb - 1)
        ]
        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.conv_in(x)
        for layer in self.shared_layers:
            out = layer(out)


        out_cls = self.conv_cls(out)
        out_bound = self.conv_bound(out)

        outputs_cls = [out_cls]
        outputs_bound = [out_bound]

        if self.training:

            for as_stage in self.asb:
                out_cls = as_stage(self.activation_asb(out_cls))
                outputs_cls.append(out_cls)

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound))
                outputs_bound.append(out_bound)


            out_cls = self.attn_cls(self.activation_asb(out_cls))
            outputs_cls.append(out_cls)

            return (outputs_cls, outputs_bound)
        else:
            for as_stage in self.asb:
                out_cls = as_stage(self.activation_asb(out_cls))

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound))

            out_cls = self.attn_cls(self.activation_asb(out_cls))
            return (out_cls, out_bound)


class Small1DCNN(nn.Module):
    def __init__(self, input_dim, model_dim, dropout: float = 0.1, avg_pool: bool = True):
        """
        Simple 1D CNN applying 3 conv1d layers, each followed by maxpool with size 2.
        The temporal length of the input sequences will be divided by 8.
        Adaptive average pooling is applied (optionally) at the end so the remaining timesteps (8x smaller than the
        original sequence length) are average pooled into 1 timestep.
        :param input_dim: [batch_size x features x input_dim]
        :param model_dim:
        :param avg_pool:
        """
        super(Small1DCNN, self).__init__()

        self.conv1d_1 = nn.Conv1d(input_dim, model_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool1d_1 = nn.MaxPool1d(2)
        self.activation_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout)

        self.conv1d_2 = nn.Conv1d(model_dim, model_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool1d_2 = nn.MaxPool1d(2)
        self.activation_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

        self.conv1d_3 = nn.Conv1d(model_dim, model_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool1d_3 = nn.MaxPool1d(2)
        self.activation_3 = nn.ReLU()
        self.dropout_3 = nn.Dropout(dropout)

        self.avg_pool = None
        if avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)  # pool along temporal dimension (last one)

    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.pool1d_1(x)
        x = self.activation_1(x)
        x = self.dropout_1(x)

        x = self.conv1d_2(x)
        x = self.pool1d_2(x)
        x = self.activation_2(x)
        x = self.dropout_2(x)

        x = self.conv1d_3(x)
        x = self.pool1d_3(x)
        x = self.activation_3(x)
        x = self.dropout_3(x)

        if self.avg_pool is not None:
            x = self.avg_pool(x)

        return x