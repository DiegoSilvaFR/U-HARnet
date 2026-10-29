import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Union, List, Dict, Any, cast
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
class Classifier(nn.Module):
    def __init__(self, input_size=1024, output_size=2):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred

class ProjectionHead(nn.Module):
    def __init__(self, input_size=1024, nn_size=256, encoding_size=100):
        super(ProjectionHead, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, encoding_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class EvaClassifier(nn.Module):
    def __init__(self, input_size=1024, nn_size=512, output_size=2):
        super(EvaClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
    
class DropoutHead(nn.Module):

    def __init__(self, input_size=1024, nn_size=512, output_size=1, p=0.5):
        super().__init__()
        if not 0.0 < p < 1.0:
            raise ValueError("MC dropout probability must be strictly between 0 and 1.")
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, output_size)
        self.p = p

    def forward(self, x, mc_dropout=False):
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.p, training= mc_dropout)
        return self.linear2(x)


class AccNet(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 5,
        classifier_input_size: int = 1,
        classifier_layer_size: int = 2048,
        init_weights: bool = True,
    ) -> None:
        super(AccNet, self).__init__()
        self.features = features

        if init_weights:
            self._initialize_weights()

    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = cnn1()

    def forward(self, x):
        feats = self.feature_extractor(x)

        return feats


class SSLNET(nn.Module):
    def __init__(
        self, output_size=2, input_size=1024, number_nn=1024, flatten_size=1024
    ):
        super(SSLNET, self).__init__()
        self.feature_extractor = cnn1()
        self.rotation_classifier = Classifier(
            input_size=flatten_size, output_size=output_size
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        rotation_y = self.rotation_classifier(feats)
        # axis_y = self.axis_classifier(feats)

        return rotation_y


def make_layers(
    cfg: List[Union[str, int]], batch_norm: bool = False
) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    i = 0
    while i < len(cfg):
        v = cfg[i]
        if v == "M":
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            i += 1
            my_kernel_size = cfg[i]

            v = cast(int, v)
            my_kernel_size = cast(int, my_kernel_size)

            conv1d = nn.Conv1d(
                in_channels, v, kernel_size=my_kernel_size, padding=1
            )
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
        i += 1
    return nn.Sequential(*layers)


class ConvBNReLU(nn.Module):
    """Convolution + batch normalization + ReLU is a common trio"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    ):
        super(ConvBNReLU, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.main(x)


class CNN(nn.Module):
    """Typical CNN design with pyramid-like structure"""

    def __init__(self, in_channels=3, num_filters_init=3):
        super(CNN, self).__init__()

        self.layer1 = ConvBNReLU(
            in_channels, num_filters_init, 3, 1, 1, bias=False
        )  # 900 -> 225
        self.layer2 = ConvBNReLU(
            num_filters_init, num_filters_init, 3, 1, 1, bias=False
        )  # 225 -> 56

        self.layer3 = ConvBNReLU(
            num_filters_init, num_filters_init * 2, 3, 1, 1, bias=False
        )  # 56 -> 14
        self.layer4 = ConvBNReLU(
            num_filters_init * 2, num_filters_init * 2, 3, 1, 1, bias=False
        )  # 14 -> 7

        self.layer5 = ConvBNReLU(
            num_filters_init * 2, num_filters_init * 4, 12, 1, 1, bias=False
        )  # 7 -> 3
        self.layer6 = ConvBNReLU(
            num_filters_init * 4, num_filters_init * 4, 12, 1, 1, bias=False
        )  # 6 -> 1

        self.layer7 = ConvBNReLU(
            num_filters_init * 4, num_filters_init * 8, 36, 1, 1, bias=False
        )  # 6 -> 1
        self.layer8 = ConvBNReLU(
            num_filters_init * 8, num_filters_init * 8, 36, 1, 1, bias=False
        )  # 7 -> 3

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())

        out = self.layer2(out)
        # print(out.size())
        out = self.max_pool(out)

        out = self.layer3(out)
        # print(out.size())

        out = self.layer4(out)
        out = self.max_pool(out)

        out = self.layer5(out)
        # print(out.size())
        out = self.layer6(out)
        out = self.max_pool(out)
        out = self.layer7(out)

        out = self.layer8(out)
        out = self.max_pool(out)

        out = torch.flatten(out, 1)
        return out


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes=2,
        in_cnn_channels=3,
        num_cnn_filters_init=32,
        lstm_layer=3,
        lstm_nn_size=1024,
        model_device="cpu",
        dropout_p=0,
        bidrectional=False,
        batch_size=10,
    ):
        super(CNNLSTM, self).__init__()
        self.feature_extractor = CNN(
            in_channels=in_cnn_channels, num_filters_init=num_cnn_filters_init
        )
        self.lstm = nn.LSTM(
            input_size=4608,
            hidden_size=lstm_nn_size,
            num_layers=lstm_layer,
            bidirectional=bidrectional,
        )
        if bidrectional is True:
            fc_feature_size = lstm_nn_size * 2
        else:
            fc_feature_size = lstm_nn_size
        self.fc1 = nn.Linear(fc_feature_size, fc_feature_size)
        self.dropout_layer = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(fc_feature_size, num_classes)
        self.fc_feature_size = fc_feature_size
        self.model_device = model_device

        self.lstm_layer = lstm_layer
        self.batch_size = batch_size
        self.lstm_nn_size = lstm_nn_size
        self.bidrectional = bidrectional

    def init_hidden(self, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        init_lstm_layer = self.lstm_layer
        if self.bidrectional:
            init_lstm_layer = self.lstm_layer * 2
        hidden_a = torch.randn(
            init_lstm_layer,
            batch_size,
            self.lstm_nn_size,
            device=self.model_device,
        )
        hidden_b = torch.randn(
            init_lstm_layer,
            batch_size,
            self.lstm_nn_size,
            device=self.model_device,
        )

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, x, seq_lengths):
        # x dim: batch_size x C x F_1
        # we will need to do the packing of the sequence
        # dynamically for each batch of input
        # 1. feature extractor

        x = self.feature_extractor(x)  # x dim: total_epoch_num * feature size
        feature_size = x.size()[-1]

        # 2. lstm
        seq_tensor = torch.zeros(
            len(seq_lengths),
            seq_lengths.max(),
            feature_size,
            dtype=torch.float,
            device=self.model_device,
        )
        start_idx = 0
        for i in range(len(seq_lengths)):
            current_len = seq_lengths[i]
            current_series = x[
                start_idx : start_idx + current_len, :
            ]  # num_time_step x feature_size
            current_series = current_series.view(
                1, current_series.size()[0], -1
            )
            seq_tensor[i, :current_len, :] = current_series
            start_idx += current_len

        seq_lengths_ordered, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        packed_input = pack_padded_sequence(
            seq_tensor, seq_lengths_ordered.cpu().numpy(), batch_first=True
        )

        # x dim for lstm: #  batch_size_rnn x Sequence_length x F_2
        # uncomment for random init state
        # hidden = self.init_hidden(len(seq_lengths))
        packed_output, _ = self.lstm(packed_input)
        output, input_sizes = pad_packed_sequence(
            packed_output, batch_first=True
        )

        # reverse back to the original order
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = output[unperm_idx]

        # reverse back to the originaly shape
        # total_epoch_num * fc_feature_size
        fc_tensor = torch.zeros(
            seq_lengths.sum(),
            self.fc_feature_size,
            dtype=torch.float,
            device=self.model_device,
        )

        start_idx = 0
        for i in range(len(seq_lengths)):
            current_len = seq_lengths[i]
            current_series = lstm_output[
                i, :current_len, :
            ]  # num_time_step x feature_size
            current_series = current_series.view(current_len, -1)
            fc_tensor[start_idx : start_idx + current_len, :] = current_series
            start_idx += current_len
        # print("lstm time: ", end-start)

        # 3. linear readout
        x = self.fc1(fc_tensor)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [
        64,
        3,
        64,
        3,
        "M",
        128,
        3,
        128,
        3,
        128,
        3,
        "M",
        256,
        3,
        256,
        3,
        256,
        3,
        256,
        3,
        "M",
        512,
        3,
        512,
        3,
        512,
        3,
        512,
        3,
        "M",
        512,
        3,
        512,
        3,
        512,
        3,
        512,
        3,
        "M",
        1024,
        30,
    ],  # converted one FC to ConV
    "B": [
        64,
        12,
        64,
        12,
        "M",
        128,
        24,
        128,
        24,
        "M",
        256,
        24,
        256,
        24,
        "M",
        512,
        24,
        512,
        24,
        "M",
        512,
        24,
        512,
        48,
        "M",
    ],
    "C": [
        64,
        12,
        64,
        12,
        "M",
        128,
        24,
        128,
        24,
        "M",
        256,
        24,
        256,
        24,
        "M",
        512,
        48,
        512,
        48,
        "M",
        512,
        48,
        512,
        92,
        "M",
    ],
    "D": [32, 12, 64, 12, "M", 128, 24, 128, 48, "M", 256, 48, 256, 96, "M"],
}


def _cnn(
    cfg: str, batch_norm: bool, pretrained: bool, **kwargs: Any
) -> AccNet:
    if pretrained:
        kwargs["init_weights"] = False
    model = AccNet(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def cnn1(pretrained: bool = False, **kwargs: Any) -> AccNet:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale
    Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    classifier_input_size = 28  # this shouldn't change
    return _cnn(
        "A",
        True,
        pretrained,
        classifier_input_size=classifier_input_size,
        **kwargs,
    )


def cnn3(pretrained: bool = False, **kwargs: Any) -> AccNet:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale
    Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    classifier_input_size = 32  # this shouldn't change
    return _cnn(
        "B",
        False,
        pretrained,
        classifier_input_size=classifier_input_size,
        **kwargs,
    )


def cnn5(pretrained: bool = False, **kwargs: Any) -> AccNet:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale
     Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    classifier_input_size = 42  # this shouldn't change
    return _cnn(
        "C",
        False,
        pretrained,
        classifier_input_size=classifier_input_size,
        **kwargs,
    )


def cnnSmall(pretrained: bool = False, **kwargs: Any) -> AccNet:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale
    Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    classifier_input_size = 12  # this shouldn't change
    classifier_layer_size = 1000
    return _cnn(
        "D",
        False,
        pretrained,
        classifier_input_size=classifier_input_size,
        classifier_layer_size=classifier_layer_size,
        **kwargs,
    )


class Downsample(nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such "
            "that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1))
        )

    def forward(self, x):
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )


class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, padding=2
    ):

        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x


class Resnet(nn.Module):
    def __init__(
        self,
        output_size=1,
        n_channels=3,
        is_eva=False,
        resnet_version=1,
        epoch_len=10,
        is_mtl=False,
        is_simclr=False,
        is_mc_dropout=False
    ):
        super(Resnet, self).__init__()

        if resnet_version == 1:
            if epoch_len == 5:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 3, 1),
                    (512, 5, 0, 5, 3, 1),
                ]

            elif epoch_len == 10:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 3, 1),
                ]

            elif epoch_len == 30:
                cgf = [
                    (64, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 4, 0),
                ]

            else:
                raise ValueError(
                    f"Unsupported epoch_len={epoch_len}. "
                    "Use 5, 10, or 30."
                )

        # elif resnet_version == 2:
        #     cgf = [
        #         (64, 5, 2, 5, 3, 1),
        #         (64, 5, 2, 5, 3, 1),
        #         (128, 5, 2, 5, 5, 1),
        #         (128, 5, 2, 5, 5, 1),
        #         (256, 5, 2, 5, 4, 0),
        #     ]

        elif resnet_version == 2:
    # Compact ResNet. Match downsampling to the input window length.
            if epoch_len == 5:
                # 150 -> 75 -> 37 -> 12 -> 4 -> 1
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 3, 1),
                ]
        
            elif epoch_len == 10:
                # 300 -> 150 -> 75 -> 15 -> 3 -> 1
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 5, 1),
                    (128, 5, 2, 5, 5, 1),
                    (256, 5, 2, 5, 3, 1),
                ]
        
            elif epoch_len == 30:
                # 900 -> 300 -> 100 -> 20 -> 4 -> 1
                cgf = [
                    (64, 5, 2, 5, 3, 1),
                    (64, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 5, 1),
                    (128, 5, 2, 5, 5, 1),
                    (256, 5, 2, 5, 4, 0),
                ]
        
            else:
                raise ValueError(
                    f"Unsupported epoch_len={epoch_len}. Use 5, 10, or 30."
                )
        
        elif resnet_version == 3:
            
            if epoch_len == 10:
                cgf = [
                    # out, kernel, resblocks, res-kernel, downfactor, downorder
                    (32,  5, 1, 5, 2, 2),  # 300 -> 150
                    (64,  5, 1, 5, 2, 2),  # 150 -> 75
                    (96,  5, 1, 5, 5, 1),  # 75  -> 15
                    (128, 5, 1, 5, 5, 1),  # 15  -> 3
                    (192, 5, 1, 5, 3, 1),  # 3   -> 1
                ]
            else:
                raise ValueError(
                    "Compact ResNet version 3 is currently configured "
                    "for epoch_len=10."
                )
        else:
            raise ValueError(
                f"Unsupported resnet_version={resnet_version}. "
                "Use 1, 2, or 3."
            )

        in_channels = n_channels
        feature_extractor = nn.Sequential()

        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params

            feature_extractor.add_module(
                f"layer{i + 1}",
                Resnet.make_layer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    conv_kernel_size=conv_kernel_size,
                    n_resblocks=n_resblocks,
                    resblock_kernel_size=resblock_kernel_size,
                    downfactor=downfactor,
                    downorder=downorder,
                ),
            )

            in_channels = out_channels

        self.feature_extractor = feature_extractor
        self.feature_size = out_channels
        self.is_mtl = is_mtl

        if is_eva:
            if is_mc_dropout:
                self.classifier = DropoutHead(
                input_size=out_channels,
                output_size=output_size,
            )
            else:
                self.classifier = EvaClassifier(
                    input_size=out_channels,
                    output_size=output_size,
                )

        elif is_mtl:
            self.aot_h = Classifier(
                input_size=out_channels,
                output_size=output_size,
            )
            self.scale_h = Classifier(
                input_size=out_channels,
                output_size=output_size,
            )
            self.permute_h = Classifier(
                input_size=out_channels,
                output_size=output_size,
            )
            self.time_w_h = Classifier(
                input_size=out_channels,
                output_size=output_size,
            )

        elif is_simclr:
            self.classifier = ProjectionHead(
                input_size=out_channels,
                encoding_size=output_size,
            )

        else:
            self.classifier = Classifier(
                input_size=out_channels,
                output_size=output_size,
            )

        weight_init(self)

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert (
            conv_kernel_size % 2
        ), "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x):
        feats = self.feature_extractor(x)

        if self.is_mtl:
            aot_y = self.aot_h(feats.view(x.shape[0], -1))
            scale_y = self.scale_h(feats.view(x.shape[0], -1))
            permute_y = self.permute_h(feats.view(x.shape[0], -1))
            time_w_h = self.time_w_h(feats.view(x.shape[0], -1))
            return aot_y, scale_y, permute_y, time_w_h
        else:
            y = self.classifier(feats.view(x.shape[0], -1))
            return y
        return y

def weight_init(self, mode="fan_out", nonlinearity="relu"):

    for m in self.modules():

        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, mode=mode, nonlinearity=nonlinearity
            )

        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(128, 128, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(128, 256, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(256, 256, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(256, 512, 3, stride=3, padding=1),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 256, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 128, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 128, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, 7, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 64, 7, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(
                64, 3, 5, stride=3, padding=3, output_padding=1
            ),
            nn.ReLU(True),
        )

        weight_init(self)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class EncoderMLP(nn.Module):
    def __init__(self, output_size):
        super(EncoderMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(128, 128, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(128, 256, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(256, 256, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(256, 512, 3, stride=3, padding=1),
            nn.ReLU(True),
        )

        self.classifier = EvaClassifier(
            input_size=512, output_size=output_size
        )

        weight_init(self)

    def forward(self, x):
        feats = self.encoder(x)
        y = self.classifier(feats.view(x.shape[0], -1))
        return y

class VanillaEncoder(nn.Module):
    def __init__(
        self,
        output_size = 1,
        in_channels = 3,
    ):

        self.encoder = nn.Sequential(
            # 1: (B, 3, 300) -> (B, 64, 149)
            nn.Conv1d(
                in_channels,
                64,
                kernel_size=5,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # 2: (B, 64, 149) -> (B, 128, 74)
            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # 3: (B, 128, 74) -> (B, 256, 36)
            nn.Conv1d(
                128,
                256,
                kernel_size=5,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # 4: (B, 256, 36) -> (B, 256, 12)
            nn.Conv1d(
                256,
                256,
                kernel_size=3,
                stride=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # (B, 256, 12) -> (B, 256, 1)
            nn.AdaptiveAvgPool1d(1),

            # (B, 256, 1) -> (B, 256)
            nn.Flatten(start_dim=1),
        )

        self.regressor = nn.Linear(
            in_features=256,
            out_features=output_size,
        )

        weight_init(self)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 256-dimensional feature vector."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        predicted_hr = self.regressor(features)
        return predicted_hr
class CompactResidualTCNBlock(nn.Module):
    """Residual temporal block used by :class:`CompactResidualTCN`.

    The first convolution can downsample the temporal dimension. Dilated
    convolutions enlarge the receptive field without substantially increasing
    the number of parameters. Group normalization is used instead of batch
    normalization because it is more stable when training with small batches.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        dilation = 1,
        dropout_p = 0.1,
        num_groups= 8,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve alignment.")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel counts must be positive.")
        if stride <= 0 or dilation <= 0:
            raise ValueError("stride and dilation must be positive.")

        padding = dilation * (kernel_size - 1) // 2
        groups = min(num_groups, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout(out)

        out = out + residual
        return self.activation(out)


class TemporalMeanStdPool(nn.Module):
    """Concatenate temporal mean and standard deviation.

    For an input of shape ``(B, C, T)``, the output has shape ``(B, 2*C)``.
    Compared with mean pooling alone, this preserves information about temporal
    variability that may be useful for downstream activity-recognition tasks.
    """

    def __init__(self, eps = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1)
        variance = x.var(dim=-1, unbiased=False)
        std = torch.sqrt(variance + self.eps)
        return torch.cat((mean, std), dim=1)


class CompactResidualTCN(nn.Module):
    """Compact residual temporal CNN for heart-rate regression.

    Default configuration
    ---------------------
    Input:
        ``(batch, 3, 300)`` for a 10-second, 30-Hz accelerometer window.
    Encoder:
        Residual dilated temporal blocks with channels 32 -> 64 -> 128.
    Pooling:
        Concatenated temporal mean and standard deviation.
    Embedding:
        128 dimensions by default.
    Output:
        One HR regression value by default.

    With ``in_channels=3``, ``embedding_size=128`` and ``output_size=1``, the
    model has 447,521 trainable parameters (approximately 0.448 million).
    """

    def __init__(
        self,
        output_size = 1,
        in_channels = 3,
        embedding_size = 128,
        dropout_p = 0.1,
    ):
        super().__init__()

        if output_size <= 0:
            raise ValueError("output_size must be positive.")
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive.")

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                32,
                kernel_size=9,
                stride=2,
                padding=4,
                bias=False,
            ),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )

        self.encoder = nn.Sequential(
            CompactResidualTCNBlock(
                32,
                32,
                kernel_size=7,
                stride=1,
                dilation=1,
                dropout_p=dropout_p,
            ),
            CompactResidualTCNBlock(
                32,
                64,
                kernel_size=7,
                stride=2,
                dilation=1,
                dropout_p=dropout_p,
            ),
            CompactResidualTCNBlock(
                64,
                64,
                kernel_size=7,
                stride=1,
                dilation=2,
                dropout_p=dropout_p,
            ),
            CompactResidualTCNBlock(
                64,
                128,
                kernel_size=5,
                stride=2,
                dilation=2,
                dropout_p=dropout_p,
            ),
            CompactResidualTCNBlock(
                128,
                128,
                kernel_size=5,
                stride=1,
                dilation=4,
                dropout_p=dropout_p,
            ),
            CompactResidualTCNBlock(
                128,
                252,
                kernel_size=5,
                stride=1,
                dilation=4,
                dropout_p=dropout_p,
            ),
            CompactResidualTCNBlock(
                252,
                252,
                kernel_size=5,
                stride=1,
                dilation=4,
                dropout_p=dropout_p,
            ),
        )

        self.pool = TemporalMeanStdPool()
        self.projection = nn.Sequential(
            nn.Linear(252 * 2, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )
        self.regressor = nn.Linear(embedding_size, output_size)

        weight_init(self)

    def forward_features(self, x):
        """Return the compact embedding used for downstream tasks."""
        if x.ndim != 3:
            raise ValueError(
                "Expected input with shape (batch, channels, time), "
                f"but received {tuple(x.shape)}."
            )

        x = self.stem(x)
        x = self.encoder(x)
        x = self.pool(x)
        return self.projection(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.regressor(features)
