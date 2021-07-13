import torch.nn as nn
import torch.optim as optim


def get_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    momentum: float = 0.9,
    dampening: float = 0.0,
    weight_decay: float = 0.0001,
    nesterov: bool = True,
) -> optim.Optimizer:

    assert optimizer_name in ["SGD", "Adam"]
    print(f"{optimizer_name} will be used as an optimizer.")

    if optimizer_name == "Adam":
        # base_params = []
        # params = []
        # for name,param in model.named_parameters():
        #     if(name.find("attn.layers")!=-1):
        #         params.append(param)
        #     else:
        #         base_params.append(param)
        #     # if(name.find("multihead_selfattn")!=-1): # or name.find("mlp")!=-1):
        #     # if(name.find("transformer_encoder")!=-1):
        #     #     params.append(param)
        #     # else:
        #     #     base_params.append(param)
        # optimizer = optim.Adam([{'params': base_params}, {'params': params, 'lr': 0.000000001}], lr=learning_rate)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    return optimizer
