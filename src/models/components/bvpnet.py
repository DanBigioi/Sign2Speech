#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Single-value Binary Value Problem network for implicit audio signal reconstruction.
"""


# TODO: Switch from torchmeta to Learning2Learn!
from torchmeta.modules import MetaModule, MetaSequential
# from torchmeta.modules.utils import get_subdict
from collections import OrderedDict

from src.vendor.siren_modules import FCBlock


class SingleBVPNet(MetaModule):
    """A canonical representation network for a BVP."""

    def __init__(
        self,
        out_features=1,
        type="sine",
        in_features=2,
        hidden_features=256,
        num_hidden_layers=3,
        **kwargs
    ):
        super().__init__()
        self.net = FCBlock(
            in_features=in_features,
            out_features=out_features,
            num_hidden_layers=num_hidden_layers,
            hidden_features=hidden_features,
            outermost_linear=True,
            nonlinearity=type,
        )

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords, self.get_subdict(params, "net"))
        return {"model_in": coords_org, "model_out": output}

    def forward_with_activations(self, model_input):
        """Returns not only model output, but also intermediate activations."""
        coords = model_input["coords"].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {
            "model_in": coords,
            "model_out": activations.popitem(),
            "activations": activations,
        }
