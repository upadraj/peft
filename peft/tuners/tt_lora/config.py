# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from torch import nn

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
# creates __init__  with the fields.
class TTLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`TTLoraModel`].

    Args:
        r (`int`):
            Lora attention dimension (the "rank").
        tt_shape ('List[int]')
            Shape of the tensor core matrix.
        lora_alpha (`int`):
            The alpha parameter for Lora scaling.
        lora_dropout (`float`):
            The dropout probability for Lora layers.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    tt_shape: list[int] = field(
        default_factory=lambda: [
            64,
            16,
            9,
            64,
        ],  # Use default_factory for mutable defaults
        metadata={
            "help": "shape of inner core tensor. The product of all elements of tt_shape should be equal to modifying layer weight elements."
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})

    def to_dict(self):
        """
        Returns the configuration for your adapter model as a dictionary. Removes runtime configurations.
        """
        rv = super().to_dict()
        rv.pop("runtime_config")
        return rv
