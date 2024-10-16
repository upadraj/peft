# +
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.utils.quantization_config import BitsAndBytesConfig

from peft import TTLoraConfig

# %%
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

model_name = "google/flan-t5-base"
model = AutoModelForQuestionAnswering.from_pretrained(model_name, device_map="auto")
# %%


lora_config = TTLoraConfig(
    r=5,
    tt_shape=[64, 16, 9, 64],
    lora_alpha=2,
    lora_dropout=0.05,
)

print(lora_config)
