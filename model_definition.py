# model_definition.py
import torch
import torch.nn as nn
from transformers import (
    SiglipImageProcessor,
    SiglipVisionConfig,
    SiglipVisionModel,
    logging,
)
from transformers.image_processing_utils import BatchFeature
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from os import PathLike

# --- IMPORTACIONES AÑADIDAS PARA COMPATIBILIDAD CON PYTHON 3.9 ---
from typing import Optional, Union
# -------------------------------------------------------------

logging.set_verbosity_error()

class AestheticPredictorV2_5Head(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.scoring_head = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.scoring_head(image_embeds)


class AestheticPredictorV2_5Model(SiglipVisionModel):
    PATCH_SIZE = 14

    def __init__(self, config: SiglipVisionConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.layers = AestheticPredictorV2_5Head(config)
        self.post_init()

    def forward(
        self,
        # --- SINTAXIS MODIFICADA ---
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        # --- FIN DE MODIFICACIÓN ---
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = super().forward(
            pixel_values=pixel_values,
            return_dict=return_dict,
        )
        image_embeds = outputs.pooler_output
        image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        prediction = self.layers(image_embeds_norm)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(prediction.squeeze(), labels.squeeze())

        if not return_dict:
            return (loss, prediction, image_embeds)

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=prediction,
            hidden_states=image_embeds,
        )


class AestheticPredictorV2_5Processor(SiglipImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> BatchFeature:
        return super().__call__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        self,
        # --- SINTAXIS MODIFICADA ---
        pretrained_model_name_or_path: Union[str, PathLike] = "google/siglip-so400m-patch14-384",
        # --- FIN DE MODIFICACIÓN ---
        *args,
        **kwargs,
    ) -> "AestheticPredictorV2_5Processor":
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)