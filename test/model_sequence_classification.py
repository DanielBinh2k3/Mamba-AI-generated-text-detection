import inspect
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache



# @add_start_docstrings(
#     """
#     The Zamba2 Model with a sequence classification head on top (linear layer).

#     [`Zamba2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
#     (e.g. GPT-2) do.

#     Since it does classification on the last token, it requires to know the position of the last token. If a
#     `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
#     no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
#     padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
#     each row of the batch).
#     """,
# )
class CustomModelForSequenceClassification(PreTrainedModel):
    def __init__(self, config, model):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.backbone = model

        # Reuse the model's lm_head for classification
        # No need to define self.score, reuse lm_head
        self.backbone.lm_head = nn.Linear(model.lm_head.in_features, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.embed_tokens

    def set_input_embeddings(self, value):
        self.backbone.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        return_dict: bool = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through the model
        transformer_outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            # past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print(transformer_outputs)
        # Use lm_head directly as the classification head
        logits = transformer_outputs.logits

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        # Determine sequence lengths if pad_token_id is defined
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            # past_key_values=transformer_outputs.past_key_values,
            # hidden_states=transformer_outputs.hidden_states,
            # attentions=transformer_outputs.attentions,
        ) 