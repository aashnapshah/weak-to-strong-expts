from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, AutoTokenizer


@dataclass
class HeadOutput:
    logits: torch.FloatTensor


class TransformerWithHead(PreTrainedModel):
    """
    This class initializes the linear head to zeros
    """

    def __init__(self, name, checkpoint=None, linear_probe=False, **kwargs):
        config = AutoConfig.from_pretrained(name, **kwargs)
        super().__init__(config)
        self.num_labels = config.num_labels
        self.checkpoint = checkpoint
        print(f"model: {name}-{checkpoint}")

        if checkpoint==None:
            lm = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        else:
            lm = AutoModelForCausalLM.from_pretrained(name, revision=checkpoint, **kwargs)

        self.lm = lm

        if (name.startswith("EleutherAI")): # for pythia models
            self.transformer = lm.gpt_neox
            self.lm_head = lm.embed_out

        else:
            self.transformer = lm.transformer
            self.lm_head = lm.lm_head

        hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", None))
        self.score = torch.nn.Linear(hidden_size, self.num_labels, bias=False).to(
            self.lm_head.weight.dtype
        )
        torch.nn.init.normal_(self.score.weight, std=0.0)
        self.linear_probe = linear_probe

    @classmethod
    def from_pretrained(cls, name, checkpoint, **kwargs):
        return cls(name, checkpoint,  **kwargs)

    def gradient_checkpointing_enable(self):
        model = self.transformer
        (
            model if hasattr(model, "save_pretrained") else model.module
        ).gradient_checkpointing_enable()

    def forward(self, input_ids: torch.LongTensor):
        """
        Forward pass of the model with a linear head.

        Parameters:
        input_ids (torch.LongTensor): Input tensor containing the token ids.

        Returns:
        HeadOutput: Output dataclass containing the logits.
        """
        input_lens = (input_ids != 0).sum(dim=-1)
        transformer_outputs = self.transformer(input_ids)
        hidden_states = torch.stack(
            [transformer_outputs[0][i, input_lens[i] - 1, :] for i in range(len(input_lens))]
        )
        self.score.to(hidden_states.device)
        if self.linear_probe:
            hidden_states = hidden_states.detach()
        logits = self.score(hidden_states)
        return logits