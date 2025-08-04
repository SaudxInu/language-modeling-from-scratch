import torch
from transformers import PreTrainedModel


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    response_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        token_entropy = -torch.sum(log_probs * torch.exp(log_probs), dim=-1)
        return {"log_probs": response_log_probs, "token_entropy": token_entropy}
    return {"log_probs": response_log_probs}
