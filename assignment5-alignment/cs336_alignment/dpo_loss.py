import torch

from cs336_alignment.get_response_log_probs import get_response_log_probs


def dpo_loss(model, model_ref, tokenizer, beta, prompt, chosen, rejected):
    with open("cs336_alignment/prompts/alpaca_sft.prompt", "r") as f:
        prompt_template = f.read().strip()

    prompt_chosen = prompt_template.format(instruction=prompt, response=chosen)
    input_ids_chosen = tokenizer.encode(prompt_chosen)
    labels_chosen = input_ids_chosen[1:] + [tokenizer.eos_token_id]
    input_ids_chosen, labels_chosen = torch.tensor(input_ids_chosen), torch.tensor(
        labels_chosen
    )

    prompt_rejected = prompt_template.format(instruction=prompt, response=rejected)
    input_ids_rejected = tokenizer.encode(prompt_rejected)
    labels_rejected = input_ids_rejected[1:] + [tokenizer.eos_token_id]
    input_ids_rejected, labels_rejected = torch.tensor(
        input_ids_rejected
    ), torch.tensor(labels_rejected)

    log_probs_model_chosen = get_response_log_probs(
        model, input_ids_chosen, labels_chosen
    )["log_probs"]
    log_probs_model_ref_chosen = get_response_log_probs(
        model_ref, input_ids_chosen, labels_chosen
    )["log_probs"]

    chosen_ratio = torch.sum(
        log_probs_model_chosen - log_probs_model_ref_chosen.to(model.device), dim=-1
    )

    log_probs_model_rejected = get_response_log_probs(
        model, input_ids_rejected, labels_rejected
    )["log_probs"]
    log_probs_model_ref_rejected = get_response_log_probs(
        model_ref, input_ids_rejected, labels_rejected
    )["log_probs"]

    rejected_ratio = torch.sum(
        log_probs_model_rejected - log_probs_model_ref_rejected.to(model.device), dim=-1
    )

    return -torch.log(torch.sigmoid(beta * (chosen_ratio - rejected_ratio))).mean()
