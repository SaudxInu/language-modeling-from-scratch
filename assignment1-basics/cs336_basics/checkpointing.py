import torch


def save_checkpoint(model, optimizer, iteration, out):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]
