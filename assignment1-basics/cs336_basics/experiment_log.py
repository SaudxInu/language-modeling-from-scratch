import wandb


def experiment_log(config):
    wandb.init(
        project=config["project"],
        job_type=config["job_type"],
        name=config["name"],
        config=config,
    )


def log_metric(data: dict):
    wandb.log(data)
