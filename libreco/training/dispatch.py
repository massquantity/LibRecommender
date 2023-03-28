from .tf_trainer import TensorFlowTrainer, WideDeepTrainer, YoutubeRetrievalTrainer
from .torch_trainer import GraphTrainer, TorchTrainer
from ..utils.constants import TF_TRAIN_MODELS


def get_trainer(model):
    from ..bases import SageBase

    train_params = {
        "model": model,
        "task": model.task,
        "loss_type": model.loss_type,
        "n_epochs": model.n_epochs,
        "lr": model.lr,
        "lr_decay": model.lr_decay,
        "epsilon": model.epsilon,
        "batch_size": model.batch_size,
        "sampler": model.sampler,
        "num_neg": model.__dict__.get("num_neg"),
    }

    if model.model_name in TF_TRAIN_MODELS:
        if model.model_name == "YouTubeRetrieval":
            train_params["num_sampled_per_batch"] = model.num_sampled_per_batch
            tf_trainer_cls = YoutubeRetrievalTrainer
        elif model.model_name == "WideDeep":
            tf_trainer_cls = WideDeepTrainer
        else:
            tf_trainer_cls = TensorFlowTrainer
        return tf_trainer_cls(**train_params)
    else:
        train_params.update(
            {
                "amsgrad": model.amsgrad,
                "reg": model.reg,
                "margin": model.margin,
                "device": model.device,
            }
        )
        if isinstance(model, SageBase):
            torch_trainer_cls = GraphTrainer
        else:
            torch_trainer_cls = TorchTrainer
        return torch_trainer_cls(**train_params)
