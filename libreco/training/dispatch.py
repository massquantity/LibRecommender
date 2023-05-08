from .tf_trainer import TensorFlowTrainer, WideDeepTrainer, YoutubeRetrievalTrainer
from .torch_trainer import GraphTrainer, TorchTrainer
from ..utils.constants import SageModels, TfTrainModels


def get_trainer(model):
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

    if TfTrainModels.contains(model.model_name):
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
        if SageModels.contains(model.model_name):
            torch_trainer_cls = GraphTrainer
        else:
            torch_trainer_cls = TorchTrainer
        return torch_trainer_cls(**train_params)
