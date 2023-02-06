import abc

from ..data.data_generator import DataGenFeat, DataGenPure, DataGenSequence


class BaseTrainer(abc.ABC):
    def __init__(
        self,
        model,
        task,
        loss_type,
        n_epochs,
        lr,
        lr_decay,
        epsilon,
        batch_size,
        num_neg,
    ):
        self.model = model
        self.task = task
        self.loss_type = loss_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.num_neg = num_neg

    def get_data_generator(self, train_data):
        if self.model.model_category == "pure":
            data_generator = DataGenPure(train_data)
        elif self.model.model_category == "feat":
            data_generator = DataGenFeat(
                train_data, self.model.sparse, self.model.dense
            )
        else:
            data_generator = DataGenSequence(
                train_data,
                self.model.data_info,
                self.model.sparse if hasattr(self.model, "sparse") else False,
                self.model.dense if hasattr(self.model, "dense") else False,
                self.model.seq_mode,
                self.model.max_seq_len,
                self.model.n_items,
            )
        return data_generator

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError


def get_trainer(model):
    from ..utils.constants import TF_TRAIN_MODELS
    from .tf_trainer import (
        BPRTrainer,
        RNN4RecTrainer,
        TensorFlowTrainer,
        WideDeepTrainer,
        YoutubeRetrievalTrainer,
    )
    from .torch_trainer import SageDGLTrainer, SageTrainer, TorchTrainer

    train_params = {
        "model": model,
        "task": model.task,
        "loss_type": model.loss_type,
        "n_epochs": model.n_epochs,
        "lr": model.lr,
        "lr_decay": model.lr_decay,
        "epsilon": model.epsilon,
        "batch_size": model.batch_size,
        "num_neg": model.__dict__.get("num_neg"),
    }

    if model.model_name in TF_TRAIN_MODELS:
        if model.model_name == "YouTubeRetrieval":
            train_params.update(
                {
                    "num_sampled_per_batch": model.num_sampled_per_batch,
                    "sampler": model.sampler,
                }
            )
            tf_trainer_cls = YoutubeRetrievalTrainer
        elif model.model_name == "BPR":
            tf_trainer_cls = BPRTrainer
        elif model.model_name == "RNN4Rec":
            tf_trainer_cls = RNN4RecTrainer
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
                "sampler": model.sampler,
                "device": model.device,
            }
        )
        if "Sage" in model.model_name:
            train_params.update(
                {
                    "paradigm": model.paradigm,
                    "num_walks": model.num_walks,
                    "walk_len": model.sample_walk_len,
                    "start_node": model.start_node,
                    "focus_start": model.focus_start,
                }
            )
            torch_trainer_cls = (
                SageDGLTrainer if "DGL" in model.model_name else SageTrainer
            )
        else:
            torch_trainer_cls = TorchTrainer
        return torch_trainer_cls(**train_params)
