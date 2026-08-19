import argparse
import logging
import os

import torch
import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from twitter.clip.config import ClipConfig
from twitter.clip.datasets.language_image_dataset import LanguageImageGcsDataset
from twitter.clip.model import TwitterCLIP
from twitter.clip.utils import train_utils

import clip


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIP training on Twitter data.",
        argument_default=argparse.SUPPRESS,
    )
    add_dataset_args(parser.add_argument_group("dataset"))
    add_model_args(parser.add_argument_group("model"))
    add_training_args(parser.add_argument_group("training"))
    return parser.parse_args()


def add_dataset_args(parser):
    parser.add_argument(
        "--dataset_path",
        help="path to GCS dataset with (text, image) pair TFRecords; "
        "this argument is required",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_resolution",
        help="input image resolution",
        type=int,
    )
    parser.add_argument(
        "--use_data_augmentation",
        help="pass this flag to add a random crop to images during training",
        action="store_true",
    )


def add_model_args(parser):
    parser.add_argument(
        "--clip_model_type",
        help="CLIP model type; if working with no internet "
        "connection, model must be cached in ~/.cache/clip/",
        type=str,
        choices=clip.available_models(),
    )
    parser.add_argument(
        "--top_feedforward",
        help="pass this flag to add a feed-forward layer to change dimensionality "
        "of CLIP output embeddings",
        action="store_true",
    )
    parser.add_argument(
        "--final_embedding_dim",
        help="dimension to map CLIP embeddings to, if --top_feedforward",
        type=int,
    )
    parser.add_argument(
        "--logit_scale",
        help="initial value of learnable temperature parameter",
        type=float,
    )


def add_training_args(parser):
    parser.add_argument(
        "--model_dir",
        help="save directory location",
        type=str,
    )
    parser.add_argument(
        "--save_interval",
        help="save checkpoints every this many steps",
        type=int,
    )
    parser.add_argument(
        "--save_last_ckpt_only",
        help="pass this flag to only save last checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--batch_size",
        help="number of samples per batch",
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        help="learning rate for optimizer",
        type=float,
    )
    parser.add_argument(
        "--num_training_steps",
        help="total number of training steps",
        type=int,
    )
    parser.add_argument(
        "--num_warmup_steps",
        help="number of steps for warmup phase of optimization",
        type=int,
    )
    parser.add_argument(
        "--num_decay_steps",
        help="total number of learning rate decay steps",
        type=int,
    )
    parser.add_argument(
        "--mixed_precision",
        help="pass this flag to train using mixed precision",
        action="store_true",
    )


def train(config):
    train_dataset = (
        LanguageImageGcsDataset(
            input_resolution=config.input_resolution,
            use_data_augmentation=config.use_data_augmentation,
            batch_size=config.batch_size,
        )
        .create_dataset(
            dataset_path=config.dataset_path,
            is_training=True,
        )
        .take(config.num_training_steps + 1)
    )

    model = TwitterCLIP(
        clip_model_type=config.clip_model_type,
        top_feedforward=config.top_feedforward,
        final_embedding_dim=config.final_embedding_dim,
        logit_scale=config.logit_scale,
    )

    image_loss_op, text_loss_op = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
    optimizer, scheduler = train_utils.create_optimizer(
        model,
        learning_rate=config.learning_rate,
        num_training_steps=config.num_training_steps,
        num_warmup_steps=config.num_warmup_steps,
        num_decay_steps=config.num_decay_steps,
    )
    writer = SummaryWriter(config.model_dir)
    if config.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()

    pbar = tqdm.tqdm(
        enumerate(train_dataset), unit="samples", unit_scale=config.batch_size
    )
    running_loss = 0.0
    for step, features in pbar:
        optimizer.zero_grad()

        def _compute_loss():
            logits = model.get_logits(features)
            targets = torch.arange(
                logits["image_logits"].size()[0], dtype=torch.long
            ).to("cuda:0" if model.use_gpu else "cpu")
            loss = (
                image_loss_op(logits["image_logits"], targets)
                + text_loss_op(logits["text_logits"], targets)
            ) / 2
            return loss

        if config.mixed_precision:
            with torch.cuda.amp.autocast():
                loss = _compute_loss()
        else:
            loss = _compute_loss()
        with torch.no_grad():
            running_loss = train_utils.get_avg(running_loss, loss.item(), step)

        if not model._multi_gpu or torch.cuda.current_device() == 0:
            scalar_summaries = {
                "batch_loss": loss.item(),
                "running_loss": running_loss,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "logit_scale": model.clip_model.logit_scale.item(),
            }
            if config.mixed_precision:
                scalar_summaries["loss_scale"] = scaler.get_scale()
            for name, val in scalar_summaries.items():
                writer.add_scalar(name, val, step)
            if step == 0:
                image_sample = torch.Tensor(features["image"][:16].numpy())
                writer.add_images("preprocessed_images", image_sample, step)

            if step % config.save_interval == 0 or step == config.num_training_steps:
                ckpt_path = (
                    "ckpt-last.pt"
                    if config.save_last_ckpt_only
                    else f"ckpt-step-{step:06d}.pt"
                )
                save_dict = {
                    "config": vars(config),
                    "step": step,
                    "image_encoder_state_dict": model.image_encoder.state_dict(),
                    "text_encoder_state_dict": model.text_encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "logit_scale": model.clip_model.logit_scale.item(),
                    "running_loss": running_loss,
                }
                torch.save(save_dict, os.path.join(config.model_dir, ckpt_path))
                writer.flush()

        if config.mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        pbar.set_description(
            f"loss: {running_loss}, step {step}/{config.num_training_steps}"
        )
        pbar.refresh()

    writer.close()


if __name__ == "__main__":
    config = ClipConfig(**vars(parse_args()))

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.info(" CLIP Training Configuration:")
    for k, v in config.__dict__.items():
        logger.info(f"\t{k}: {v}")

    train(config)
