"""
Adapted from  https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

# Usage
python finetune.py
"""

import warnings

warnings.filterwarnings("ignore")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# GPU를 사용을 위한 코드
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


import argparse

# keras_cv 모델을 임포트
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from tensorflow.keras import mixed_precision

from datasets import DatasetUtils
from trainer import Trainer

MAX_PROMPT_LENGTH = 77
CKPT_PREFIX = "ckpt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to fine-tune a Stable Diffusion model."
    )
    # 데이터셋 관련 parser argument
    parser.add_argument("--dataset_archive", default=None, type=str)
    parser.add_argument("--img_height", default=256, type=int)
    parser.add_argument("--img_width", default=256, type=int)
    # 하이퍼파라미터 parser argument
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--wd", default=1e-2, type=float)
    parser.add_argument("--beta_1", default=0.9, type=float)
    parser.add_argument("--beta_2", default=0.999, type=float)
    parser.add_argument("--epsilon", default=1e-08, type=float)
    parser.add_argument("--ema", default=0.9999, type=float)    # Exponential moving average 설정
    parser.add_argument("--max_grad_norm", default=1.0, type=float) # 최대 기울기 설정
    # Batch size, Epochs
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    # Others.
    parser.add_argument(
        # 혼합 정밀도 설정, 학습 속도를 높이고 메모리 사용량을 조절할 수 있음
        "--mp", action="store_true", help="Whether to use mixed-precision." 
    )
    # 이전에 학습시킨 모델에 이어서 계속 학습하는 parser argument
    parser.add_argument(
        "--pretrained_ckpt",
        default=None,
        type=str,
        help="Provide a local path to a diffusion model checkpoint in the `h5`"
        " format if you want to start over fine-tuning from this checkpoint.",
    )

    return parser.parse_args()


def run(args):
    # 혼합 정밀도 설정 관련
    if args.mp:
        print("Enabling mixed-precision...")
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        assert policy.compute_dtype == "float16"
        assert policy.variable_dtype == "float32"

    # 데이터셋 준비
    print("Initializing dataset...")
    data_utils = DatasetUtils(
        dataset_archive=args.dataset_archive,
        batch_size=args.batch_size,
        img_height=args.img_height,
        img_width=args.img_width,
    )
    training_dataset = data_utils.prepare_dataset()

    print("Initializing trainer...")
    
    # 모델 결과 파일명 설정
    ckpt_path = (
        CKPT_PREFIX
        + f"_epochs_{args.num_epochs}"
        + f"_res_{args.img_height}"
        + f"_mp_{args.mp}"
        + ".h5"
    )
    # 인코더 부분
    image_encoder = ImageEncoder(args.img_height, args.img_width)
    # Trainer 객체 호출
    diffusion_ft_trainer = Trainer(
        # keras의 디퓨전 모델을 호출하여 height, width, Max prompt length (77)를 설정
        diffusion_model=DiffusionModel(
            args.img_height, args.img_width, MAX_PROMPT_LENGTH
        ),
        # keras의 Variational Autoencoder 호출
        vae=tf.keras.Model(
            image_encoder.input,
            image_encoder.layers[-2].output,
        ),
        # keras_cv의 NoiseScheduler 호출
        noise_scheduler=NoiseScheduler(),
        pretrained_ckpt=args.pretrained_ckpt,
        mp=args.mp,
        ema=args.ema,
        max_grad_norm=args.max_grad_norm,
    )

    # 옵티마이저 호출 (AdamW)
    print("Initializing optimizer...")
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=args.lr,
        weight_decay=args.wd,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        epsilon=args.epsilon,
    )
    if args.mp:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    # 모델 컴파일
    print("Compiling trainer...")
    diffusion_ft_trainer.compile(optimizer=optimizer, loss="mse")

    # 모델 학습
    print("Training...")
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        save_weights_only=True,
        monitor="loss",
        mode="min",
    )
    diffusion_ft_trainer.fit(
        training_dataset, epochs=args.num_epochs, callbacks=[ckpt_callback]
    )


if __name__ == "__main__":
    args = parse_args()
    run(args)
