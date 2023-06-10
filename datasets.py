"""
Adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""

import os
from typing import Dict, Tuple

import keras_cv
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder

PADDING_TOKEN = 49407
MAX_PROMPT_LENGTH = 77
AUTO = tf.data.AUTOTUNE
POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
AUTO = tf.data.AUTOTUNE


class DatasetUtils:
    def __init__(
        self,
        dataset_archive: str = None,
        batch_size: int = 4,
        img_height: int = 256,
        img_width: int = 256,
    ):
        self.tokenizer = SimpleTokenizer()
        self.text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
        self.augmenter = keras_cv.layers.Augmenter(
            layers=[
                keras_cv.layers.CenterCrop(img_height, img_width),
                keras_cv.layers.RandomFlip(),
                tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
            ]
        )

        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

        # 데이터셋 경로 설정
        data_path = './data/description'
        image_path = './data/images'

        # dataframe으로 된 데이터셋으로부터 이미지와 텍스트 받아오기
        self.data_frame = pd.read_csv(os.path.join(data_path, "image_caption.csv"))
        self.data_frame["image_path"] = self.data_frame["image_path"].apply(
            lambda x: os.path.join(image_path, x)
        )

    # keras_cv 토크나이저로 텍스트 임베딩
    def process_text(self, caption: str) -> np.ndarray:
        tokens = self.tokenizer.encode(caption)
        tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
        return np.array(tokens)

    # 이미지와 리사이징
    def process_image(
        self, image_path: tf.Tensor, tokenized_text: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, 3)
        image = tf.image.resize(image, (self.img_height, self.img_width))
        return image, tokenized_text

    # 이미지 데이터 증강
    def apply_augmentation(
        self, image_batch: tf.Tensor, token_batch: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.augmenter(image_batch), token_batch

    # 이미지 배치와 토큰 배치를 그대로 반환하면서 텍스트를 인코딩된 벡터로 반환
    def run_text_encoder(
        self, image_batch: tf.Tensor, token_batch: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return (
            image_batch,
            token_batch,
            self.text_encoder([token_batch, POS_IDS], training=False),
        )

    # 데이터를 딕셔너리 형태로 변환
    def prepare_dict(
        self,
        image_batch: tf.Tensor,
        token_batch: tf.Tensor,
        encoded_text_batch: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        return {
            "images": image_batch,
            "tokens": token_batch,
            "encoded_text": encoded_text_batch,
        }

    # 위의 함수들을 사용하여 텍스트와 이미지를 tf.data.Dataset 객체에 맞게 변환
    def prepare_dataset(self) -> tf.data.Dataset:
        all_captions = list(self.data_frame["caption"].values)
        tokenized_texts = np.empty((len(self.data_frame), MAX_PROMPT_LENGTH))
        for i, caption in enumerate(all_captions):
            tokenized_texts[i] = self.process_text(caption)

        image_paths = np.array(self.data_frame["image_path"])

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, tokenized_texts))
        dataset = dataset.shuffle(self.batch_size * 10)
        dataset = dataset.map(self.process_image, num_parallel_calls=AUTO).batch(
            self.batch_size
        )
        dataset = dataset.map(self.apply_augmentation, num_parallel_calls=AUTO)
        dataset = dataset.map(self.run_text_encoder, num_parallel_calls=AUTO)
        dataset = dataset.map(self.prepare_dict, num_parallel_calls=AUTO)
        return dataset.prefetch(AUTO)
