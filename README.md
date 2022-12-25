# Fine-tuning Stable Diffusion using Keras

This repository provides code for fine-tuning [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) in Keras. It is adapted from this [script by Hugging Face](https://github.com/fchollet/stable-diffusion-tensorflow/blob/master/text2image.py). The pre-trained model used for fine-tuning comes from [KerasCV](https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion). To know about the original model check out [this documentation](https://huggingface.co/CompVis/stable-diffusion-v1-4).  

**The code provided in this repository is for research purposes only**. Please check out [this section](https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion#uses) to know more about the potential use cases and limitations.

By loading this model you accept the CreativeML Open RAIL-M license at https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE.

[Add results]

## Dataset 

Following the original script from Hugging Face, this repository also uses the [Pokemon dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions). But it was regenerated to suit this repository. The regenerated version of the dataset is hosted [here](https://huggingface.co/datasets/sayakpaul/pokemon-blip-original-version). Check out that link for more details.

## Training

Fine-tuning code is provided in `finetune.py`. Before running training, ensure you have the dependencies (refer to `requirements.txt`) installed.

You can launch training with the default arguments by running `python finetune.py`. Run `python finetune.py -h` to know about the supported command-line arguments.

For avoiding OOM and faster training, it's recommended to use a V100 GPU at least. We used an A100.

**Some details to note**:

* Only the diffusion model is fine-tuned.The image encoder and the text encoder are kept frozen. 
* Mixed-precision training is not yet supported. As a result, instead of 512x512 resolution, this repository uses 256x256.
* Distributed training is not yet supported. 
* One major difference from the Hugging Face implementation is that the EMA averaging of weights doesn't follow any schedule for the decay factor.

You can find the fine-tuned diffusion model weights [here](https://huggingface.co/sayakpaul/kerascv_sd_pokemon_finetuned/tree/main). 

## Inference

Upcoming

## Results

Upcoming

## Acknowledgements

* Thanks to Hugging Face for providing the fine-tuning script. It's quite readable.
* Thanks to the ML Developer Programs' team at Google for providing GCP credits.