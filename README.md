# Getting Started

## Dependencies and Installation

- Pytorch >= 1.7.1
- CUDA >= 10.1
- Other required packages in `requirements.txt`
```

# create new anaconda env
conda create -n keep python=3.10 -y
conda activate keep

# install python dependencies
pip3 install -r requirements.txt
python setup.py develop
conda install -c conda-forge ffmpeg 
```

[Optional] If you forget to clone the repo with `--recursive`, you can update the submodule by 
```
git submodule init
git submodule update
```

## Quick Inference

### Prepare Testing Data
We provide both synthetic (VFHQ) and real (collected) examples in `assets/examples` folder. If you would like to test your own face videos, place them in the same folder.



### Inference
**[Note]** We have prepared two pre-trained networks, `weights/KEEP/KEEP-b76feb75.pth` is trained on VFHQ datasets and `KEEP_Asian-4765ebe0.pth` is fine-tuned on a combination of VFHQ and part of collected Asian face datasets. See the report for more details.


Video Face Restoration for synthetic data (cropped and aligned face)
```
# Add '--bg_upsampler realesrgan' to enhance the background regions with Real-ESRGAN
# Add '--face_upsample' to further upsample restorated face with Real-ESRGAN
# Add '--draw_box' to show the bounding box of detected faces.
# Add '--has_aligned' to specify the video faces are already cropped and aligned.

# Specify '--model_type' to 'KEEP' or 'Asian' for two models.

# For cropped and aligned faces
python inference_keep.py -i=./assets/examples/synthetic_1.mp4 --has_aligned --save_video --model_type=KEEP

# For real data in the wild
python inference_keep.py -i=./assets/examples/real_1.mp4 --draw_box --save_video --bg_upsampler=realesrgan --model_type=KEEP

# For Asian data
python inference_keep.py -i=./assets/examples/asian_1.mp4 --draw_box --save_video --model_type=Asian
```


## Training

To train your own model, this repo follows a three-stage training process.

First, train a VQGAN.
```
python -u basicsr/train.py -opt options/train/stage1_VQGAN.yml --launcher="slurm"
```

Then, train the main part of KEEP.
```
python -u basicsr/train.py -opt options/train/stage2_KEEP.yml --launcher="slurm"
```

Finally, fine-tune the CFT and CFA.
```
python -u basicsr/train.py -opt options/train/stage3_KEEP.yml --launcher="slurm"
```


## Contact
If you have any question, please feel free to contact us via `ruicheng002@ntu.edu.sg`.
