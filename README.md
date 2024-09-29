## Head360: Learning a Parametric 3D Full-Head for Free-View Synthesis in 360°


<h1 align='Center'>Head360: Learning a Parametric 3D Full-Head for Free-View Synthesis in 360°</h1>

<div align='Center'>
    Yuxiao He<sup>1</sup>,</span>&emsp;
    <a href='https://scholar.google.com/citations?user=hk-3z3UAAAAJ&hl=en' target='_blank'>Yiyu Zhuang</a><sup>1</sup>,</span>&emsp;
    Yanwen Wang<sup>1</sup>,</span>&emsp;
    <a href='https://yoyo000.github.io/' target='_blank'>Yao YAo</a><sup>1</sup>,
            </span>&emsp;
    <a href='https://sites.google.com/site/zhusiyucs/home' target='_blank'>Siyu Zhu</a><sup>2</sup>&emsp;
    <a href='https://xiaoyu258.github.io/' target='_blank'>Xiaoyu Li</a><sup>3</sup>&emsp;
    <a href='https://xiaoyu258.github.io/' target='_blank'>Qi Zhang</a><sup>3</sup>&emsp;
    <a href='https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html' target='_blank'>Xun Cao</a><sup>1</sup>&emsp;
    <a href='http://zhuhao.cc/home/' target='_blank'>Hao Zhu</a><sup>+1</sup>&emsp;
</div>
<div align='Center'>
    <sup>1 </sup>State Key Laboratory for Novel Software Technology, Nanjing University, China
    <sup>2 </sup>Fudan University <sup>3 </sup>Tencent AI Lab, Shenzhen, China
</div>
<div align='Center'>
<i><strong><a href='https://eccv2024.ecva.net' target='_blank'>ECCV 2024</a></strong></i>
</div>


[**Project**](https://nju-3dv.github.io/projects/Head360) | [**Paper**](https://arxiv.org/abs/2408.00296) | [**Youtube**](https://www.youtube.com/watch?v=wuY8gA8G4OI)


Abstract:*Creating a 360° parametric model of a human head is a very challenging task. While recent advancements have demonstrated the efficacy of leveraging synthetic data for building such parametric head models, their performance remains inadequate in crucial areas such as expression-driven animation, hairstyle editing, and text-based modifications. In this paper, we build a dataset of artist-designed high-fidelity human heads and propose to create a novel parametric 360° renderable parametric head model from it. Our scheme decouples the facial motion/shape and facial appearance, which are represented by a classic parametric 3D mesh model and an attached neural texture, respectively. We further propose a training method for decompositing hairstyle and facial appearance, allowing free-swapping of the hairstyle. A novel inversion fitting method is presented based on single image input with high generalization and fidelity. To the best of our knowledge, our model is the first parametric 3D full-head that achieves 360° free-view synthesis, image-based fitting, appearance editing, and animation within a single model. Experiments show that facial motions and appearances are well disentangled in the parametric space, leading to SOTA performance in rendering and animating quality.*

## Requirements

* 1&ndash;8 high-end NVIDIA GPUs. We have done all testing and development using V100, RTX3090, and A100 GPUs.
* 64-bit Python 3.9 and PyTorch 1.12.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.3 or later.
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `cd Head360`
  - `conda env create -f environment.yml`
  - `conda activate head360`

## Getting started
See [project page](https://nju-3dv.github.io/projects/Head360/#section_data) to get our training data.
The dataset should be organized as below:

### Preparing datasets(stage1)
```
    ├── /path/to/dataset
    │   ├── images
            ├── split_512
                ├── fs_002
                    ├── browDwonLeft
                        ├── 0000.png
                        ├── ...
                    ├── ...
                ├── ...
            ├── split_512_hs
            ├── dataset_pose.json
            ├── total_dataset_exp.json
        ├── mesh_lms
            ├── meshes
                ├── fs_002_browDownLeft.obj
                ├── ...
            ├── lms
                ├── fs_002_browDownRight.txt
                ├── ...

```
You can train new origin networks using `train_head360.py`. For example:
```.bash
# Train with Synhead360  with raw neural rendering resolution=64, using 8 GPUs.

python train_head360.py --outdir=~/training-runs --cfg=synhead --data=/path/to/dataset/images
  --rdata /path/to/dataset/mesh_lms --gpus=8
--batch=32
--gamma=4
--topology_path=data/mh/template.obj
--gen_pose_cond=True
--gen_exp_cond=True
--disc_c_noise=0
--load_lms=True
--model_version=next3d
--discriminator_version=DualLabelDualDiscriminator
```

### Preparing datasets(stage2)
```
    coming soon

```
You can train new origin networks using `train_head360.py`. For example:
```.bash
# Train with Synhead360  with raw neural rendering resolution=64, using 8 GPUs.

python train_head360_st2.py --outdir=~/training-runs --cfg=synhead --data=/path/to/dataset/images
  --rdata /path/to/dataset/mesh_lms --gpus=8
--batch=32
--gamma=4
--topology_path=data/mh/template.obj
--gen_pose_cond=True
--gen_exp_cond=True
--disc_c_noise=0
--load_lms=True
--model_version=next3d
--discriminator_version=DualLabelDualDiscriminator
```


## Citation
If you find Head360 useful for your work please cite:

```
@inproceedings{he2024head360,
  title={Head360: Learning a Parametric 3D Full-Head for Free-View Synthesis in 360 degrees},
  author={He, Yuxiao and Zhuang, Yiyu and Wang, Yanwen and Yao, Yao and Zhu, Siyu and Li, Xiaoyu and Zhang, Qi and Cao, Xun and Zhu, Hao},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

## Acknowledgements

Part of the code is borrowed from [EG3D](https://github.com/NVlabs/eg3d) and[Next3D](https://github.com/MrTornado24/Next3D).
