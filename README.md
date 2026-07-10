# Revisiting Pose Sensitivity in Splat-based Computed Tomography under Sparse-view Reconstruction
X-ray computed tomography (CT) reconstructs volumetric representations of objects from projection images obtained by transmitting X-rays through a target. Recent splat-based tomography, which represents a volume as a continuous distribution of 3D Gaussians, has demonstrated both high reconstruction quality and fast convergence in cone-beam sparse-view CT. However, when deployed in real CT systems with limited and non-uniform view distributions, we observe distinctive streak and strip artifacts that are far more pronounced than in conventional reconstruction methods. Through detailed analysis, we show that these artifacts primarily originate from pose inaccuracies in the acquisition geometry rather than from view sparsity itself. We revisit pose sensitivity in the splatting formulation and derive a stable gradient-based framework that jointly refines geometric parameters during reconstruction. Our study not only identifies how pose perturbations propagate through the differentiable projection operator but also reveals why splat-based CT is particularly vulnerable to geometric misalignment. The resulting formulation remains lightweight and easily integrable into existing pipelines while substantially improving reconstruction fidelity under real-world sparse-view conditions.

### [Project page](https://vclab.kaist.ac.kr/cvpr2026p2/index.html) | [Paper](https://vclab.kaist.ac.kr/cvpr2026p2/cbct_pose_main.pdf) | [Supplemental](https://vclab.kaist.ac.kr/cvpr2026p2/cbct_pose_supp.pdf)
[Kiseok Choi](https://sites.google.com/view/kiseokchoi), 
[Hyeongjun Cho](http://vclab.kaist.ac.kr/hjcho/index.html), 
[Inchul Kim](https://inchul-kim.github.io/), 
[Min H. Kim](http://vclab.kaist.ac.kr/minhkim/index.html)

This repository consists of runnable codes as:

- `data_generator/generate_data_with_noisy_pose.py`: Generates the randomly transformed projections
- `initialize_pcd.py`: Initializes the Gaussian point cloud
- `train.py`: Runs CBCT reconstruction with pose calibration

## Tested Environment
```
CPU: Intel Xeon 4214R @ 2.4Ghz
GPU: NVIDIA A6000 48GB
RAM: 256GB
OS: Ubuntu 22.04
Docker: 24.0.2
CUDA: 13.1 (Driver: 590.48.01)
```

## Setup
Ensure that your system supports CUDA and Docker. Then clone this repository and move into the project directory.
```
git clone https://github.com/KAIST-VCLAB/ct-pose-calibration.git
cd ct-pose-calibration
```

Build the Docker image using Dockerfile. This process may take several minutes.
```
docker build -t r2gs_pose:cuda118 .
```
Launch the Docker container. Modify the options below according to your environment.
```
docker run -it --gpus all \
  --name r2gs_pose \
  -v /mnt/datassd/kschoi/data:/workspace/data \
  -p 20000:20000 \
  r2gs_pose:cuda118
```

## Data Preparation
Download one of the datasets from [link](https://drive.google.com/drive/folders/15hwnG4oUeEz-t1Psrwi0Zm--22L_T70O?usp=drive_link) and extract it to your preferred directory.

## Gaussian Initialization
Run the following command to initialize the Gaussian representation.<br>
($SCENE_PATH can be ``walnut`` or something).
```
python initialize_pcd.py --data $SCENE_PATH
```
This step can be skipped if the dataset directory already contains initialized Gaussians (e.g., ``init_$SCENE_PATH.npy``).

## Reconstruction
Modify ``source_path`` in ``./config/default.yaml`` and start reconstruction using:
```
python train.py --config $CONFIG_FILE_PATH
```
After optimization completes (30,000 iterations), the reconstructed volume will be saved to:
```
./output/$CONFIG_FILE_NAME_MM-DD-hh-mm-ss/point_cloud/iteration_30000/vol_pred.npy
```

## Citation
```	
@InProceedings{Choi_2026p2_CVPR,
   author = {Choi, Kiseok and Cho, Hyeongjun and Kim, Inchul and Kim, Min H.},
   title = {Revisiting Pose Sensitivity in Splat-based Computed 
            Tomography under Sparse-view Reconstruction},
   booktitle = {IEEE Conference on Computer Vision and 
      Pattern Recognition (CVPR)},
   month = {June},
   year = {2026}
} 
```

## License

This project is released under the MIT License.

This code is built upon and contains modifications of the [R2-Gaussian](https://github.com/ruyi-zha/r2_gaussian) project, which is also distributed under the MIT License. We gratefully acknowledge the authors for making their code publicly available.

See the LICENSE file for details.
