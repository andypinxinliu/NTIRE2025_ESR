# [NTIRE 2025 Challenge on Efficient Super-Resolution](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

<div align=center>
<img src="https://github.com/Amazingren/NTIRE2025_ESR/blob/main/figs/logo.png" width="400px"/> 
</div>

## Team Rochester 
ID: 38
Team Leader: Pinxin Liu
Contact: andypinxinliu@gmail.com
Team Members: Pinxin Liu, Yongsheng Yu, Hang Hua, Yunlong Tang



## The Environments

The evaluation environments adopted by us is recorded in the `requirements.txt`. After you built your own basic Python (Python = 3.9 in our setting) setup via either *virtual environment* or *anaconda*, please try to keep similar to it via:

- Step1: install Pytorch first:
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`

- Step2: install other libs via:
```pip install -r requirements.txt```

or take it as a reference based on your original environments.

## The Validation datasets and test dataset download
After downloaded all the necessary validate dataset ([DIV2K_LSDIR_valid_LR](https://drive.google.com/file/d/1YUDrjUSMhhdx1s-O0I1qPa_HjW-S34Yj/view?usp=sharing) and [DIV2K_LSDIR_valid_HR](https://drive.google.com/file/d/1z1UtfewPatuPVTeAAzeTjhEGk4dg2i8v/view?usp=sharing), and [DIV2K_LSDIR_test_LR](https://drive.google.com/file/d/12MatzcLHUuvQYfYASMBtg39pRuSqNskj/view?usp=sharing))
please organize them as follows:

```
|NTIRE2025_ESR_Challenge/
|--DIV2K_LSDIR_valid_HR/
|    |--000001.png
|    |--000002.png
|    |--...
|    |--000100.png
|    |--0801.png
|    |--0802.png
|    |--...
|    |--0900.png
|--DIV2K_LSDIR_valid_LR/
|    |--000001x4.png
|    |--000002x4.png
|    |--...
|    |--000100x4.png
|    |--0801x4.png
|    |--0802x4.png
|    |--...
|    |--0900.png
|--DIV2K_LSDIR_test_LR/
|    |--000001x4.png
|    |--000002x4.png
|    |--...
|    |--000100x4.png
|    |--0801x4.png
|    |--0802x4.png
|    |--...
|    |--0900.png
|--NTIRE2025_ESR/
|    |--...
|    |--test_demo.py
|    |--...
|--results/
|--......
```

## How to test the baseline model?

1. `git clone https://github.com/andypinxinliu/NTIRE2025_ESR.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 38
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.
3. More detailed example-command can be found in `run.sh` for your convenience.

The result of our method (ESRNet):
- Average PSNR on DIV2K_LSDIR_valid: 26.94 dB
- Average PSNR on DIV2K_LSDIR_test: 27.01 dB
- Number of parameters: 0.157 M
- Runtime: 17.26  ms (on Valid Dataset)
- FLOPs on an LR image of size 256×256: 10.30 G

    Note that the results reported above are the average of 5 runs, and each run is conducted on the same device (i.e., NVIDIA RTX A6000 GPU).

## How to run our model?

1. `git clone https://github.com/andypinxinliu/NTIRE2025_ESR.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 38
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.
3. More detailed example-command can be found in `run.sh` for your convenience.