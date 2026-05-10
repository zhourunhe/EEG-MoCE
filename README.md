# EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts

This is the PyTorch implementation of the EEG-MoCE in our paper:

> **[EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts](https://arxiv.org/abs/2604.12579)**  
> Runhe Zhou, Shanglin Li, Guanxiang Huang, Xinliang Zhou, Qibin Zhao, Motoaki Kawanabe, Yi Ding, Cuntai Guan  
> *International Conference on Machine Learning (ICML), 2026*  

## Dataset

This is the implementation of EEG-MoCE for emotion recognition using the **EAV Dataset**. 
For more information about the dataset, please refer to:
- GitHub repository: https://github.com/nubcico/EAV
- Paper: Lee et al. (2024). EAV: EEG-Audio-Video Dataset for Emotion Recognition in Conversational Contexts. *Scientific Data*, 11, 1026.

This study incorporates trials with all three synchronized modalities (EEG, audio, and video) from 42 subjects.


## Installation

### Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yaml
conda activate EEGMOCE
```

## Usage

### Data preprocessing

Please follow the official preprocessing procedures from the EAV repository. Additionally, convert audio to mel spectrograms and use OpenFace to extract AU features from vision.

- OpenFace: https://github.com/TadasBaltrusaitis/OpenFace

#### Data Format

- **EEG**: `(B, num_electrodes, chunk_size)` - Batch of EEG signals
- **Audio**: `(B, 128, 1024)` - Batch of audio spectrograms  
- **Vision**: `(B, time_length, 35)` - Batch of facial action units 

### Basic Training

```bash
python main-Multimodal.py \
    --eeg_path ./data/EAV_EEG \
    --audio_path ./data/EAV_Audio \
    --vision_path ./data/EAV_Vision_AU
```

Results will be saved in the `results/` directory.

## CBCR License

| Permissions     | Limitations         | Conditions                       |
| --------------- | ------------------- | -------------------------------- |
| ✅ Modification  | ❌ Commercial use   | ⚠️ License and copyright notice |
| ✅ Distribution |                     |                                  |
| ✅ Private use  |                     |                                  |

## Cite

```bibtex
@inproceedings{zhou2026eegmoce,
  title={EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts},
  author={Zhou, Runhe and Li, Shanglin and Huang, Guanxiang and Zhou, Xinliang and Zhao, Qibin and Kawanabe, Motoaki and Ding, Yi and Guan, Cuntai},
  booktitle={Forty-Third International Conference on Machine Learning},
  year={2026}
}
```

