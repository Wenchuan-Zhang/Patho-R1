<div align="center" style="font-family: charter;">

<h1><i>Patho-R1</i>:</br>A Multimodal Reinforcement Learning-Based Pathology Expert Reasoner</h1>

<img src="docs/overview.png" width="100%"/>

<br />

<a href="https://arxiv.org/abs/2505.11404" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Patho--R1-red?logo=arxiv" height="20" />
</a>
<a href="https://huggingface.co/WenchuanZhang/Patho-R1-3B">
  <img alt="HF Model: Patho-R1-3B" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-Patho--R1--3B-ffc107?color=ffc107&logoColor=white" height="20" />
</a>
<a href="https://huggingface.co/WenchuanZhang/Patho-R1-7B">
  <img alt="HF Model: Patho-R1-7B" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-Patho--R1--7B-ffc107?color=ffc107&logoColor=white" height="20" />
</a>
</div>

# Introduction📝
While vision-language models have shown impressive progress in general medical domains, pathology remains a challenging subfield due to its high-resolution image requirements and complex diagnostic reasoning.  

To address this gap, we introduce **Patho-R1**, a multimodal pathology reasoner designed to enhance diagnostic understanding through structured reasoning. **Patho-R1** is trained using a three-stage pipeline:  
1. *Continued pretraining* on **3.5M pathology figure-caption pairs** for domain knowledge acquisition  
2. *Supervised fine-tuning* on **500k expert-annotated Chain-of-Thought samples** to encourage reasoning  
3. *Reinforcement learning* with **Group Relative Policy Optimization** and **Decoupled CLIP and Dynamic sAmpling Policy Optimization** to refine response quality  

Experimental results show that **Patho-R1** achieves strong performance across key pathology tasks, including **multiple choice questions** and **visual question answering**, highlighting its potential for real-world pathology AI applications.

## TODOS📌
- [x] `2025-05-29` ⭐️: Initial release of Patho-R1 models and inference pipeline
- [ ] 🎯: Release Patho-CLIP model weights and evaluation script
- [ ] 🎯: Release training code and detailed dataset construction pipeline

# Installation🛠️
```bash
# Create and activate a new conda environment
conda create -n patho-r1 python=3.10 -y  
conda activate patho-r1

# Clone the repository and install dependencies
git clone https://github.com/Wenchuan-Zhang/Patho-R1.git
cd Patho-R1
pip install -e .
```

# Training🚉
```bash
torchrun 
```

# Acknowledgements✨
We gratefully acknowledge the contributions of the open-source community, particularly the following projects which laid the foundation for various components of this work:

- [Qwen](https://github.com/QwenLM) for providing powerful vision language models that significantly advanced our multimodal understanding and generation capabilities.  
- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) for document layout detection.  
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for comprehensive optical character recognition.  
- [ModelScope Swift](https://github.com/modelscope/ms-swift) for efficient model serving and deployment tools.  
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for robust LLM training and fine-tuning pipelines.  
- [VERL](https://github.com/volcengine/verl) for valuable visual-language pretraining resources.  
- [DeepSeek](https://github.com/deepseek-ai) for high-quality models and infrastructure supporting text understanding.

We thank the authors and contributors of these repositories for their dedication and impactful work, which made our development of Patho-R1 possible.

# Citation❤️
If you find our work helpful, a citation would be greatly appreciated. Also, consider giving us a star ⭐ on GitHub to support the project!

```
@article{zhang2025patho,
  title={Patho-R1: A Multimodal Reinforcement Learning-Based Pathology Expert Reasoner},
  author={Zhang, Wenchuan and Zhang, Penghao and Guo, Jingru and Cheng, Tao and Chen, Jie and Zhang, Shuwan and Zhang, Zhang and Yi, Yuhao and Bu, Hong},
  journal={arXiv preprint arXiv:2505.11404},
  year={2025}
}
```