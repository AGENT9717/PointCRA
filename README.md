# PointCRA: Point Cloud Analysis Network with Channel-Level Relation for Attentive Aggregation

This is the official code repository for **PointCRA**. The corresponding pretrained weights and training logs are available on Hugging Face:  
[https://huggingface.co/agent9717/PointCRA](https://huggingface.co/agent9717/PointCRA)

## Supported Frameworks

This repository provides code for two frameworks: **DeLA** and **OpenPoint**.

### DeLA

The DeLA version provides a complete and runnable codebase. 
- Follow the setup instructions of [DeLA_v2](https://github.com/Matrix-ASC/DeLA_v2) to configure the environment and datasets.
- Simply run the following command to start training:
  ```bash
  python train.py
