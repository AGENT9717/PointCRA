# PointCRA

This is the official code repository for **PointCRA**, a point cloud analysis network proposed in our paper:

> **Channel-Level Relation to Attentive Aggregation with Neighborhood-Homogeneity Constraint for Point Cloud Analysis**

Pretrained weights and training logs are available on [Hugging Face][https://huggingface.co/agent9717/PointCRA](https://huggingface.co/agent9717/PointCRA)

This repository provides code for two frameworks: **DeLA** and **OpenPoint**.

---

## DeLA

The DeLA version provides a complete and runnable codebase.

- Follow the setup instructions of [DeLA_v2](https://github.com/Matrix-ASC/DeLA_v2) to configure the environment and datasets.
- Run the following command to start training:

```bash
python train.py
```

---

## OpenPoint

The OpenPoint version contains only the modified files based on [PointNeXt](https://github.com/guochengqian/PointNeXt). Please copy the files to their corresponding paths according to the folder names (remember to update the respective `__init__.py` files accordingly). Follow the environment and dataset setup instructions of [PointNeXt (OpenPoint)](https://github.com/guochengqian/PointNeXt), then start training using the following command:

```bash
CUDA_VISIBLE_DEVICES=$GPUs python examples/$task_folder/main.py --cfg $cfg $kwargs
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{shi2025pointcra,
  title     = {Channel-Level Relation to Attentive Aggregation with Neighborhood-Homogeneity Constraint for Point Cloud Analysis},
  author    = {Jiaqi Shi, Jin Xiao, Xiaoguang Hu, Wenxuan Ji, Zichong Jia, Zifan Long, and Tianyou Chen},
  journal   = {arXiv preprint},
  year      = {2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
