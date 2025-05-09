1. Experiment Results

| Architecture (hidden units) | Learning Rate | Train Accuracy | Validation Accuracy |
|-----------------------------|---------------|----------------|---------------------|
| 64                          | 0.0001        | 67 %           | 57.66 %             |
| 64                          | 0.001         | 84 %           | 53.46 %             |
| 128                         | 0.001         | 95 %           | 54.04 %             |
| 128                         | 0.01          | 63 %           | 55.59 %             |

Analysis  
The best validation accuracy was achieved by the smallest model with the slowest learning rate (64 units, 0.001), likely because its limited capacity and conservative updates prevented overfitting, while larger or faster-learning models either memorized the training data or became unstable during training.

---

2. Running the Docker Image

The image is published on Docker Hub.

```bash
# 1) Pull the image
docker pull ke5102su/hog-classifier:v3

# 2) Reproduce the baseline test (~56.6 % accuracy expected)
docker run --rm ke5102su/hog-classifier:v3 \
  python solution.py --mode test --feature_type hog --num_unit 64

# 3) Example: launch a custom training run
docker run --rm ke5102su/hog-classifier:v3 \
  python solution.py --mode train --feature_type hog --num_unit 128 --lr 0.001
```

---

3. Training Curves (Weights & Biases)

Public W&B links for full metrics and curves:

| Configuration | W&B Run |
|---------------|---------|
| 128 units, LR 0.001 | https://wandb.ai/yuguangsong-stanford-university/hog-training-job/runs/qt3h595z |
| 128 units, LR 0.01  | https://wandb.ai/yuguangsong-stanford-university/hog-training-job/runs/bazgemf8 |
| 64 units,  LR 0.001  | https://wandb.ai/yuguangsong-stanford-university/hog-training-job/runs/guzv7k6w |
| 64 units,  LR 0.0001 | https://wandb.ai/yuguangsong-stanford-university/hog-training-job/runs/gbekzhv1 |

Click any run to inspect accuracy/loss curves and download the saved best model weights.