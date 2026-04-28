# Run Commands Reference

Custom images `airbnb-cpu` and `airbnb-gpu` have all dependencies pre-baked — no pip
install step needed. Build once with:

```bash
docker build -f docker/Dockerfile.cpu -t airbnb-cpu .
docker build -f docker/Dockerfile.gpu -t airbnb-gpu .
```

**Monitor a running container:** `docker logs -f <container_name>`  
**Check if still running:** `docker ps`  
**See exit status of finished container:** `docker ps -a`  
**Remove a finished container:** `docker rm <container_name>`

HuggingFace models are cached at `/mnt/nvme_data/linux_sys/ml` — downloaded once,
reused by every container via the two volume/env flags below.

```
HF_CACHE="-v /mnt/nvme_data/linux_sys/ml:/hf_cache -e HF_HOME=/hf_cache -e TRANSFORMERS_CACHE=/hf_cache/hub"
```

---

## LoRA Models (GPU — `airbnb-gpu`)

### fusion_lora.py

**Smoke test (run first to verify no errors):**
```bash
docker run --rm --gpus all --shm-size=8g \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  -v /mnt/nvme_data/linux_sys/ml:/hf_cache \
  -e HF_HOME=/hf_cache -e TRANSFORMERS_CACHE=/hf_cache/hub \
  airbnb-gpu python scripts/models/fusion_lora.py \
  --variant normal_bc --image-size 224 --lora-rank 16 --fusion-head deep_256 \
  --smoke-test --workers 0
```

**Run #31 — PRIORITY 1 (224px, rank 16, deep_256):** ← already running as `fusion_lora_priority1`
```bash
docker run -d --name fusion_lora_priority1 --gpus all --shm-size=8g \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  -v /mnt/nvme_data/linux_sys/ml:/hf_cache \
  -e HF_HOME=/hf_cache -e TRANSFORMERS_CACHE=/hf_cache/hub \
  airbnb-gpu python scripts/models/fusion_lora.py \
  --variant normal_bc --image-size 224 --lora-rank 16 --fusion-head deep_256 \
  --batch-size 16 --accum-steps 2 --workers 4 --run-name priority1
```

**Run #32 — PRIORITY 2 (336px, rank 16, deep_256):**
```bash
docker run -d --name fusion_lora_priority2 --gpus all --shm-size=8g \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  -v /mnt/nvme_data/linux_sys/ml:/hf_cache \
  -e HF_HOME=/hf_cache -e TRANSFORMERS_CACHE=/hf_cache/hub \
  airbnb-gpu python scripts/models/fusion_lora.py \
  --variant normal_bc --image-size 336 --lora-rank 16 --fusion-head deep_256 \
  --batch-size 16 --accum-steps 2 --workers 4 --run-name priority2
```

**Run #35 — head ablation (336px, rank 16, shallow_64):**
```bash
docker run -d --name fusion_lora_head_ablation --gpus all --shm-size=8g \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  -v /mnt/nvme_data/linux_sys/ml:/hf_cache \
  -e HF_HOME=/hf_cache -e TRANSFORMERS_CACHE=/hf_cache/hub \
  airbnb-gpu python scripts/models/fusion_lora.py \
  --variant normal_bc --image-size 336 --lora-rank 16 --fusion-head shallow_64 \
  --batch-size 16 --accum-steps 2 --workers 4 --run-name head_ablation
```

**Run #36 — rank ablation (336px, rank 8, deep_256):**
```bash
docker run -d --name fusion_lora_rank_ablation --gpus all --shm-size=8g \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  -v /mnt/nvme_data/linux_sys/ml:/hf_cache \
  -e HF_HOME=/hf_cache -e TRANSFORMERS_CACHE=/hf_cache/hub \
  airbnb-gpu python scripts/models/fusion_lora.py \
  --variant normal_bc --image-size 336 --lora-rank 8 --fusion-head deep_256 \
  --batch-size 16 --accum-steps 2 --workers 4 --run-name rank_ablation
```

---

### image_lora.py

**Smoke test:**
```bash
docker run --rm --gpus all --shm-size=8g \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  -v /mnt/nvme_data/linux_sys/ml:/hf_cache \
  -e HF_HOME=/hf_cache -e TRANSFORMERS_CACHE=/hf_cache/hub \
  airbnb-gpu python scripts/models/image_lora.py \
  --variant normal_bc --image-size 336 --lora-rank 16 --fusion-head deep_256 \
  --smoke-test --workers 0
```

**Run #33 — ablation (336px, rank 16, deep_256):**
```bash
docker run -d --name image_lora_ablation --gpus all --shm-size=8g \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  -v /mnt/nvme_data/linux_sys/ml:/hf_cache \
  -e HF_HOME=/hf_cache -e TRANSFORMERS_CACHE=/hf_cache/hub \
  airbnb-gpu python scripts/models/image_lora.py \
  --variant normal_bc --image-size 336 --lora-rank 16 --fusion-head deep_256 \
  --batch-size 16 --accum-steps 2 --workers 4 --run-name ablation
```

---

### text_lora.py

**Smoke test:**
```bash
docker run --rm --gpus all --shm-size=8g \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  -v /mnt/nvme_data/linux_sys/ml:/hf_cache \
  -e HF_HOME=/hf_cache -e TRANSFORMERS_CACHE=/hf_cache/hub \
  airbnb-gpu python scripts/models/text_lora.py \
  --variant normal_bc --lora-rank 16 --fusion-head deep_256 \
  --smoke-test --workers 0
```

**Run #34 — ablation (rank 16, deep_256):**
```bash
docker run -d --name text_lora_ablation --gpus all --shm-size=8g \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  -v /mnt/nvme_data/linux_sys/ml:/hf_cache \
  -e HF_HOME=/hf_cache -e TRANSFORMERS_CACHE=/hf_cache/hub \
  airbnb-gpu python scripts/models/text_lora.py \
  --variant normal_bc --lora-rank 16 --fusion-head deep_256 \
  --batch-size 32 --accum-steps 1 --workers 4 --run-name ablation
```

---

## Tree / Classical Models (CPU — `airbnb-cpu`)

### decision_tree.py

**Smoke test:**
```bash
docker run --rm \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/decision_tree.py --variant normal_raw --smoke-test
```

**Runs #1–4 (all four variants):**
```bash
# Run #1
docker run -d --name dt_normal_raw \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/decision_tree.py --variant normal_raw

# Run #2
docker run -d --name dt_cleaned_raw \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/decision_tree.py --variant cleaned_raw

# Run #3
docker run -d --name dt_normal_bc \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/decision_tree.py --variant normal_bc

# Run #4
docker run -d --name dt_cleaned_bc \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/decision_tree.py --variant cleaned_bc
```

### random_forest.py

**Smoke test:**
```bash
docker run --rm \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/random_forest.py --variant normal_raw --smoke-test
```

**Runs #5–6:**
```bash
# Run #5
docker run -d --name rf_normal_raw \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/random_forest.py --variant normal_raw

# Run #6
docker run -d --name rf_cleaned_raw \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/random_forest.py --variant cleaned_raw
```

### gradient_boosting.py

**Smoke test:**
```bash
docker run --rm \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/gradient_boosting.py --variant normal_raw --smoke-test
```

**Runs #7–8:**
```bash
# Run #7
docker run -d --name gb_normal_raw \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/gradient_boosting.py --variant normal_raw

# Run #8
docker run -d --name gb_cleaned_raw \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/gradient_boosting.py --variant cleaned_raw
```

### lightgbm_model.py

**Smoke test:**
```bash
docker run --rm \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/lightgbm_model.py --variant normal_raw --smoke-test
```

**Runs #9–10:**
```bash
# Run #9
docker run -d --name lgb_normal_raw \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/lightgbm_model.py --variant normal_raw

# Run #10
docker run -d --name lgb_cleaned_raw \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/lightgbm_model.py --variant cleaned_raw
```

### ridge_model.py

**Smoke test:**
```bash
docker run --rm \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/ridge_model.py --variant normal_raw --smoke-test
```

**Runs #11–12:**
```bash
# Run #11
docker run -d --name ridge_normal_raw \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/ridge_model.py --variant normal_raw

# Run #12
docker run -d --name ridge_cleaned_raw \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/ridge_model.py --variant cleaned_raw
```

### polynomial_ridge.py

**Smoke test:**
```bash
docker run --rm \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/polynomial_ridge.py --variant normal_raw --smoke-test
```

**Runs #13–14:**
```bash
# Run #13
docker run -d --name poly_ridge_normal_raw \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/polynomial_ridge.py --variant normal_raw

# Run #14
docker run -d --name poly_ridge_cleaned_raw \
  -v /home/admin/ML-AirBnb-Price-Estimator:/workspace -w /workspace \
  airbnb-cpu python scripts/models/polynomial_ridge.py --variant cleaned_raw
```

---

## Monitoring Cheatsheet

```bash
docker ps                          # running containers
docker ps -a                       # all containers including finished
docker logs -f <name>              # stream live logs
docker logs --tail 50 <name>       # last 50 lines
docker rm <name>                   # clean up finished container
docker rm $(docker ps -aq -f status=exited)  # clean up all finished containers
```
