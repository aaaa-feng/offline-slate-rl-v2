# KL=0.05 GeMS + IQL: 验证拓宽云是否能解 reward 崩塌

## 对比逻辑

| 组 | GeMS | 对比 |
|----|------|------|
| 旧 baseline | ideal_init KL=1.0 | `beta_ablation_repreduce` 已有 |
| A | ideal_init KL=0.05 | vs 旧 baseline |
| B | mf_init KL=0.05 | vs A (embedding 是否有额外影响) |

## 矩阵

每组: 2 env × 5 beta (0,2,5,8,10) = 10 runs
两组共 20 runs

## IQL 参数

与 `beta_ablation_repreduce` 完全一致:
```
lambda_bc: 0.0, expectile: 0.8, max_timesteps: 100000
seed: 58407201, actor: gaussian, gru: qv_shared_detach
```

## GeMS ckpt 路径

```
checkpoints/gems/GeMS_{env}_b5_pretrained_latent32_beta0.05_click1.0_seed58407201_{tag}.ckpt
```

其中 tag = ideal_init 或 mf_init

## 启动

```bash
# Group A: ideal_init KL=0.05
bash experiments/action_cloud/kl005_iql/ideal_init/run.sh all

# Group B: mf_init KL=0.05
bash experiments/action_cloud/kl005_iql/mf_init/run.sh all
```