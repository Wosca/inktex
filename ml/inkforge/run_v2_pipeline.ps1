$ErrorActionPreference = "Stop"

$Py = "C:\Users\oscar\Home\Github\math-recogniser\ml\.venv\Scripts\python.exe"
$Repo = "C:\Users\oscar\Home\Github\inktex"
$CacheV2 = "E:\math-recogniser-training\inkforge_data_v2"
$Stage1Out = "E:\math-recogniser-training\inkforge_v2_model"
$Stage2Out = "E:\math-recogniser-training\inkforge_v2_star_dpo"

New-Item -ItemType Directory -Force -Path $Stage1Out | Out-Null
New-Item -ItemType Directory -Force -Path $Stage2Out | Out-Null

Set-Location $Repo

Write-Output "[pipeline] === STAGE 1: InkForge v2 Pretraining (62.7M, 860k samples, text CFG + sigma_min=0.008) ==="
& $Py -u ml\inkforge\train_inkforge.py `
  --cache-dir $CacheV2 `
  --out-dir $Stage1Out `
  --d-model 512 `
  --n-heads 8 `
  --d-ff 2048 `
  --enc-layers 6 `
  --dec-layers 10 `
  --style-dim 64 `
  --mdn-components 20 `
  --dropout 0.10 `
  --sigma-min 0.008 `
  --text-drop-prob 0.10 `
  --preview-cfg-scale 1.5 `
  --grad-checkpoint `
  --batch-size 32 `
  --grad-accum 2 `
  --total-steps 60000 `
  --warmup-steps 1500 `
  --kl-warmup-steps 5000 `
  --lr 3.5e-4 `
  --kl-weight 0.025 `
  --free-bits 0.15 `
  --style-drop-prob 0.30 `
  --input-noise 0.010 `
  --guided-attn-weight 4.0 `
  --guided-attn-sigma 0.24 `
  --transition-weight 4.0 `
  --eval-every 2000 `
  --log-every 100 `
  --num-workers 6 `
  1>> "$Stage1Out\train_stdout.log" 2>> "$Stage1Out\train_stderr.log"

if ($LASTEXITCODE -ne 0) {
  throw "Stage 1 train_inkforge.py exited with code $LASTEXITCODE"
}

Write-Output "[pipeline] === STAGE 2: InkTeX-Guided STaR + DPO Post-Training ==="
& $Py -u ml\inkforge\star_dpo.py `
  --init-ckpt "$Stage1Out\best.pt" `
  --out-dir $Stage2Out `
  --cache-dir $CacheV2 `
  --rounds 4 `
  --prompts-per-round 1200 `
  --rollouts-per-prompt 6 `
  --gen-batch-size 12 `
  --judge-batch-size 32 `
  --train-batch-size 12 `
  --max-updates-per-round 200 `
  --rollout-temp 0.45 `
  --cfg-scale 1.5 `
  --dpo-beta 0.003 `
  --sft-weight 0.25 `
  --replay-weight 0.35 `
  --lr 1e-5 `
  --grad-checkpoint `
  1>> "$Stage2Out\star_dpo_stdout.log" 2>> "$Stage2Out\star_dpo_stderr.log"

if ($LASTEXITCODE -ne 0) {
  throw "Stage 2 star_dpo.py exited with code $LASTEXITCODE"
}

Write-Output "[pipeline] === ALL STAGES COMPLETE ==="
