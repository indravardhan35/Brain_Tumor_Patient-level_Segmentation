New-Item -ItemType Directory -Force cv_results\logs | Out-Null
foreach ($m in "patient","random") { foreach ($f in 0..4) {
  $csv = "cv_results\baseline_${m}_fold$f.csv"
  if (Test-Path $csv) { Write-Host "skip $m fold $f (already done)"; continue }
  $start = Get-Date
  $tag = "_${m}_f$f"
  py -3.13 -u train_aspp.py --model baseline --seed 0 --split_file "cv_folds\${m}_fold$f.json" --tag $tag *> "cv_results\logs\train_${m}_fold$f.log"
  if ($LASTEXITCODE -ne 0) { Write-Host "TRAINING FAILED on $m fold $f. Stopping. See cv_results\logs\train_${m}_fold$f.log"; exit 1 }
  py -3.13 -u eval_cv.py --split_file "cv_folds\${m}_fold$f.json" --checkpoint "best_model_baseline_seed0$tag.pth" --model baseline --out $csv *> "cv_results\logs\eval_${m}_fold$f.log"
  if ($LASTEXITCODE -ne 0) { Write-Host "EVAL FAILED on $m fold $f. Stopping."; Remove-Item $csv -ErrorAction SilentlyContinue; exit 1 }
  $mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
  "$m fold $f : $mins min" | Add-Content cv_results\logs\timing.txt
  Write-Host "$m fold $f done in $mins min"
} }
Write-Host "ALL DONE"
