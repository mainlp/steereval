## INCLINE but for metrics

```
python intervention_comet.py
```

Args: "model", "dimension", "data", "prompting_lang", "gpu_id" "metric" "--layers"
```
python intervention_gptscore.py bigscience/bloomz-7b1-mt coherence ncwm_old tgt 0 accuracy --layers 0 1 2 3
```

