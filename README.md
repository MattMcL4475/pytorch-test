# GPU Audio – Wee Cooper o' Fife (Fife Synthesis)

This project now renders an automatically synthesized fife (breathy flute) performance of the traditional Scottish tune *“Wee Cooper o' Fife”* using PyTorch (GPU if available, CPU otherwise). The prior electric guitar / Star‑Spangled Banner chain has been fully replaced for a lighter, faster demo suitable for quick GPU validation or audio synthesis experimentation.

## Quick Setup

1. **Set up Azure Container Registry:**
   ```bash
   ./setup-acr.sh
   ```

2. **Configure environment variables:**
   ```bash
   cp setup-env-template.sh setup-env.sh
   # Edit setup-env.sh with your namespace and PVC details
   source setup-env.sh
   ```

3. **Run locally or submit to cluster:**
   ```bash
   # Local (writes wee-cooper-of-fife.wav to current dir)
   python gpu-audio.py --bpm 112 --out .

   # Single GPU job (output goes to mounted PVC if you set --out /data/yourdir)
   dev-k8s.submit cmd --gpus 1 -- python gpu-audio.py --out /data/$USER

   # (Optional) Multi-GPU launch is unnecessary now (script uses only one device), but still valid:
   dev-k8s.submit cmd --gpus 1 --replicas 1 -- python gpu-audio.py
   ```

## Project Structure

- `gpu-audio.py` - Main audio processing script (Wee Cooper o' Fife fife synthesis)
- `Dockerfile` - Container definition with PyTorch and dependencies
- `pyproject.toml` - Python project configuration with rats-devtools enabled
- `setup-acr.sh` - Script to create and configure Azure Container Registry
- `setup-env-template.sh` - Environment variables template

## Environment Variables Required

- `RESYS_K8S_NAMESPACE` - Your B200 namespace
- `RESYS_K8S_CLAIM_NAME` - Your PVC claim name (format: pvc-local-[name])
- `DEVTOOLS_IMAGE_REGISTRY` - ACR registry (msftmattmcl.azurecr.io)
- `DEVTOOLS_IMAGE_PUSH_ON_BUILD` - Set to "1" to enable image pushing

## Monitoring Jobs

```bash
# List your running jobs
dev-k8s.ctl get pods,vcjob

# Get logs from a pod
kubectl logs <pod-name>

# Clean up (use cautiously)
dev-k8s.ctl delete vcjob,configmap,secret
```

## Audio Details
* Sample rate: 48 kHz (override with --sr)
* Stereo: simple early reflections (short inter-channel delay) for width
* Synthesis features: harmonic stack, breath noise, vibrato, chiff attack, crossfaded legato

## Output File
`wee-cooper-of-fife.wav` is written to the directory specified by `--out` (default current directory). On the cluster, set `--out` to a mounted persistent volume path (e.g. `/data/$USER`).

## Changing the Melody
Edit the `SCORE` list in `gpu-audio.py` (tuples of `(NOTE, beats, velocity)` where NOTE uses scientific pitch, REST is allowed). BPM adjustable via `--bpm`.

## License / Attribution
Traditional tune. Synthesis code © 2025.