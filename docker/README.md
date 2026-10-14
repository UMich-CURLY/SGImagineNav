# Docker

The image includes SGImagineNav's dependencies, GroundingDINO/SAM checkpoints and Ollama. To get started, build it from the repository root:

```bash
bash docker/build.sh
```

Fill in `.env` as described in the [main README](../README.md#run-evaluation), then open a shell in the container:

```bash
bash docker/start.sh
```

The repository is mounted at `/workspace/SG-VLN`, your datasets at `/datasets`, and model downloads are stored in a persistent Docker volume. Follow the [dataset and evaluation instructions](../README.md#prepare-the-dataset) in the main README.

You can check that the GPU and Habitat renderer are working with:

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace/SG-VLN" \
  sg-vln:hm3d python scripts/check_installation.py
```
