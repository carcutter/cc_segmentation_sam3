# Trailer Segmentation — PSD Export Pipeline

Watches a folder for incoming trailer images and outputs multi-layer PSD files, one per image. Each PSD contains:

- **Input Image** (bottom)
- **trailer** — full silhouette mask from a fine-tuned model
- **16 detail layers** — chain, wheel, axle, hitch, ramp, fender, shadow, etc. from the base model with smart zoom-in for small parts

Processing time is roughly **15–20 seconds per image** on a GPU.

---

## Prerequisites

- Docker + NVIDIA Container Toolkit (for GPU support)
- The `trailer-seg` image loaded locally (see below)

---

## Loading the image

The image is distributed as a `.tar` file. Load it once:

```bash
docker load -i trailer-seg.tar
```

Verify it loaded:

```bash
docker images | grep trailer-seg
```

---

## Running

### Option A — one-shot batch (process all images in a folder and exit)

```bash
docker run --rm --gpus all \
  -v /path/to/your/images:/input \
  -v /path/to/output:/output \
  trailer-seg
```

The container processes every `.jpg`/`.png` in `/input` that doesn't already have a matching `.psd` in `/output`, then exits.

### Option B — folder watcher (leave running, auto-processes new files)

```bash
docker run --gpus all \
  -v /path/to/your/images:/input \
  -v /path/to/output:/output \
  --name trailer-seg \
  --restart unless-stopped \
  trailer-seg
```

Drop images into the input folder at any time. The container checks every 10 seconds, waits for each file to finish copying (stable size check), then processes it. Already-exported images are skipped automatically.

Stop it with:

```bash
docker stop trailer-seg
```

### Using docker-compose

Edit the volume paths in `docker-compose.yml`, then:

```bash
docker compose up -d        # start in background
docker compose logs -f      # follow logs
docker compose down         # stop and remove
```

---

## Output

For each input image `photo.jpg`, a file `photo.psd` is written to the output folder. Open in Photoshop or GIMP. Each mask layer is a black-and-white alpha mask — toggle visibility per layer to inspect segmentations.

Layer order (top to bottom in the PSD):
1. shadow
2. fender
3. jack
4. trailer floor
5. trailer frame
6. ramp
7. gate
8. coupler
9. trailer tongue
10. hitch
11. axle
12. tire
13. wheel
14. cable
15. safety chain
16. chain
17. trailer *(full silhouette, fine-tuned model)*
18. Input Image

---

## Troubleshooting

**No GPU available / running on a CPU-only machine**

Add `--device cpu` to the run command (no `--gpus` flag needed):

```bash
docker run --rm \
  -v /path/to/your/images:/input \
  -v /path/to/output:/output \
  trailer-seg --device cpu
```

Warning: CPU mode works but is very slow (~10 minutes per image).

**Image already processed but you want to re-run it**

Delete the corresponding `.psd` from the output folder — the container will pick it up on the next poll cycle.

**Check what the container is doing**

```bash
docker logs -f trailer-seg
```
