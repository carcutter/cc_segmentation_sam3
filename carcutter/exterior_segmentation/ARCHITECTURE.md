# Exterior car segmentation — architecture and deployment spec

Reference implementation: `carcutter/exterior_segmentation/`. Verified bit-identical to the masks
delivered for the 2026-08 quality-manager evaluation (490 images).

Replaces production's `a_rank_exterior_segmentation_model` (+ the holes / window models).

---

## 1. Pipeline

```
raw photo
   |
   +-- RF-DETR Medium (2 classes: car, antenna) on the FULL frame
   |      car box = union(best car, antenna boxes)        <- antennas unioned so a roof mast is not clipped
   |      no detection -> whole frame                     <- close-framed vehicles; 17/490 on the QM set
   |
   +-- route(box aspect) -> 1 of 5 buckets -> pad 0.08 -> letterbox
   |      SPLICED BiRefNet, ONE Swin-L encoder pass, two decoders
   |        ch0 outline (body silhouette)
   |        ch1 tint    (windows / see-through glass)
   |        ch2 antenna (head; reference only, NOT used -- see 5)
   |
   +-- SAME detector re-run on the CAR CROP -> antenna boxes
   |      per box: tight crop -> EfficientNet-B4 UNet @256 -> antenna mask
   |      (antennas are 1-2 px in the full frame; only the zoom sees them)
   |
   +-- assemble
          body     = keep_main(ch0, close=9)
          outline  = keep_main(ch0 | antenna, close=9)
          punchout = binary_fill_holes(outline) & ~outline
          windows  = ch1
          antenna  = antenna & ~body & outline        <- only what the antenna model ADDED
          cutout alpha = outline & ~punchout
```

`keep_main` keeps the largest connected component after a 9 px close: it drops a floating stub while
keeping a mast separated from the roof by a few pixels of sky.

## 2. Aspect buckets

A square 1024 input spends ~76% of its pixels on background and letterbox, because car-crop aspect is
bimodal (side ~2.53, straight-on ~1.20). Five shapes, all `/32`, all ~1.0 Mpx so compute is constant:

| bucket | W x H | centre aspect | typical view |
|---|---|---|---|
| P | 960 x 1056 | 0.91 | portrait / interior-ish |
| A | 1088 x 896 | 1.21 | straight front / rear |
| B | 1248 x 800 | 1.56 | corner, 3/4 |
| C | 1376 x 704 | 1.95 | side |
| D | 1600 x 640 | 2.50 | long side / wide |

Route = nearest centre in **log** aspect, on the **detector** box. Then pad 0.08 per side, letterbox.

**This is the main deployment decision.** Production today is a single fixed 3x1024x1280 input.
Five shapes need either 5 TensorRT optimisation profiles behind one engine, or 5 model versions plus
a server-side router in a Triton ensemble. Nothing else in the registry does this yet.

## 3. Weights

| role | file | size |
|---|---|---|
| car + antenna detector | `rfdetr_ca_unified_v1/checkpoint_best_ema.pth` | 127 MB |
| outline decoder + shared encoder | `birefnet_aspect_prod_bucket5.pt` | 844 MB |
| tint + antenna decoder | `birefnet_trihead_bucket.pt` | 844 MB |
| antenna zoom UNet | `unet_antenna_v3/checkpoints/best.pt` | 237 MB |

`birefnet_spliced_3head.pt` (940 MB) merges the two BiRefNet checkpoints into one artifact; the
reference implementation loads the two separately because that is the path that was validated.

**The BiRefNet is a fork.** `config.py` and `models/birefnet.py` are patched against
`ZhengPeng7/BiRefNet` (added `BIREFNET_NHEADS` so the tri-head decoder emits 3 channels, plus
`BIREFNET_SIZE` / `BIREFNET_BS`). Upstream will not load these checkpoints. Set `BIREFNET_REPO`.
Exporting to ONNX removes this dependency -- a good reason to hand Triton ONNX rather than `.pt`.

## 4. Thresholds

| knob | value | note |
|---|---|---|
| car confidence | 0.30 | detector `predict` runs at 0.15, cars filtered at 0.30 |
| antenna box confidence | 0.50 | the effective FP lever; 0.25 raises recall at ~2x the false fires |
| BiRefNet mask | 0.50 | all three channels |
| antenna UNet mask | 0.50 | |
| `keep_main` close | 9 px | |
| crop pad | 0.08 (body) / context 1.5 (antenna zoom) | the antenna crop MUST match training geometry |

## 5. Deliberate omissions — each one measured, do not "fix" without re-measuring

**The BiRefNet antenna head is not used.** It is free (same pass) and beats the bare outline, but it
plants a stray stick on 4% of images versus the zoom UNet's 0.4%, and 27% of its added components
float free versus the UNet's 5%. Returned as `antenna_head` for reference only.

**The 2-stage hole-UNet refiner is off.** +7pt small-hole recall for -4pt precision, and its "false
positives" are mostly genuine see-through (roof rails, step bars, bumper slats) that the ground truth
under-labels. Re-enable after a relabel pass, when the metric will credit it.

**The separate 1536 body pass is gone.** The bucketed pass captures small holes *better* (56% vs 48%)
at ~40% less compute, because the car fills the frame instead of ~24% of a square.

**Windows are ours but production may keep its own.** Our tint IoU 0.684 is competitive with prod's
~0.70; genuine losses are 6/862 rows (0.7%). Not a clear win, so it was deferred.

## 6. Measured performance (930-image held-out test, production framing, real detector boxes)

| metric | ours | production |
|---|---|---|
| outline BF@1 | **0.822** | 0.697 |
| outline IoU | 0.993 | 0.987 |
| punchout recall (small / med / large) | **62 / 94 / 94 %** | 42 / 70 / 71 % |
| punchout precision | 78% | 87% |
| antenna coverage | 0.678 | none (no antenna class) |

Outline wins at every tolerance and on every view; side views gain most (+0.17 BF@1). The
catastrophic-drop tail is essentially gone: 0 images lose >5% of the car, versus 17/930 in prod.

Antenna A/B (507 antenna images), coverage / thin-mast tip / stray-stick rate:

| arm | coverage | mast tip | stray sticks |
|---|---|---|---|
| outline only | 0.758 | 47% | 0% |
| + BiRefNet head (free) | 0.802 | 65% | 4% |
| **+ zoom UNet (shipped)** | **0.829** | **71%** | **0.4%** |

Antenna is proposer-limited: given a box, the UNet traces ~95% of the mast, but the detector only
proposes ~85% of antennas. Improve recall there, not in the segmenter.

## 7. Compute

Reference profile on an L40S, before the bucketing work removed a pass: ~1058 ms/img total, of which
3 Swin-L passes were 699 ms. The current design is 1 bucketed Swin-L pass + 2 RF-DETR passes + a few
tiny UNet crops. Remaining levers, in order of value: merge the two RF-DETR passes into one; distil
Swin-L to a lighter student; TensorRT FP16/INT8; batch the per-box UNet crops into one forward.

## 8. Open items

- ONNX export of all three networks (removes the BiRefNet fork dependency).
- The 5-profile TensorRT / ensemble-router question in section 2.
- Detector recall on antennas (85%) is the antenna ceiling.
- Detector misses on vehicles that fill the frame -- currently handled by the full-frame fallback.
- Hole refiner and window model both wait on a GT relabel pass for see-through.
