"""End-to-end exterior car segmentation: raw photo in, four masks out.

    seg = ExteriorSegmenter(WEIGHTS)
    out = seg.predict(rgb)          # rgb: HxWx3 uint8
    out["outline"]                  # bool HxW -- the cutout silhouette (antenna included)
    out["punchout"]                 # bool HxW -- structural see-through (wheel gaps, rails, under-body)
    out["windows"]                  # bool HxW -- tinted / see-through glass
    out["antenna"]                  # bool HxW -- what the antenna model added on top of the body
    out["car_box"], out["bucket"]   # (x0,y0,x1,y1) and the aspect bucket that was used

STAGES
  1. RF-DETR car+antenna detector on the RAW frame -> union(best car box, antenna boxes).
     Antennas are unioned in so the crop is not clipped through a roof mast.
     No detection -> fall back to the whole frame (close-framed vehicles that fill the photo are
     exactly what the car-extent detector never saw in training; 17/490 on the QM set).
  2. Route the box aspect to one of 5 buckets, pad 0.08, letterbox -> one Swin-L pass ->
     outline / tint / antenna-head logits.
  3. SAME detector re-run on the CAR CROP for antenna boxes (recall needs the crop) -> per box, the
     zoom UNet segments a tight 256 tile. Antennas are ~1-2 px in the full frame; only the zoom sees them.
  4. Assemble.

DELIBERATE OMISSIONS, each measured:
  * The BiRefNet antenna head (ch2) is NOT unioned into the outline. It is free and it beats the bare
    outline, but it plants a stray stick on 4% of images versus the zoom UNet's 0.4%. It is returned
    as `antenna_head` for reference only.
  * The 2-stage hole-UNet refiner is OFF. It buys +7pt small-hole recall for -4pt precision, and its
    "false positives" are mostly real see-through (roof rails, step bars) that the GT under-labels.
    Re-enable once the labels are cleaned.
  * The separate 1536 body pass is GONE. The bucketed pass captures small holes better (56% vs 48%)
    at ~40% less compute, because the car fills the frame instead of occupying ~24% of a square.
"""
import cv2
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes

from .buckets import BUCKETS, crop_pad_box, letterbox, normalize, route

CAR_CLS, ANTENNA_CLS = 0, 1


def keep_main(mask, close_px=9):
    """Keep only the largest connected component, after a close that bridges small gaps.

    Drops floating blobs (a stray antenna stub in mid-air) while keeping a mast that is separated
    from the roof by a few pixels of sky.
    """
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return mask
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        bridged = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    else:
        bridged = m
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bridged, 8)
    if n <= 1:
        return mask
    return mask & (lbl == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def _tight_crop(mask, context=1.5, min_pad_px=20):
    """(top, left, h, w) around a 0/255 mask, padded by `context` -- must match UNet training."""
    H, W = mask.shape[:2]
    rows = np.where((mask > 127).any(axis=1))[0]
    cols = np.where((mask > 127).any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    y1, y2, x1, x2 = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
    ph = max(int((y2 - y1 + 1) * (context - 1) / 2), min_pad_px)
    pw = max(int((x2 - x1 + 1) * (context - 1) / 2), min_pad_px)
    top, left = max(0, y1 - ph), max(0, x1 - pw)
    return top, left, min(H, y2 + 1 + ph) - top, min(W, x2 + 1 + pw) - left


class ExteriorSegmenter:
    def __init__(self, weights, device="cuda", birefnet_repo=None,
                 det_thr=0.5, car_thr=0.3, antenna_thr=0.5, seg_thr=0.5, close_px=9):
        """`weights` is a dict with keys: detector, outline, trihead, antenna_unet."""
        from .models import load_antenna_unet, load_detector, load_spliced
        self.device = device
        self.det = load_detector(weights["detector"])
        self.bn = load_spliced(weights["outline"], weights["trihead"], device, birefnet_repo)
        self.unet = load_antenna_unet(weights["antenna_unet"], device)
        self.det_thr, self.car_thr = det_thr, car_thr
        self.antenna_thr, self.seg_thr, self.close_px = antenna_thr, seg_thr, close_px

    # ---- stage 1 -------------------------------------------------------------------------------
    def car_box(self, pil_img, H, W):
        det = self.det.predict(pil_img, threshold=0.15)
        xy = np.asarray(det.xyxy)
        if len(xy):
            cls = np.asarray(det.class_id)
            conf = np.asarray(det.confidence)
            cars = [i for i in range(len(xy)) if cls[i] == CAR_CLS and conf[i] >= self.car_thr]
            if cars:
                boxes = [xy[cars[int(np.argmax(conf[cars]))]]]
                boxes += [xy[i] for i in range(len(xy)) if cls[i] == ANTENNA_CLS and conf[i] >= 0.15]
                a = np.array(boxes)
                x0, y0 = max(0, int(a[:, 0].min())), max(0, int(a[:, 1].min()))
                x1, y1 = min(W, int(a[:, 2].max())), min(H, int(a[:, 3].max()))
                if x1 - x0 >= 32 and y1 - y0 >= 32:
                    return (x0, y0, x1, y1), False
        return (0, 0, W, H), True                      # full-frame fallback

    # ---- stage 2 -------------------------------------------------------------------------------
    @torch.no_grad()
    def _birefnet(self, img, box, bucket):
        W, H = BUCKETS[bucket]
        Hh, Ww = img.shape[:2]
        cx0, cy0, cx1, cy1 = crop_pad_box(box, Hh, Ww)
        crop = img[cy0:cy1, cx0:cx1]
        ch, cw = crop.shape[:2]
        lb, (ox, oy, nw, nh) = letterbox(crop, W, H)
        t = torch.from_numpy(normalize(lb)[None]).to(self.device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outs = self.bn(t)
        res = []
        for o in outs:
            pr = o.sigmoid()[0, 0].float().cpu().numpy()[oy:oy + nh, ox:ox + nw]
            full = np.zeros((Hh, Ww), np.float32)
            full[cy0:cy1, cx0:cx1] = cv2.resize(pr, (cw, ch))
            res.append(full > 0.5)
        return res

    # ---- stage 3 -------------------------------------------------------------------------------
    @torch.no_grad()
    def _antenna(self, img, box_mask, size=256):
        """Segment one detector antenna box at zoom. Crop geometry must match UNet training."""
        p = _tight_crop(box_mask)
        H, W = box_mask.shape
        top, left, ch, cw = p if p else (0, 0, H, W)
        crop = img[top:top + ch, left:left + cw]
        side = max(ch, cw)
        xo, yo = (side - cw) // 2, (side - ch) // 2
        sq = np.zeros((side, side, 3), np.uint8)
        sq[yo:yo + ch, xo:xo + cw] = crop
        t = torch.from_numpy(normalize(cv2.resize(sq, (size, size)))[None]).to(self.device)
        pr = torch.sigmoid(self.unet(t)[0, 0]).cpu().numpy()
        pr = cv2.resize(pr, (side, side))[yo:yo + ch, xo:xo + cw]
        full = np.zeros((H, W), np.float32)
        full[top:top + ch, left:left + cw] = pr
        return full > self.antenna_thr

    # ---- all together --------------------------------------------------------------------------
    def predict(self, rgb):
        from PIL import Image
        img = np.ascontiguousarray(rgb)
        H, W = img.shape[:2]
        (x0, y0, x1, y1), fellback = self.car_box(Image.fromarray(img), H, W)

        body, tint, head = self._birefnet(img, (x0, y0, x1, y1), route((x1 - x0) / (y1 - y0)))

        antenna = np.zeros((H, W), bool)
        dd = self.det.predict(Image.fromarray(img[y0:y1, x0:x1]), threshold=self.det_thr)
        cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else np.zeros(len(dd.xyxy))
        for i, b in enumerate(np.asarray(dd.xyxy)):
            if int(cls[i]) != ANTENNA_CLS:
                continue
            bx0, by0, bx1, by1 = [int(v) for v in b]
            bm = np.zeros((H, W), np.uint8)
            bm[max(0, y0 + by0):min(H, y0 + by1 + 1), max(0, x0 + bx0):min(W, x0 + bx1 + 1)] = 255
            antenna |= self._antenna(img, bm)

        body_only = keep_main(body, self.close_px)
        outline = keep_main(body | antenna, self.close_px)
        return {
            "outline": outline,
            "punchout": binary_fill_holes(outline) & ~outline,
            "windows": tint,
            "antenna": antenna & ~body_only & outline,   # only what the antenna model ADDED
            "antenna_head": head,                        # reference only, not in the outline
            "car_box": (x0, y0, x1, y1),
            "bucket": route((x1 - x0) / (y1 - y0)),
            "full_frame_fallback": fellback,
        }

    @staticmethod
    def cutout_alpha(out):
        """Alpha for an RGBA cutout: silhouette minus see-through gaps."""
        return out["outline"] & ~out["punchout"]
