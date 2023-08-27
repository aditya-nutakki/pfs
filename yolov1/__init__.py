"""
https://arxiv.org/pdf/1506.02640.pdf

Abstract:
We present YOLO, a new approach to object detection.
Prior work on object detection repurposes classifiers to per-
form detection. Instead, we frame object detection as a re-
gression problem to spatially separated bounding boxes and
associated class probabilities. A single neural network pre-
dicts bounding boxes and class probabilities directly from
full images in one evaluation. Since the whole detection
pipeline is a single network, it can be optimized end-to-end
directly on detection performance.

Our unified architecture is extremely fast. Our base
YOLO model processes images in real-time at 45 frames
per second. A smaller version of the network, Fast YOLO,
processes an astounding 155 frames per second while
still achieving double the mAP of other real-time detec-
tors. Compared to state-of-the-art detection systems, YOLO
makes more localization errors but is less likely to predict
false positives on background. Finally, YOLO learns very
general representations of objects. It outperforms other de-
tection methods, including DPM and R-CNN, when gener-
alizing from natural images to other domains like artwork.

"""