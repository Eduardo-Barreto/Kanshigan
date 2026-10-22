= Materials and Methods <sec-method>

== Study design

This is applied, quantitative, experimental research. We propose no new detector or tracker: we compare existing components on real tournament video, against a human-reviewed reference set. The research question splits into two manipulated factors. The first is the detector architecture: YOLOv8s @jocher2023yolo (11.1M parameters, 28.4 GFLOPs) versus the compact YOLO26n @jocher2026yolo26 (2.4M, 5.2 GFLOPs), trained with identical split, seed and protocol. The second is the tracking algorithm: four trackers applied to the same detections, so that measured differences belong to the tracker, not the detector. The dependent variables are accuracy and runtime cost, because the question seeks the balance between them.

The experiments form six points (@tab-experiments). Runtime cost is measured on the E2 and E4 configurations.

#figure(
  caption: [Experimental points and what each isolates.],
  table(
    columns: (auto, 1fr, 1fr),
    align: (center, left, left),
    stroke: 0.4pt,
    table.header([*ID*], [*Configuration*], [*Question*]),
    [E1], [SAM 3 against the reference set], [Does the annotator agree with human review?],
    [E2], [YOLOv8s and YOLO26n fine-tuned on the domain], [How much does detector capacity matter?],
    [E3], [YOLOv8s with COCO weights, no fine-tuning], [Does the domain require specific training?],
    [E4], [Four trackers × two detectors, fixed detections], [Does appearance improve identity? At what cost?],
    [E5], [YOLOv8s with and without 14 extra JP rounds, three seeds], [Does more annotation improve the detector?],
    [E6], [Label-free evaluation and arena ablation], [Where does the pipeline fail outside the reference set?],
  ),
) <tab-experiments>

== Pipeline

The pipeline follows tracking-by-detection: a detector locates each robot with a bounding box in every frame, and a tracker links the boxes over time (@fig-pipeline). Inference runs on every frame, at the native frame rate, because decisive events span few frames. Each stage writes an inspectable output, which lets us evaluate components separately.

#figure(
  placement: top,
  scope: "parent",
  image("/docs/paper/figures/pipeline.svg", width: 100%),
  caption: [Pipeline stages. The first row prepares the frame; the second extracts information. The dashed branch is offline annotation with SAM 3, which builds the training set and stays out of inference.],
) <fig-pipeline>

*Arena and calibration.* The _dohyo_ geometry is fixed by the rules: a dark 154 cm disk with a 5 cm white border @fujisoft2024rules. The arena is therefore detected with classical vision, without a trained model. We threshold luminance to isolate the white border, clean the result with morphology and fit ellipses to the contours. A score combining size, centrality and aspect ratio selects the ellipse; in frames where fitting fails, we reuse the last valid ellipse. A circle seen in oblique perspective projects to an ellipse whose major axis corresponds to the real diameter. The ratio between the real diameter and the major axis in pixels sets the centimeter-per-pixel scale of each frame, without camera calibration or markers. The isotropic scale ignores perspective foreshortening, a limitation E6 quantifies.

*Crop and detection.* Robots occupy a small fraction of the frame. Downscaled to the detector's 640 px input, the full frame leaves the robots with few pixels, which motion blur erases, and the background (audience, operators' hands) produces false positives. We therefore crop a fixed rectangle per clip, with a 20% margin around the median ellipse of 12 sampled frames. On the Brazilian reference round, the crop discards 62% of the frame area and enlarges the robots about 1.2 times at the detector input; boxes map back to native coordinates by an offset. The YOLO detector runs on the crop. We discard detections whose center falls outside the arena ellipse enlarged by 15% and keep at most two robots per frame.

*Tracking and identity.* We compare two motion-only and two appearance-based trackers, implemented in the BoxMOT library. OC-SORT @cao2023ocsort corrects the Kalman filter with recent observations, which favors non-linear trajectories (C1). ByteTrack @zhang2022bytetrack also associates low-confidence detections, common under blur. DeepOCSORT @maggiolino2023deepocsort and BoT-SORT @aharon2022botsort add an appearance re-identification (ReID) descriptor, OSNet @zhou2019osnet trained on pedestrians, without domain adaptation. These two test whether appearance adds information when targets are nearly identical (C2). Trackers emit arbitrary identifiers. We call robot A the leftmost robot in the first frame where both appear. We keep only the two longest trajectories, which leaves gaps when a trajectory fragments.

*Kinematics and events.* The pipeline projects each robot onto the _dohyo_ coordinate system, in centimeters and relative to the arena center. Because the arena is fixed in the world, relative position cancels handheld camera translation. A Savitzky-Golay filter @savitzky1964smoothing smooths the trajectory before velocity and acceleration are derived, because direct finite differences would amplify detection noise. Deterministic rules mark three events. Round start is the first frame where either robot exceeds 15 cm/s for three consecutive frames, skipping the first two frames, where the filter has a boundary artifact. First contact requires, in the same frame, a distance below 30 cm and a speed change above 40 cm/s. Ring-out occurs when a robot's center crosses the arena radius. Thresholds are initial values fixed a priori, not calibrated. This paper validates neither kinematics nor events; @sec-discussion discusses what that validation requires.

== Data and annotation

No public dataset exists for the domain. We built ours from two sources of tournament video with opposite camera regimes (C4). The Brazilian source (BR) gathers IRONCup 2025 matches filmed with a handheld camera, at an oblique angle, at 60 frames per second. The Japanese source (JP) is a 412 s video of a regional tournament, filmed by a fixed overhead camera. We segment videos into rounds, one clip each, and split training, validation and test by round, not by frame: neighboring frames are nearly identical, and a frame split would leak information. The test set is therefore held out by round, not by tournament. The JP test round comes from the same video as JP training, and the BR one from the same tournament as BR training.

Annotating boxes by hand would take hours per round. We use SAM 3 @carion2025sam3 as a semi-automatic annotator, following the practice of using a large open-vocabulary model as the annotator and a compact detector, trained on its labels, for inference @liu2024groundingdino @ravi2024sam2. SAM 3 propagates masks through video from a text concept; we use the concept "toy", chosen in an initial test of ten concepts. We convert each mask to a box through its largest connected component and scale boxes to native resolution. The annotator receives the arena crop downscaled to 960 px width at 10 frames per second. To fit in 8 GB of GPU memory, it processes 60-frame windows and resets the prompt in each one. The JP robots, small and black in the overhead view, scored low for the concept and fell under the default threshold. Lowering the threshold to 0.15 and discarding boxes outside the arena recovered both robots. Training labels were not hand-reviewed; we kept only frames with both robots labeled. The JP training labels were generated before a fix to the box filter that, in rounds annotated later, swapped a robot for a tripod outside the arena; we did not measure that effect on the original training set.

The reference set has one round per source, held out from training. SAM 3 proposed the boxes, and we reviewed them by hand in every frame sampled at 10 Hz: 59 frames with 112 boxes for BR and 57 frames with 82 boxes for JP. Because review starts from the SAM 3 proposal, the agreement measured in E1 is an upper bound. On the BR round, an identity reference marks which robot is A and which is B in the same 59 frames (118 boxes, including 6 that fall outside the crop), which allows tracking evaluation. @tab-dataset summarizes the composition.

#figure(
  caption: [Dataset composition in annotated frames (sampled at 10 Hz). Rounds counts BR and JP together. Training and validation include only frames with both robots labeled, so the detector does not learn to ignore one; the reference set includes every sampled frame of the round.],
  table(
    columns: 5,
    align: (left, center, center, center, left),
    stroke: 0.4pt,
    table.header([*Subset*], [*Rounds*], [*BR*], [*JP*], [*Labels*]),
    [Train], [14], [423], [202], [SAM 3],
    [Validation], [2], [59], [15], [SAM 3],
    [Reference (test)], [2], [59], [57], [SAM 3 + review],
  ),
) <tab-dataset>

== Training and evaluation

Each detector starts from the COCO weights @lin2014coco of its architecture and trains with the same split, 640 px input, default Ultralytics 8.4.55 augmentation, batch 16, up to 100 epochs with early stopping after 20 epochs without validation improvement, and seed 42. We froze the reported detectors before the final evaluation. The arena crop, however, we chose during development, after observing detector failures on full frames.

For detection, we measure mean average precision (mAP) at an IoU threshold of 0.5 and averaged over 0.5 to 0.95, plus precision and recall at the maximum-F1 point, as Ultralytics reports them. For tracking, we measure MOTA @bernardin2008clear, which sums false positives, false negatives and identity switches; IDF1 @ristani2016idf1, which measures how long each robot keeps the right identity; and HOTA @luiten2021hota, which balances detection and association. For cost, we measure frames per second (FPS) and peak GPU memory on a laptop with an 8 GB RTX 4070 Laptop GPU. Every number is reported per source and always measured, never projected; unless stated otherwise, each comes from a file under `results/`.

Hyperparameters, versions (PyTorch 2.12, Ultralytics 8.4.55, BoxMOT 19.0.0) and scripts are versioned in the repository. The protocol for E1 to E4 was registered in a project log before execution; E5 and E6 came later, in response to the qualifying committee. The log also documents deviations, failures and fixes.
