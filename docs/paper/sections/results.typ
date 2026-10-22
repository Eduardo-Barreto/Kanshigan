= Results <sec-results>

The reference set is small: one round per source, and a single round with identities. We therefore treat small differences as noise and state, in each case, what the sample supports.

== Annotator validation (E1)

SAM 3 annotates the training set, so it must agree with what human review accepts. We compare its boxes with the reference boxes at IoU ≥ 0.5 (@tab-annotator). On the BR round, image mode with the concept "robot" reaches 0.96 F1, with 0.91 mean IoU on matched boxes; video mode with the concept "toy", which annotated BR training, was not measured on that round. On JP, in the overhead view, image mode barely finds the small black robots (0.16 recall). Video mode, which propagates masks through time, recovers both robots: 0.94 recall and 0.91 F1 on the JP reference round.

The 14 extra JP rounds of E5 give a larger measure of video mode. Over 448 frames and 878 reviewed boxes, it reaches 0.75 F1. The boxes it proposes nearly coincide with human ones (0.94 IoU), but about three in ten robots get no box, and review changed 71% of the frames. Review was not a finishing touch: it is what made those rounds usable. The labels of the reported training set did not go through this review, and their quality remains unknown.

#figure(
  caption: [SAM 3 agreement with human review (IoU ≥ 0.5). P: precision; R: recall; IoU: mean IoU of matched boxes. Every review started from the SAM 3 proposal, so values are upper bounds. BR reference: image mode with the concept "robot"; the other rows use "toy" with a 0.15 threshold. Video mode annotated the training set of both sources. The 14 rounds were annotated after a fix that ranks boxes by distance to the arena center, and do not measure the original JP labels.],
  table(
    columns: 6,
    align: (left, left, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Set*], [*Mode*], [*P*], [*R*], [*F1*], [*IoU*]),
    [BR ref.], [image], [0.98], [0.94], [0.96], [0.91],
    [JP ref.], [image], [1.00], [0.16], [0.27], [0.92],
    [JP ref.], [video], [0.88], [0.94], [0.91], [0.98],
    [14 JP rounds], [video], [0.80], [0.71], [0.75], [0.94],
  ),
) <tab-annotator>

== Detection (E2 and E3) <sec-detection>

Both fine-tuned detectors exceed 0.96 mAP\@0.5 on both sources (@tab-detector). YOLO26n, with a fifth of the parameters, reaches 0.968 versus 0.965 on BR and 0.993 versus 0.976 on JP. For scale, YOLOv8s trained with three seeds varies by 0.019 (standard deviation) on JP (@sec-scaling). The gap between architectures is of the same order. The data therefore support practical equivalence, not superiority: the extra capacity of YOLOv8s buys no measurable accuracy in this domain. YOLO26n was trained with a single seed.

Domain fine-tuning, in contrast, is decisive. We evaluate YOLOv8s with COCO weights (E3), accepting any COCO class as a robot: it reaches 0.757 on BR and 0.359 on JP. Under the default single-class evaluation, which compares only COCO class 0 (person), it scores 0.026 and 0.062. The generic detector detects some robots as some COCO class, but fails precisely on the overhead view. Domain training lifts both cases above 0.96.

#figure(
  placement: top,
  caption: [Detection on each source's reference round, on the arena crop. Precision and recall at the maximum-F1 confidence, as Ultralytics reports them. COCO: YOLOv8s without fine-tuning, any class accepted as a robot.],
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Detector*], [*Src.*], [*mAP\@.5*], [*mAP\@.5:.95*], [*P*], [*R*]),
    [YOLOv8s], [BR], [0.965], [0.767], [0.990], [0.964],
    [YOLOv8s], [JP], [0.976], [0.695], [0.991], [0.915],
    [YOLO26n], [BR], [0.968], [0.772], [0.963], [0.955],
    [YOLO26n], [JP], [0.993], [0.691], [0.986], [0.976],
    [COCO], [BR], [0.757], [0.570], [0.764], [0.693],
    [COCO], [JP], [0.359], [0.252], [0.398], [0.463],
  ),
) <tab-detector>

The high precision in @tab-detector depends on the threshold. At a fixed 0.25 confidence, the one that feeds the tracker, YOLOv8s produces 30 false positives in the 57 JP frames (0.72 precision). Seventeen of them fall outside the arena and are almost all the robot already pushed out, which the JP training labels include and the JP reference does not. The arena filter the pipeline applies removes those boxes and raises precision to 0.86; 13 false positives remain inside the arena. For YOLO26n, the filter raises precision from 0.86 to 0.98; on BR, it changes nothing. The part outside the arena comes from the JP annotation convention, not from the detector.

== Tracking (E4)

We apply the four trackers to the same detections from each detector, on the BR round with identities (@tab-tracking). Metrics use only the 59 annotated frames#footnote[Six of the 118 identity boxes fall outside the crop, because the identity reference predates a crop fix; no tracker can match them. Scoring only the 112 visible boxes, OC-SORT on YOLOv8s rises from 0.881 to 0.929 MOTA and from 0.810 to 0.853 HOTA, without changing the tracker ranking. On raw trajectories, before the A/B convention, three combinations reach two switches and the IDF1 of OC-SORT on YOLOv8s drops from 0.933 to 0.925; the ranking does not change.].

#figure(
  placement: bottom,
  scope: "parent",
  caption: [Detector × tracker grid on the BR round with identities (59 frames at 10 Hz, 118 boxes). IDSW: identity switches. FPS: full pipeline on the crop, on GPU, including decoding; mean of three runs, standard deviation at most 3.4 FPS.],
  table(
    columns: 8,
    align: (left, left, left, center, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Detector*], [*Tracker*], [*Type*], [*MOTA*], [*IDF1*], [*HOTA*], [*IDSW*], [*FPS*]),
    [YOLOv8s], [OC-SORT], [motion], [0.881], [0.933], [0.810], [1], [155],
    [YOLOv8s], [ByteTrack], [motion], [0.898], [0.947], [0.824], [0], [151],
    [YOLOv8s], [DeepOCSORT], [appearance], [0.881], [0.933], [0.810], [1], [57],
    [YOLOv8s], [BoT-SORT], [appearance], [0.907], [0.952], [0.833], [0], [61],
    [YOLO26n], [OC-SORT], [motion], [0.881], [0.933], [0.796], [1], [128],
    [YOLO26n], [ByteTrack], [motion], [0.890], [0.938], [0.804], [1], [130],
    [YOLO26n], [DeepOCSORT], [appearance], [0.856], [0.919], [0.774], [1], [54],
    [YOLO26n], [BoT-SORT], [appearance], [0.890], [0.938], [0.804], [1], [57],
  ),
) <tab-tracking>

All eight combinations sustain identity, with at most one switch in 118 boxes. Accuracy differences fit within one or two boxes: BoT-SORT on YOLOv8s leads by a margin a single round cannot separate from noise. The cost difference, in contrast, is large and does not depend on sample size. The ReID step drops the whole pipeline from 128–155 to 54–61 FPS. Measured without the detector, the tracking step is 31 to 35 times slower with ReID. Motion-only trackers deliver the same identity at a fraction of the cost.

*Why does appearance not help?* We expected the answer to be C2: robots too similar for ReID to separate. We measured this premise and it does not hold for this round. We extracted descriptors from 111 crops of the round (the visible ones with a side of at least 12 px) and compared cosine similarity between crops of the same robot and of different robots. With the OSNet the trackers use, mean intra-robot similarity is 0.85 and inter-robot 0.54, with an area under the ROC curve (AUC) of 0.94; the nearest neighbor recovers identity for 97% of crops. Even a color histogram separates the two robots (0.92 AUC). The explanation the data support is different: motion alone already solves this round, and no identity error is left for appearance to fix. On YOLOv8s, DeepOCSORT produces trajectories nearly identical to OC-SORT's (697 of 698 lines equal): ReID almost never changes an association. This round therefore does not stress C2, and a match between truly indistinguishable robots remains unmeasured.

== Runtime cost <sec-cost>

The full pipeline, from decoding to events, runs in real time on native video on a laptop GPU (@tab-viability). On 1080p at 60 FPS, it processes 97 FPS with YOLOv8s and 89 with YOLO26n, a gap that individual runs do not separate. On the crop, YOLOv8s is faster (155 versus 128 FPS) despite having almost five times the parameters: on GPU, network size is not the dominant cost. On the same laptop's CPU, without GPU, the order flips: YOLO26n runs at 40 FPS and YOLOv8s at 21. Post-match analysis remains feasible on CPU, below real time. On 4K HEVC (H.265) video, decoding becomes the bottleneck and the pipeline drops to 16 FPS. An isolated process takes 250 to 272 MiB of GPU memory, CUDA context included, under 4% of the 8 GB card; with ReID, up to 334 MiB. During annotation, the project log recorded SAM 3 at about 2 FPS and about 7 GB, which justifies keeping it out of inference.

#figure(
  placement: top,
  caption: [Full pipeline cost (decoding, arena, crop, detection, OC-SORT, kinematics and events) on the BR reference round; speed in FPS, mean and standard deviation of three warm runs; the 4K row uses a radio-controlled (RC) Sumo round. Laptop with Intel Core Ultra 9 185H and RTX 4070 Laptop GPU (8 GB). Memory: NVML peak on the first run of a single-model process, CUDA context included.],
  table(
    columns: 4,
    align: (left, left, center, center),
    stroke: 0.4pt,
    table.header([*Input*], [*Device*], [*YOLOv8s*], [*YOLO26n*]),
    [1080p native], [GPU], [97.0 ± 7.0], [88.9 ± 2.2],
    [1080p native], [CPU], [21.3 ± 1.7], [39.6 ± 2.3],
    [Crop 1558×506], [GPU], [154.5 ± 2.2], [127.7 ± 1.2],
    [4K HEVC (RC)], [GPU], [16.1 ± 2.6], [16.5 ± 1.4],
    [GPU memory], [GPU], [250 MiB], [272 MiB],
    [Weights], [disk], [22.5 MB], [5.4 MB],
  ),
) <tab-viability>

== JP training scale (E5) <sec-scaling>

JP training uses 8 of the 26 rounds segmentation finds in the source video. To measure whether annotation volume limits the detector, we annotated with SAM 3 and hand-reviewed another 14 rounds of the same video (448 frames). JP training grew from 202 to 650 frames. We retrained YOLOv8s with the same hyperparameters and three seeds, without changing the reference sets (@tab-scaling).

#figure(
  placement: bottom,
  scope: "parent",
  caption: [YOLOv8s without and with the 14 extra JP rounds, seeds 42, 0 and 1. mAP as mean ± standard deviation. Counts per seed, at minimum confidence 0.25 and IoU ≥ 0.5 (112 robots on BR, 82 on JP). Missed: false negatives. FP in and FP out: false positives inside and outside the arena, before the pipeline filter.],
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Evaluated / training*], [*mAP\@.5*], [*mAP\@.5:.95*], [*Missed*], [*FP in*], [*FP out*]),
    [BR / original], [0.968 ± 0.006], [0.786 ± 0.018], [4, 4, 4], [1, 28, 1], [0, 0, 0],
    [BR / + 14 JP], [0.957 ± 0.032], [0.728 ± 0.047], [3, 4, 3], [1, 70, 1], [0, 1, 1],
    [JP / original], [0.963 ± 0.019], [0.669 ± 0.023], [5, 5, 10], [13, 1, 0], [17, 20, 9],
    [JP / + 14 JP], [0.978 ± 0.009], [0.717 ± 0.020], [0, 2, 1], [0, 21, 1], [54, 37, 47],
  ),
) <tab-scaling>

mAP\@0.5 does not separate the configurations: intervals overlap on both sources. What changes is the type of error. On the JP reference, the detector with more data misses fewer robots across all three seeds (from 5–10 to 0–2, out of 82), including blurred robots and robots at the frame edge, and fits boxes better (mAP\@0.5:0.95 of 0.703 to 0.740, versus 0.652 to 0.695). The gain has a cost on the other source. On the BR reference, mAP\@0.5:0.95 drops across all three seeds, and one seed splits a robot into partial boxes. False positives inside the arena vary more across seeds than across configurations: with seed 0, a robot splits into two or three partial boxes in both configurations. The new frames flip the training balance (423 BR to 650 JP); the imbalance is the likely explanation, but we did not test it in isolation. Because the new rounds were also hand-reviewed and include 18 frames with a single robot, which the original training excluded, the experiment does not separate the effect of more data from that of reviewed data. Detections outside the arena triple on JP and are, in the inspected frames, the robot already pushed out, the same convention mismatch as in @sec-detection. Expanding annotation of a single source therefore had a cost: it improved that source and hurt the other. We keep the detector trained on the original set as the reported one.

== Label-free evaluation (E6) <sec-label-free>

Outside the two reference rounds there are no labels. The domain, however, imposes a constraint on every correct output: during the round, there are exactly two robots on the _dohyo_. We count the fraction of frames violating this constraint, between the round start and end the pipeline itself marks, and anchor the indicator on the BR round, where accuracy is known (0.93 IDF1 and 6.3% violation). The indicator counts robots; it does not check whether boxes are correct. On the evaluated clips, it matches the qualitative assessment recorded before the measurement: 44% on a low-resolution BR video and 100% on a stretch of the All Japan Robot-Sumo Tournament 2025 grand final. Two clips from earlier runs, whose source video we could not recover, score 0% (a fixed-camera JP round) and 54% (another stretch of the final).

The same pipeline, without retraining, ran on the 23 radio-controlled (RC) Sumo rounds of the ThundeRatz team archive, a category outside training, with differently shaped robots and phone video. Median violation is 27%. Ten of the 23 rounds fall at or below the reference round's rate, three of them with short windows, under 60 frames, where the rate is unstable; seven exceed 50%. This suggests that the detector transfers to another category in part of the rounds; it is not an accuracy measurement.

In the Japanese grand final, an official broadcast with a blue arena, a score overlay and violent collisions, the failure mode is visible (@fig-worlds). During the slow approach at the start, the detector finds both robots, and the arena of a new color is detected. In fast motion, extreme blur prevents both detections, and the second half of the round has almost no detections. SAM 3, on the crop and with temporal propagation, still holds one robot. The limit is not robot similarity, since those robots carry distinct flags, but the lack of training data with that blur.

#figure(
  placement: top,
  scope: "parent",
  image("/docs/paper/figures/worlds_model_vs_sam_en.png", width: 80%),
  caption: [Our detector (left, native frame) and SAM 3 (right, arena crop) at the same instants of the first round of the All Japan Robot-Sumo Tournament 2025 grand final, outside the training distribution; a different clip from the two audited stretches. Top: slow movement, both find both robots. Bottom: fast movement, the detector loses both; SAM 3 keeps one.],
) <fig-worlds>

The arena stage was also measured without labels. Because the _dohyo_ is fixed and has a known diameter, a correct fit yields a stable scale over the round. We compare the score-based selector with picking the largest white contour, on the same frames of 30 clips, counting as a tie a difference below 0.01 in the coefficient of variation of the scale. The two tie on 17, the score wins on 9 and loses on 4. When it wins, the gap can be two orders of magnitude: on a handheld BR video, the coefficient of variation goes from 0.44 to 0.002. The score is indifferent with a single bright candidate and decisive when the background offers a competitor, as with handheld cameras. The same measure shows how much the scale drifts. The coefficient of variation is 0.001 on the fixed JP camera, 0.002 to 0.064 on the three BR videos and 0.18 median on the 24 RC clips (the 23 rounds and a demo clip), up to 0.38. The drift mixes fitting error with zoom and real camera motion; in-frame foreshortening error was not measured. Under phone cameras, the centimeter scale is not reliable without rectification, and every speed inherits that error.
