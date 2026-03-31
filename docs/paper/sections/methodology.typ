= Methodology <sec:methods>

Our pipeline consists of five sequential stages: dohyo detection, robot detection, multi-object tracking, metric extraction, and event detection.

// TODO: Adicionar figura do pipeline
// #figure(
//   image("figures/pipeline.png", width: 100%),
//   caption: [Overview of the Kanshigan pipeline.],
//   placement: top,
// ) <fig:pipeline>

== Dohyo Detection

The dohyo is a circular arena that appears as an ellipse under perspective projection. We detect the dohyo boundary by exploiting the high contrast between its black surface and white border (tawara). The detected ellipse parameters provide a region of interest (ROI) for subsequent stages and enable spatial calibration from pixel coordinates to real-world centimeters, given the known dohyo diameter of 154cm for the 3kg class @fujisoft-rules.

== Robot Detection and Tracking

We implement and compare three detection-tracking approaches:

#figure(
  caption: [Detection and tracking approaches compared in this study.],
  placement: top,
  table(
    columns: (auto, auto, auto, auto),
    align: (center, center, center, left),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },
    table.header[Approach][Detection][Tracking][Characteristics],
    [A], [YOLOv8/v11], [ByteTrack], [Fast baseline, motion-only association],
    [B], [YOLOv8/v11], [BoT-SORT], [Appearance features, better occlusion handling],
    [C], [RT-DETR], [ByteTrack], [Transformer-based detection, no NMS],
  )
) <tab:approaches>

All detectors are initialized with weights pre-trained on COCO and fine-tuned on our Robot Sumo dataset. The tracking stage assigns consistent identities (Robot A and Robot B) throughout each round, enabling per-robot metric extraction.

== Metric Extraction

From the continuous trajectories produced by the tracker, we compute the following metrics in the dohyo reference frame (calibrated via the detected ellipse):

- *Position:* $(x, y)$ coordinates per frame, mapped from pixel to centimeter space.
- *Trajectory:* The complete path traversed during a round.
- *Velocity:* First derivative of position over time, expressed in cm/s.
- *Acceleration:* Second derivative of position over time.
- *Reaction time:* Number of frames between round start and first significant displacement.
- *Position heatmap:* Spatial distribution of each robot across the dohyo.

== Event Detection

We detect four key events per round:

+ *Round start:* First frame where either robot exhibits significant displacement.
+ *Contact:* First frame where robot bounding boxes overlap or reach minimum distance.
+ *Ring-out:* Frame where a robot's bounding box center exits the detected dohyo ellipse.
+ *Round end:* Determined by ring-out detection or absence of motion (timeout).

The primary approach is rule-based, leveraging the extracted metrics and the spatial relationship between robot positions and the dohyo boundary.
