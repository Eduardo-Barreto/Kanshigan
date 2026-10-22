= Discussion <sec-discussion>

== What the results answer

The question asked for the detector and tracker combination that best balances accuracy and viability. In the measured data, accuracy does not separate the combinations, and cost does. The two fine-tuned detectors tie within seed variation; the four trackers tie within one or two boxes. The choice is therefore decided by cost. Among trackers, motion-only ones win clearly: ReID cuts the pipeline to about 40% of its speed with no measurable gain. Among detectors, the winner depends on the hardware: on GPU the two are close on native video and YOLOv8s is faster on the crop; on CPU, YOLO26n nearly doubles the speed.

What actually moved accuracy was domain fine-tuning, not architecture: it lifted mAP\@0.5 from 0.36 to above 0.97 on the overhead view, while multiplying the parameters by almost five changed nothing. The arena crop also mattered during development, by removing the background that produced false positives, but that gain cannot be reproduced from the saved artifacts. The transferable lesson is that, for rigid targets in a known arena, domain data pays off more than model capacity.

Two results do not follow the reasoning the literature suggested. First, OC-SORT, designed for non-linear motion (C1), did not beat ByteTrack in this round. With 59 frames at 10 Hz, the reference captures few of the collisions where the OC-SORT correction would matter; C1 remains untested. Second, ReID did not help, as in DanceTrack @sun2022dancetrack, but for a different reason: there the targets look alike; here they are distinguishable, and motion suffices. The claim the data support is narrow. When motion preserves identity, appearance is not worth its cost. Whether it pays off when robots are indistinguishable and motion fails, as in long occlusions, remains open.

== Threats to validity

*Sample.* Detection is evaluated on one round per source, and tracking on a single BR round, with 59 frames sampled at 10 Hz. Short identity switches between sampled frames do not appear in the metric. We report no confidence intervals because frames of the same round are strongly correlated; the three seeds of E5 are the only variance estimate, and only for YOLOv8s. The tracking grid uses seed 42; with seed 0, the same detector produces 28 false positives inside the arena on the BR round, and we did not measure the effect on tracking.

*Train-test independence.* The split is by round, not by tournament. The BR reference round is an IRONCup 2025 semifinal between two robots that also appear in training; one of them is in every BR training and validation round. The JP round comes from the same continuous video as JP training. The numbers measure generalization to new rounds of known events. Generalization to new events only appears in the label-free evaluation (@sec-label-free), and there it fails under extreme blur.

*Labels.* Training labels were not reviewed, and reviewing the 14 JP rounds changed 71% of their frames. The reference set started from the SAM 3 proposal, and human review tends to accept plausible boxes, which favors the annotator in E1 and may favor detectors trained on the same proposals.

*Temporal resolution.* In the BR clip, 48% of consecutive frame pairs are nearly identical, consistent with a 30 FPS recording delivered at 60. If confirmed, the effective temporal resolution of BR is half the nominal one.

*Development decisions.* We adopted the arena crop after observing, on the BR reference round, the full-frame detector's precision (about 0.71). That detector's weights were not preserved.

== Kinematics and events

The pipeline computes position, velocity and events, but this work does not validate them, for three reasons. First, the isotropic scale reaches a coefficient of variation of 0.38 within a match filmed with a phone (@sec-label-free); speeds in centimeters per second inherit that error until the arena is rectified by homography. Second, the event rules depend on the detector: on the BR round, round start fires at frame 20 with YOLOv8s and at frame 310 with YOLO26n. Third, there is no annotated event set separate from the test, and calibrating thresholds on the test itself would leak the evaluation. Until that validation exists, event output serves as a suggestion for human review, not as a measurement.

== Future work

The most valuable step is to expand the identity reference, with more rounds from both sources, including matches between truly similar robots and collisions annotated at every frame. That would test C1 and C2 for real. The repository's annotation Studio already records identity, events and round boundaries. Next come rectifying the arena by homography and validating the scale against known measurements; annotating events in a separate calibration set; rebalancing the sources and unifying the convention for the robot already pushed out; adding blurred broadcast video to training; and comparing detectors from another family, such as RT-DETR @zhao2024rtdetr. In the long run, the pipeline can support an open platform for robot combat analysis.

= Conclusion

This work asked which detector and tracker combinations best balance accuracy and viability when tracking visually similar targets in fast, non-linear motion, from uncalibrated, markerless video. We used autonomous 3 kg Robot Sumo as the domain and built Kanshigan, an open tracking-by-detection pipeline, with detectors trained on SAM 3 annotations and evaluated against hand-reviewed rounds.

The answer, under the measured conditions, is a YOLO detector fine-tuned on the arena crop paired with a motion-only tracker, such as OC-SORT or ByteTrack: YOLOv8s or YOLO26n with a GPU, YOLO26n with CPU only. Accuracy comes from domain fine-tuning, not architecture, and appearance is not worth its cost when motion already preserves identity. The pipeline runs in real time on 1080p video on a laptop GPU, using under 300 MiB of memory, and at 40 FPS on CPU alone.

The scope is limited by one test round per source and a single identity round, both from events present in training. Kinematics and events are produced but not validated. The code, the scripts and results behind every table, and the project log recording every decision and every mistake are at #link("https://github.com/Eduardo-Barreto/Kanshigan")[github.com/Eduardo-Barreto/Kanshigan].
