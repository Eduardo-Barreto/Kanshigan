= Related Work

== Computer Vision in Sports Analytics

// TODO: Expandir

Computer vision has been increasingly applied to sports analytics across multiple domains. Player tracking systems using YOLO-based detectors combined with multi-object trackers such as DeepSORT @wojke2017deep have become standard in soccer and basketball analysis. Automated scoring systems have been developed for taekwondo using human-in-the-loop approaches, and electronic referee systems for table tennis have achieved 97.8% accuracy in ball position detection. Action spotting frameworks such as SoccerNet @giancola2018soccernet enable automatic detection of key events in match footage.

== Object Detection and Multi-Object Tracking

// TODO: Expandir

Real-time object detection has been dominated by the YOLO family @jocher2023yolo, with recent versions achieving strong accuracy-speed trade-offs. Transformer-based approaches such as RT-DETR @zhao2024rtdetr offer end-to-end detection without non-maximum suppression. For multi-object tracking, ByteTrack @zhang2022bytetrack associates both high and low confidence detections to maintain tracking continuity, while BoT-SORT @aharon2022botsort incorporates appearance features for improved robustness during occlusion.

== Computer Vision in Robot Competitions

// TODO: Expandir

The closest related work involves automated refereeing systems for robot soccer, which process video at 20fps to detect robots and make event-based decisions. On the robot side, BattleBots competitor Orbitron integrated YOLOv8 for autonomous opponent detection. A small dataset of sumo robot images exists on Roboflow Universe for detection purposes. However, *no prior work addresses external video-based analysis of Robot Sumo matches* for performance extraction and event classification.
