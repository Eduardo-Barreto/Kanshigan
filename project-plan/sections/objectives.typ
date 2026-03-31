#import "../utils/sections.typ": make-objectives-section

#make-objectives-section(
  general-objective: [
    To develop and evaluate a computer vision pipeline for detection, tracking, and automated performance metric extraction in autonomous Robot Sumo matches (3kg) from competition videos.
  ],
  specific-objectives: (
    "Build an annotated dataset of 3kg Robot Sumo matches from competition videos (All Japan, Brazilian competitions).",
    "Implement automatic detection of the dohyo (circular arena) in video frames with varying camera angles.",
    "Implement and compare robot detection and tracking approaches, evaluating accuracy and processing speed.",
    "Extract quantitative metrics from tracking data: position, trajectory, velocity, acceleration, and reaction time.",
    "Detect key events in each round: start, contact, ring-out, and end.",
    "Evaluate pipeline feasibility on videos of heterogeneous quality (professional broadcasts vs. amateur recordings).",
    "Publicly release the dataset, source code, and trained models."
  )
)
