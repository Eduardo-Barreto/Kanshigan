#import "../utils/sections.typ": make-methodology-section

#make-methodology-section(
  research-design: [Applied, quantitative, exploratory, and experimental research. Existing computer vision techniques are applied to a specific Robot Sumo domain problem. Independent variables (detection architecture, tracking algorithm, resolution, frame rate) are systematically manipulated and their effects measured on evaluation metrics.],
  data: [
    - *Source:* YouTube broadcasts from the All Japan Robot Sumo Tournament, recordings from Brazilian competitions, and material from partner teams
    - *Selection criteria:* 3kg autonomous category only
    - *Expected volume:* 150 to 200 individual round clips
    - *Annotation:* Semi-automatic segmentation via foundation models (SAM), converted to YOLO-format bounding boxes with manual review. Manual labeling of event timestamps (round start, first contact, ring-out, round end) and outcome labels
    - *Splits:* 70% training, 15% validation, 15% test, stratified by video source, quality, and camera angle
  ],
  variables: [
    - *Independent:* detection architecture (YOLOv8/v11, RT-DETR), tracking algorithm (ByteTrack, BoT-SORT), input resolution, frame rate
    - *Dependent:* mAP, MOTA, IDF1, ID Switches, event precision/recall, FPS, VRAM usage
  ],
  tools: [
    - *Languages:* Python
    - *Frameworks:* PyTorch, Ultralytics
    - *Libraries:* OpenCV, SAM, supervision
    - *Hardware:* GPU workstation for training and inference benchmarking
  ],
  metrics: [
    - Detection: mAP\@0.5, mAP\@0.5:0.95
    - Tracking: MOTA, MOTP, IDF1, ID Switches
    - Events: precision and recall per event type
    - Spatial metrics: mean position/velocity error against manual ground truth
    - Feasibility: FPS on reference GPU, VRAM consumption
  ]
)
