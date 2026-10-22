= Related Work <sec-related>

Computer vision already appears in Robot Sumo, but on board the robot rather than as external analysis: the robot detects its opponent with infrared, ultrasound or its own camera during the fight. As far as our search goes, post-match analysis by an external observer, from third-person video, has no published pipeline, dataset or benchmark#footnote[Search run in June 2026 on a web search engine and on arXiv, with four query strings combining "robot sumo", "sumo robot" and "combat robot" with terms for vision, detection, tracking, metrics and refereeing. Strings, hits and categorization are versioned in the repository. Authenticated databases such as IEEE Xplore and Scopus were not covered.]. We organize the review into three lines close to the problem and derive from them six conditions (C1 to C6) that characterize the domain (@tab-matrix).

== Multi-object tracking and sports

Tracking-by-detection is the dominant paradigm in multi-object tracking. SORT @bewley2016sort set the minimal formulation: a constant-velocity Kalman filter predicts each target in the next frame, and an assignment on box overlap (IoU) links detections, without appearance. ByteTrack @zhang2022bytetrack also associates low-confidence detections, which reduces trajectory gaps under occlusion and blur (63.1 HOTA on MOT17). OC-SORT @cao2023ocsort addressed another weakness: the constant-velocity Kalman filter accumulates error when the target maneuvers or disappears. The method re-estimates the filter from the observations around the occlusion and reaches 63.2 HOTA on MOT17. Robots that collide, spin and bounce break exactly the constant-velocity assumption. Hence the first condition, *C1, non-linear motion*: collisions and abrupt direction changes happen in every round.

MOT17 films pedestrians in distinct clothing, which appearance helps to separate. DanceTrack @sun2022dancetrack isolates the opposite case: 100 group-dance videos with uniformly dressed dancers and diverse motion. There, the trackers that dominated MOT17 degrade: ByteTrack drops to 47.7 HOTA and OC-SORT stays at 55.1 (values from the DanceTrack table). Successors that reinforce geometric cues, such as C-BIoU @yang2023cbiou and Hybrid-SORT @yang2024hybridsort, reach 60.6 and 62.2. Two combat robots are dark, similar boxes, which defines *C2, uniform appearance across targets*: identity must come from motion.

SportsMOT @cui2023sportsmot extends the problem to 240 basketball, volleyball and soccer videos with high, variable speed. The authors' method reaches 74.1 HOTA, and basketball, the sport with the most physical contact, is the hardest. Deep-EIoU @huang2024deepeiou and Deep HM-SORT @deephmsort2024 combine expanded boxes with appearance for players. In Robot Sumo the time scale is more extreme: the contact that decides a round spans few frames. Hence *C3, decisive events spanning few frames*: losing ten frames in a volleyball rally costs little, but in Sumo those frames may hold the push that decides the match. Finally, SoccerNet-Tracking @cioppa2022soccernet uses the main camera of professional soccer broadcasts, in high resolution. Sumo tournament video is the opposite: handheld spectator cameras, oblique angles and gym lighting, next to fixed overhead cameras in Japan. Hence *C4, heterogeneous video quality*.

== Identical targets beyond sports

Tracking indistinguishable targets is a mature field in biology. idtracker.ai @romeroferrero2019idtracker identifies up to 100 unmarked animals with accuracy above 99.9%. TRex @walter2021trex tracks hundreds of individuals, and SLEAP @pereira2022sleap tracks the poses of multiple animals. These systems, however, learn identity from the video itself. They require long, fixed-camera recordings with a stable background, in which every target appears separated at least once. Robot Sumo denies these assumptions: a round lasts seconds, the robots spend much of it in contact, and the Brazilian camera is handheld. There is neither the time nor the capture conditions needed to learn each robot's appearance.

== Combat and competition robots

Two domains come closest to Robot Sumo. In human combat, Hudovernik and Skočaj @hudovernik2022jiujitsu score jiu-jitsu fights from fixed-camera video, using pose estimation to resolve body contact. The work shows that analyzing combat from recorded video is feasible. That regime defines *C5, post-match analysis*. The technique, however, does not apply to rigid chassis, which have no pose to estimate.

Among competition robots, SSL-Vision @zickler2010sslvision, the official system of the RoboCup Small Size League since 2010, localizes fast robots with fixed cameras and colored patterns on top of each robot. Tracking fast robots is tractable when the instrumentation is controlled. Robot Sumo denies that premise: the rules do not provide for markers, the video archive has none, and there is no calibrated camera. Hence *C6, no fiducial markers*, which rules out instrumented solutions and requires detecting robots by their natural appearance.

== The open question

Not all conditions weigh the same. C5 describes the usage regime, not a technical difficulty. C6 does not measure difficulty either: almost every work in @tab-matrix operates without markers, and the condition serves to exclude instrumented solutions. C3 and C4 describe the Robot Sumo instance, not the general problem. What remains open, with transferable value, is the pair *C1+C2* under uncontrolled video (C4, C6): tracking uniform-appearance targets in non-linear motion. DanceTrack isolates this pair, but with stable cameras and targets fully in frame. Biology solves it with long, fixed-camera recordings. Under short, uncalibrated tournament video, judged by accuracy and by the cost of running on consumer hardware, the best balance remains unanswered. That is the question this work addresses.

#figure(
  placement: top,
  scope: "parent",
  caption: [Coverage of the six domain conditions (#sym.circle.filled covers, #sym.circle.filled.tiny partial, #sym.circle does not force). C1 non-linear motion; C2 uniform appearance; C3 decisive events spanning few frames; C4 heterogeneous video; C5 post-match analysis; C6 no markers. For C5 and C6, #sym.circle.filled means the work operates in that regime. This is a map of conditions, not a performance scoreboard; the last row describes the target domain, not a result.],
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center, center),
    stroke: 0.4pt,
    table.header(
      table.cell(rowspan: 2)[*Work*],
      table.cell(colspan: 3, align: center)[*Technical challenge*],
      table.cell(colspan: 3, align: center)[*Data and usage regime*],
      [*C1*], [*C2*], [*C3*], [*C4*], [*C5*], [*C6*],
    ),
    [ByteTrack @zhang2022bytetrack], [#sym.circle.filled.tiny], [#sym.circle.filled.tiny], [#sym.circle], [#sym.circle], [#sym.circle], [#sym.circle.filled],
    [OC-SORT @cao2023ocsort], [#sym.circle.filled], [#sym.circle.filled.tiny], [#sym.circle], [#sym.circle], [#sym.circle], [#sym.circle.filled],
    [DanceTrack @sun2022dancetrack], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle], [#sym.circle], [#sym.circle], [#sym.circle.filled],
    [SportsMOT @cui2023sportsmot], [#sym.circle.filled.tiny], [#sym.circle.filled], [#sym.circle], [#sym.circle], [#sym.circle], [#sym.circle.filled],
    [SoccerNet-Tracking @cioppa2022soccernet], [#sym.circle.filled.tiny], [#sym.circle.filled], [#sym.circle.filled.tiny], [#sym.circle], [#sym.circle.filled.tiny], [#sym.circle.filled],
    [idtracker.ai @romeroferrero2019idtracker], [#sym.circle.filled.tiny], [#sym.circle.filled], [#sym.circle], [#sym.circle], [#sym.circle.filled], [#sym.circle.filled],
    [Jiu-jitsu @hudovernik2022jiujitsu], [#sym.circle.filled], [#sym.circle.filled.tiny], [#sym.circle.filled.tiny], [#sym.circle], [#sym.circle.filled], [#sym.circle.filled],
    [SSL-Vision @zickler2010sslvision], [#sym.circle.filled], [#sym.circle], [#sym.circle.filled], [#sym.circle], [#sym.circle], [#sym.circle],
    [_Robot Sumo (target domain)_], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle.filled],
  ),
) <tab-matrix>

We hypothesize that the answer carries over to other fast contests between similar rigid targets, such as other robot combat categories; @sec-label-free gives a first indication with radio-controlled Sumo.
