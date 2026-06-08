= Experiments <sec:experiments>

// TODO: Preencher com resultados

== Experimental Setup

// TODO: Descrever setup (GPU, hiperparâmetros, splits)

== Detection and Tracking Results

For the tracking axis of the research question, we compare four trackers on identical detections (fixed YOLO26n detector, held-out round), scored against a manually reviewed gold with annotated identities (@tab:tracking). Two are motion-only (OC-SORT, ByteTrack) and two use appearance via ReID (DeepOCSORT, BoT-SORT). Holding the detections fixed isolates the tracker: any gap is the tracker's, not the detector's.

All four hold identity in the typical case, with at most one switch, always during the approach between the two near-identical robots. Appearance does not improve accuracy: BoT-SORT ties ByteTrack (MOTA 0.89, IDF1 0.94) and DeepOCSORT trails both motion-only trackers. The ReID step, however, cuts tracker throughput from over 3000 frames per second to under 100, roughly 35 to 40 times slower, with no accuracy return. The cause is structural: two small, near-identical black robots offer little appearance signal, so motion alone suffices. For this domain, the motion-only tracker is the right choice on the accuracy-versus-viability axis. The FPS column reports the tracking stage in isolation (detections fixed), so it measures the tracker's own cost.

#figure(
  caption: [Four trackers over identical detections (fixed detector), scored against the gold identities on the held-out round. FPS is measured for the tracking stage alone.],
  placement: top,
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, center, center, center, center),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },
    table.header[Tracker][Type][MOTA][IDF1][ID switches][FPS],
    [OC-SORT], [motion], [0.88], [0.93], [1], [3183],
    [ByteTrack], [motion], [0.89], [0.94], [1], [3448],
    [DeepOCSORT], [appearance], [0.86], [0.92], [1], [81],
    [BoT-SORT], [appearance], [0.89], [0.94], [1], [94],
  )
) <tab:tracking>

== Event Detection Results

// TODO: Precision/recall por tipo de evento

== Qualitative Analysis

// TODO: Casos de sucesso e falha

== Real-time Viability

// TODO: Análise de FPS e viabilidade de processamento em tempo real
