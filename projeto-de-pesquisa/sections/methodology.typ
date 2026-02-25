= Metodologia

O pipeline proposto consiste em cinco etapas sequenciais: detecção do dohyo, detecção dos robôs, multi-object tracking, extração de métricas e detecção de eventos.

== Construção do dataset

*Coleta:* Vídeos de partidas serão obtidos de transmissões do All Japan Robot Sumo Tournament no YouTube, gravações de competições brasileiras e material de equipes parceiras. Todos os vídeos serão da categoria 3kg autônomos, com meta de 150 a 200 clips de rounds individuais.

*Anotação semi-automática:* Foundation models de segmentação (SAM é o candidato mais provável, com resultados preliminares positivos no domínio) gerarão máscaras de segmentação para robôs e limites do dohyo, convertidas em bounding boxes no formato YOLO. Revisão manual garantirá a qualidade das anotações.

*Anotação de eventos:* Labeling manual de timestamps para início do round, primeiro contato, ring-out e fim do round, além de labels de resultado (qual lado venceu).

*Splits:* 70% treino, 15% validação, 15% teste, estratificados por fonte do vídeo (Japão vs. Brasil), qualidade e ângulo de câmera.

== Detecção do dohyo

O dohyo é uma arena circular que aparece como elipse sob projeção em perspectiva. A detecção explora o alto contraste entre a superfície preta e a borda branca (tawara). Os parâmetros da elipse detectada fornecem uma região de interesse (ROI) e permitem calibração espacial de pixels para centímetros, dado o diâmetro conhecido de 154cm para a classe 3kg @fujisoft-rules.

== Detecção e tracking dos robôs

Três abordagens de detecção e tracking serão comparadas:

#figure(
  caption: [Abordagens de detecção e tracking comparadas neste estudo.],
  table(
    columns: (auto, auto, auto, auto),
    align: (center, center, center, left),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },
    table.header[Abordagem][Detecção][Tracking][Característica],
    [A], [YOLOv8/v11], [ByteTrack], [Baseline rápido, associação apenas por movimento],
    [B], [YOLOv8/v11], [BoT-SORT], [Features de aparência, melhor em oclusão],
    [C], [RT-DETR], [ByteTrack], [Detecção baseada em transformers, sem NMS],
  )
) <tab:approaches>

Todos os detectores serão inicializados com pesos pré-treinados no COCO e fine-tuned no dataset de Sumô de Robôs. O tracking atribui identidades consistentes (Robô A e Robô B) ao longo de cada round.

*Métricas de avaliação do tracking:* MOTA, MOTP, IDF1, ID Switches e FPS.

== Extração de métricas

A partir das trajetórias contínuas produzidas pelo tracker, as seguintes métricas serão calculadas no referencial do dohyo:

- *Posição:* coordenadas $(x, y)$ por frame, mapeadas de pixels para centímetros via calibração da elipse.
- *Trajetória:* caminho completo percorrido durante o round.
- *Velocidade:* primeira derivada da posição no tempo (cm/s).
- *Aceleração:* segunda derivada da posição no tempo.
- *Tempo de reação:* frames entre o início do round e o primeiro deslocamento significativo.
- *Heatmap de posição:* distribuição espacial de cada robô no dohyo.

== Detecção de eventos

Quatro eventos-chave por round:

+ *Início do round:* primeiro frame onde qualquer robô apresenta deslocamento significativo.
+ *Contato:* primeiro frame onde bounding boxes dos robôs se sobrepõem ou atingem distância mínima.
+ *Ring-out:* frame onde o centro do bounding box de um robô ultrapassa a elipse do dohyo.
+ *Fim do round:* determinado pela detecção de ring-out ou ausência de movimento (timeout).

A abordagem primária é baseada em regras, utilizando as métricas extraídas e a relação espacial entre posições dos robôs e os limites do dohyo.

== Avaliação experimental

*Protocolo:* Comparação das três abordagens na mesma split de teste, com avaliação separada por tipo de vídeo (broadcast vs. amador), ângulo de câmera e duração do round.

*Métricas do pipeline:* Acurácia de detecção (mAP\@0.5, mAP\@0.5:0.95), qualidade do tracking (MOTA, IDF1, ID Switches), detecção de eventos (precision/recall por tipo), acurácia das métricas (erro médio de posição/velocidade contra ground truth manual) e viabilidade de tempo real (FPS em GPU de referência).
