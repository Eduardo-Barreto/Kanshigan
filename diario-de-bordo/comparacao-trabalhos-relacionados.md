# Tabela Comparativa de Trabalhos Relacionados

Este documento serve duplo propósito: artefato da seção *Related Work* do artigo IEEE do Kanshigan, e justificativa interna para escolhas arquiteturais (detector, tracker, hardware). A pergunta-problema do trabalho orienta a seleção: equilíbrio entre acurácia e viabilidade prática para extração automatizada de métricas em partidas de Sumô de Robôs autônomos de 3 kg.

## Metodologia da Busca

**Bases consultadas:** arXiv, IEEE Xplore, ACM Digital Library, CVF Open Access, Springer Link, MDPI, PMC/PubMed, ScienceDirect, Nature Scientific Reports.

**Queries representativas:**
- "YOLO ByteTrack soccer player tracking benchmark MOTA IDF1"
- "SoccerNet tracking challenge 2023 player detection"
- "SportsMOT dataset multi-object tracking sports"
- "table tennis ball tracking deep learning real-time"
- "TrackNet shuttlecock badminton trajectory detection"
- "ice hockey player tracking computer vision"
- "SSL-Vision RoboCup small size league"
- "MMA combat sports action recognition computer vision"
- "taekwondo karate automatic scoring computer vision"
- "OC-SORT observation-centric SORT CVPR 2023"
- "BoT-SORT tracking IDF1"
- "RT-DETR real-time object detection"
- "Grounding DINO SAM pseudo-labeling detector"
- "SAM 2 video annotation pseudo labels"

**Critérios de inclusão:**
1. Domínio analogamente desafiador: rastreamento multiobjeto rápido, oclusão mútua, aparência semelhante, ou objetos pequenos em alta velocidade.
2. Detector e/ou tracker explicitamente descritos.
3. Métricas reportadas em benchmark público (MOTA, IDF1, HOTA, mAP, FPS, acurácia).
4. Venue: CVPR, ICCV, ECCV, ICRA, IROS, WACV, ACM MM, TPAMI, ou periódico indexado. arXiv aceito apenas para 2022+, ou se preenche gap não coberto.
5. Para o Eixo B (foundation models), apenas obras seminais 2023 a 2024.

**Critérios de exclusão:**
- Blog posts e Medium genéricos.
- Tutoriais sem contribuição original.
- Repositórios sem paper acompanhante (exceto SSL-Vision, citado pelo paper original RoboCup 2009 e ainda em uso).

**Distribuição alvo:** 70 por cento sports analytics, 20 por cento robôs em competição, 10 por cento foundation models.

## Tabela Comparativa

| ID | Citação | Eixo | Domínio | Detector | Tracker | Dataset | Métricas | Hardware | Tempo real | Restrições do domínio | Gap p/ Kanshigan | Link |
|----|---------|------|---------|----------|---------|---------|----------|----------|------------|----------------------|------------------|------|
| W1 | Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022 | A | Pedestres / MOT genérico | YOLOX | ByteTrack (Kalman + IoU em dois estágios) | MOT17, MOT20 | MOTA 80.3, IDF1 77.3, HOTA 63.1, 30 FPS V100 | NVIDIA V100 | Sim, 30 FPS | Pedestres densos, oclusão | Não testado em alvos de baixa diversidade visual e arena fechada | https://arxiv.org/abs/2110.06864 |
| W2 | Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian Tracking", arXiv 2022 | A | Pedestres / MOT genérico | YOLOX | BoT-SORT (motion + appearance + camera motion compensation) | MOT17, MOT20 | MOTA 80.5, IDF1 80.2, HOTA 65.0 | GPU desktop | Online | Movimento de câmera, IDs estáveis | Compensação de câmera é overkill para câmera fixa cenital do dohyo | https://arxiv.org/abs/2206.14651 |
| W3 | Cao et al., "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking", CVPR 2023 | A | MOT robusto a movimento não linear | YOLOX | OC-SORT (ORU + OCM) | DanceTrack, MOT17, MOT20, KITTI | SOTA em DanceTrack; 700+ FPS em CPU dado detecções prontas | CPU | Sim | Movimento não linear, oclusão | Promissor para Sumô: rotações rápidas e contato físico geram trajetórias não lineares | https://arxiv.org/abs/2203.14360 |
| W4 | Sun et al., "DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion", CVPR 2022 | A | Tracking com aparência uniforme | Diversos (benchmark) | Diversos (benchmark) | DanceTrack (100K frames) | Benchmark, mostra que aparência sozinha falha | N/A | N/A | Aparência uniforme, movimento não linear | Excelente analogia: caixas pretas em Sumô compartilham aparência uniforme | https://arxiv.org/abs/2111.14690 |
| W5 | Cui et al., "SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes", ICCV 2023 | A | Tracking multiesporte (basquete, vôlei, futebol) | YOLOX | MixSort (MixFormer-like association) | SportsMOT (240 vídeos, 1.6M caixas) | SOTA em SportsMOT e MOT17 | GPU | Online | Velocidade variável, aparência similar | Provê metodologia de avaliação mas não cobre robôs nem arena circular pequena | https://arxiv.org/abs/2304.05170 |
| W6 | Gran-Henriksen et al., "Deep HM-SORT: Enhancing MOT in Sports with Deep Features, Harmonic Mean, and Expansion IOU", arXiv 2024 | A | Tracking esportivo | YOLOX | Deep HM-SORT (média harmônica de motion e appearance) | SportsMOT, SoccerNet-Tracking 2023 | HOTA 80.1 (SportsMOT), HOTA 85.4 (SoccerNet) | GPU | Online | Aparência similar, troca de IDs | Não validado em câmera amadora nem em alvos de baixíssima diversidade visual | https://arxiv.org/abs/2406.12081 |
| W7 | Shitrit et al., "SoccerNet 2023 Tracking Challenge: 3rd place MOT4MOT Team Technical Report", arXiv 2023 | A | Tracking de jogadores e bola | YOLOv8l (ball) + detector SOTA (jogadores) | MOT online + interpolação + appearance merging | SoccerNet 2023 | HOTA 66.27 (3o lugar) | GPU | Online | Pequenas bolas, oclusão | Confirma YOLOv8 viável para objetos pequenos rápidos, mas em vídeo broadcast estabilizado | https://arxiv.org/abs/2308.16651 |
| W8 | Cioppa et al., "A Context-Aware Loss Function for Action Spotting in Soccer Videos", CVPR 2020 | A | Action spotting (eventos) | CNN com loss contextual | N/A (event detection) | SoccerNet (500 jogos, 764 h) | +12.8 por cento sobre baseline | GPU | Não tempo real | Eventos esparsos, localização temporal | Útil como referência para detectar eventos de Sumô (jyusho, push out) | https://arxiv.org/abs/1912.01326 |
| W9 | Hong et al., "TrackNetV3: Enhancing Shuttlecock Tracking with Augmentations and Trajectory Rectification", ACM MM Asia 2023 | A | Detecção de shuttlecock | TrackNetV3 (heatmap U-Net based) | Trajetória por inpainting | Dataset próprio de badminton | Accuracy 97.51 por cento (vs 87.72 baseline) | GPU | Sim | Objeto pequeno, blur de movimento, oclusão | Mostra que heatmap dedicado bate detector genérico para alvos minúsculos: relevante se Sumô virar objeto-pequeno | https://dl.acm.org/doi/10.1145/3595916.3626370 |
| W10 | Voeikov et al., "TTNet: Real-time Temporal and Spatial Video Analysis of Table Tennis", CVPRW 2020 | A | Tênis de mesa: detecção de bola e eventos | TTNet (multitask CNN) | Heatmap + temporal | OpenTTGames | Detecção e eventos em tempo real | GPU consumer | Sim | Bola minúscula, alta velocidade | Demonstra pipeline multitask leve: útil para combinar detecção e classificação de eventos em uma rede | https://arxiv.org/abs/2004.09927 |
| W11 | Vats et al., "Puck Localization and Multi-Task Event Recognition in Broadcast Hockey Videos", CVPRW 2021 | A | Hóquei: localização de puck + eventos | CNN multitask | N/A | NHL broadcast | Resultados em puck localization + event recognition | GPU | Não real-time reportado | Movimentação rápida, panning de câmera | Confirma valor do multitask em domínio com objeto pequeno e câmera móvel | https://openaccess.thecvf.com/content/CVPR2021W/CVSports/html/Vats_Puck_Localization_and_Multi-Task_Event_Recognition_in_Broadcast_Hockey_Videos_CVPRW_2021_paper.html |
| W12 | Marcon et al., "Video-Based Detection of Combat Positions and Automatic Scoring in Jiu-jitsu", ACM MMSports 2022 | A | Jiu-jitsu: posições e scoring | Detector de pessoas + pose estimation | Associação temporal por pose + visual cues | Dataset próprio (125K train) | Acurácia em 18 posições | GPU | Não reportado | Oclusão severa, contato corpo a corpo | Domínio mais próximo: combate corpo a corpo em arena pequena com oclusão pesada | https://dl.acm.org/doi/10.1145/3552437.3555707 |
| W13 | Zickler et al., "SSL-Vision: The Shared Vision System for the RoboCup Small Size League", RoboCup 2009 | C | RoboCup SSL: detecção e localização de robôs | Color segmentation + Pattern (ArUco-like) | Filtragem por padrão de cores | SSL real-time | Localização em tempo real | Multi-camera + CPU | Sim, 60 Hz | Robôs com marcadores coloridos, arena calibrada | Assume marcadores fiduciais: Sumô não permite alteração visual dos robôs | https://www.cs.cmu.edu/~mmv/papers/09robocup-sslvision.pdf |
| W14 | Li et al., "AI-Enhanced Precision in Sport Taekwondo: Increasing Fairness, Speed, and Trust in Competition (FST.ai)", arXiv 2025 | A | Taekwondo: detecção de chute na cabeça | Detector custom + edge inference | N/A (classificação) | Próprio | Redução de minutos para segundos no scoring | Edge device | Sim | Ações rápidas, oclusão | Aborda arbitragem automática em combate: pipeline análoga ao Kanshigan | https://arxiv.org/abs/2507.14657 |
| W15 | Li et al., "Continual Learning for Robust Gate Detection under Dynamic Scene Changes in Autonomous Drone Racing", arXiv 2024 | C | Drone racing: detecção de gates | YOLO + continual learning | N/A | Próprio | Robustez a mudança de iluminação | Embedded GPU | Sim | Cena dinâmica, mudança de luz | Útil para lidar com iluminação heterogênea entre torneios brasileiros e All Japan | https://arxiv.org/abs/2405.01054 |
| W16 | Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection (RT-DETR)", CVPR 2024 | A | Detector real-time genérico | RT-DETR (transformer end-to-end) | N/A | COCO | AP 53.1 (R50) e 54.3 (R101); 108/74 FPS T4 | NVIDIA T4 | Sim | Sem NMS, end-to-end | Candidato a detector se YOLOv8 não der conta da arena fechada com alta similaridade | https://arxiv.org/abs/2304.08069 |
| W17 | Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection", ECCV 2024 | B | Detecção open-set guiada por texto | Grounding DINO | N/A | COCO, LVIS, ODinW | SOTA em open-set | GPU | Online (rápido para anotação) | Funciona com prompt textual | Útil como anotador para gerar pseudo-labels de "sumo robot" antes de treinar detector leve | https://arxiv.org/abs/2303.05499 |
| W18 | Ravi et al., "SAM 2: Segment Anything in Images and Videos", arXiv 2024 (Meta AI) | B | Segmentação universal em vídeo | SAM 2 (transformer + streaming memory) | Memória temporal interna | SA-V (maior dataset de segmentação em vídeo) | 3x menos interações que SAM; 6x mais rápido em imagem | GPU | Real-time em vídeo | Zero-shot, promptable | Já testado no projeto como anotador de partidas; justifica pipeline semi-automática | https://arxiv.org/abs/2408.00714 |

## Matriz de Combinatorialidade do Problema

A justificativa científica do Kanshigan não está em ser "Sumô de Robôs", e sim na soma de restrições do problema. A literatura cobre cada restrição isoladamente; nenhuma linha pinta a combinação completa. Essa matriz mapeia cada trabalho catalogado por característica relevante do nosso problema.

**Características avaliadas:**

- **C1:** eventos sub-segundo com decisão crítica (a janela de erro é menor que 1 s; uma classificação errada compromete a análise da partida inteira).
- **C2:** aparência uniforme entre alvos (objetos visualmente idênticos; nenhuma cor de time, número, textura ou padrão facial distintivos).
- **C3:** vídeo de qualidade heterogênea (broadcast profissional misturado com gravação amadora; precisa funcionar em ambos).
- **C4:** ausência de marcadores fiduciais (não se pode adicionar tags coloridas ou ArUco aos alvos; o vídeo de torneio existente é a entrada).
- **C5:** movimento não-linear (rotações abruptas, ricochetes, colisões; o filtro de Kalman padrão erra).
- **C6:** análise post-match não embarcada (não é arbitragem em tempo real nem controle do robô; é extração de métricas a partir do vídeo gravado).

**Notação:** "sim" indica que o trabalho aborda a característica como problema do seu domínio; "parcial" indica que o trabalho toca o ponto mas não é central; "não" indica que o domínio do trabalho não força essa restrição.

| Trabalho | C1 sub-segundo | C2 aparência uniforme | C3 vídeo heterogêneo | C4 sem fiduciais | C5 movimento não-linear | C6 análise post-match |
|----------|:--------------:|:---------------------:|:--------------------:|:----------------:|:-----------------------:|:---------------------:|
| W1 ByteTrack | não | parcial | não | sim | parcial | não |
| W2 BoT-SORT | não | parcial | parcial | sim | parcial | não |
| W3 OC-SORT | não | parcial | não | sim | sim | não |
| W4 DanceTrack | não | sim | não | sim | sim | não |
| W5 SportsMOT | não | sim | não | sim | parcial | não |
| W6 Deep HM-SORT | não | sim | não | sim | parcial | não |
| W7 SoccerNet 2023 Tracking | parcial | sim | não | sim | parcial | parcial |
| W8 SoccerNet action spotting | sim | não | não | sim | não | sim |
| W9 TrackNetV3 shuttlecock | parcial | não | não | sim | sim | não |
| W10 TTNet table tennis | sim | não | não | sim | sim | parcial |
| W11 Hockey puck multitask | sim | não | não | sim | sim | parcial |
| W12 Jiu-jitsu scoring | parcial | parcial | não | sim | sim | sim |
| W13 SSL-Vision RoboCup | sim | sim | não | NÃO (usa fiduciais) | sim | não |
| W14 FST.ai taekwondo | sim | não | não | sim | parcial | parcial |
| W15 Continual Learning drone gates | sim | não | sim | sim | sim | não |
| W16 RT-DETR | n/a (detector) | n/a | n/a | n/a | n/a | n/a |
| W17 Grounding DINO | n/a (anotador) | n/a | n/a | n/a | n/a | n/a |
| W18 SAM 2 | n/a (anotador) | n/a | n/a | n/a | n/a | n/a |
| **Kanshigan** | **sim** | **sim** | **sim** | **sim** | **sim** | **sim** |

Cada linha individual deixa pelo menos uma coluna sem cobrir. Os trabalhos mais próximos (W4 DanceTrack, W12 jiu-jitsu, W15 drone racing) ainda assim cobrem no máximo 4 das 6 características, e cada um deixa duas restrições críticas de fora. O Kanshigan é a primeira pipeline a operar sob a interseção completa.
