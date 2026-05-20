# Mapa de Literatura: Related Work do Kanshigan

## Resumo Executivo

A literatura de visão computacional aplicada a esportes converge fortemente em uma arquitetura tracking-by-detection com YOLO como detector e variantes de SORT como tracker. O par YOLOX mais ByteTrack se tornou baseline padrão em MOT genérico [W1], e o mesmo padrão se replica em benchmarks esportivos: SportsMOT, SoccerNet e DanceTrack avaliam variantes desse esquema [W4, W5, W7]. Os trabalhos recentes que superam ByteTrack agregam motion compensation (BoT-SORT [W2]), associação observation-centric (OC-SORT [W3]) ou fusão de features profundos por média harmônica (Deep HM-SORT [W6]). Para objetos pequenos e rápidos, redes heatmap dedicadas como TrackNetV3 [W9] e TTNet [W10] superam detectores genéricos. Em combate corpo a corpo com oclusão severa, pose estimation acoplada a associação multimodal funciona melhor que bounding boxes (jiu-jitsu [W12]). RT-DETR [W16] surge como alternativa transformer end-to-end com paridade de FPS e AP superior a YOLO em COCO. Foundation models (SAM 2 [W18], Grounding DINO [W17]) são adotados como anotadores semi-automáticos, não como pipeline final, devido ao custo computacional. A escolha para o Kanshigan é YOLOv8 ou RT-DETR para detecção e OC-SORT ou Deep HM-SORT para tracking, com SAM 2 e Grounding DINO restritos à etapa de pseudo-labeling do dataset.

## Padrões Observados

**Convergência em tracking-by-detection.** Treze dos dezoito trabalhos analisados (todos do Eixo A) usam o paradigma detect-then-track. Apenas SSL-Vision [W13], TTNet [W10] e TrackNetV3 [W9] divergem: SSL-Vision usa segmentação por cor com marcadores fiduciais (inviável em Sumô, pois os robôs vêm como caixas pretas sem marcadores), TTNet e TrackNetV3 usam heatmap específico para objeto pequeno único.

**Família YOLO domina detecção.** YOLOX aparece em W1, W2, W3 e W5. YOLOv8 em W7. YOLO generalista em W15. Apenas RT-DETR [W16] propõe alternativa não-YOLO competitiva em tempo real, e ainda assim é transformer pós-DETR.

**ByteTrack e variantes dominam tracking.** Sete trabalhos usam ByteTrack ou descendente direto (BoT-SORT, OC-SORT, Deep HM-SORT). DeepSORT clássico não aparece como SOTA em sports tracking pós-2022, suplantado por trackers que combinam motion e appearance de forma mais sofisticada.

**Aparência uniforme exige features profundos.** DanceTrack [W4] estabeleceu o benchmark de aparência uniforme: trackers que dependem só de IoU falham, e os que adicionam features profundos com Re-ID (BoT-SORT [W2], Deep HM-SORT [W6]) lideram. Este é o cenário mais próximo do Sumô de Robôs (caixas pretas visualmente idênticas).

**Movimento não linear é o calcanhar de Aquiles do Kalman padrão.** OC-SORT [W3] foi projetado especificamente para isso e atinge SOTA em DanceTrack. Sumô tem colisões e giros bruscos que quebram o modelo linear.

**Objetos pequenos rápidos pedem rede dedicada.** Em badminton [W9] e tênis de mesa [W10], detectores generalistas perdem para heatmaps temporais. No Sumô, o robô não é pequeno (20 cm em arena de 154 cm), então YOLO genérico é suficiente.

**Combate corpo a corpo precisa de pose.** Em jiu-jitsu [W12], bounding box falha porque os atletas se sobrepõem; pose estimation com associação multimodal resolve. Sumô de Robôs tem o mesmo padrão de contato e empurrão, mas robôs não têm articulações: chassis rígido pode ser tratado por orientação (yaw) em vez de pose humana.

**Robôs em competição usam marcadores fiduciais.** SSL-Vision [W13] e VSSS dependem de padrões coloridos no topo dos robôs. Esse atalho não está disponível para o Kanshigan, que analisa partidas oficiais sem instrumentação adicional.

**Foundation models são anotadores, não inferência final.** Grounding DINO [W17] e SAM 2 [W18] aparecem em 2024 a 2025 como geradores de pseudo-labels para treinar modelos menores. Nosso experimento 08 confirmou: SAM 2 e SAM 3 não rodam em tempo real para o caso Sumô, mas servem para acelerar anotação.

## Recomendação Fundamentada para o Kanshigan

**Detector: YOLOv8 (small ou medium), com RT-DETR como ablação.**
- YOLOv8 é o estado da arte prático para tracking-by-detection em esporte [W7]. O time MOT4MOT venceu posição 3 na SoccerNet 2023 usando YOLOv8l até para a bola, que é objeto muito menor que um robô de 20 cm.
- RT-DETR [W16] é alternativa válida para ablation, com AP superior em COCO. Vale comparar empiricamente, pois a arena é controlada (fundo branco do dohyo) e pode favorecer transformer end-to-end sem NMS.
- YOLOv8 small (~11M parâmetros) é suficiente para o domínio: a complexidade visual do dohyo é baixa, robôs são alvos grandes relativos ao quadro.

**Tracker: OC-SORT como baseline, Deep HM-SORT como upgrade.**
- OC-SORT [W3] lida bem com movimento não linear (giros bruscos pós-colisão) e roda a centenas de FPS em CPU. É o tracker certo dado que rounds têm menos de 1 segundo e exigem alta confiabilidade temporal.
- Deep HM-SORT [W6] adiciona features profundos: justificável porque robôs são visualmente idênticos (DanceTrack-style [W4]). Os 85.4 HOTA em SoccerNet-Tracking 2023 sugerem que o ganho compensa o overhead de Re-ID.
- BoT-SORT [W2] traz camera motion compensation, mas a câmera do dohyo é fixa cenital: o custo extra não se paga.

**Não usar marcadores fiduciais.**
- A literatura RoboCup [W13] depende deles, mas o Kanshigan analisa vídeos de torneio existentes (All Japan, brasileiros) onde os robôs não podem ser modificados. Toda a pipeline parte de "como o robô é apresentado" sem assumir instrumentação.

**Anotação: SAM 2 e Grounding DINO em pipeline semi-automática.**
- Grounding DINO [W17] gera bounding boxes a partir do prompt "sumo robot" sem treino prévio.
- SAM 2 [W18] propaga máscaras frame-a-frame em vídeo, reduzindo trabalho humano. Conforme observado em [W18], três vezes menos interações que SAM 1.
- Combinar os dois (Grounded-SAM-2) gera supervisão para treinar YOLOv8 ou RT-DETR final.

**Métricas a reportar.**
- MOTA, IDF1 e HOTA, alinhadas com SportsMOT [W5] e SoccerNet [W7].
- mAP@0.5 e mAP@0.5:0.95 para o detector isolado, alinhado com COCO style [W16].
- FPS em hardware alvo (notebook + GPU consumer; sem assumir cluster).
- Adicionar uma métrica de domínio: acurácia na decisão de jyusho (push out) e ippon, equivalente ao scoring automático em jiu-jitsu [W12] e taekwondo [W14].

## Gaps que o Kanshigan Endereça

A contribuição científica do Kanshigan não é "Sumô de Robôs como domínio inédito". Esse enquadramento seria fraco: a banca questionaria se um nicho justifica pesquisa. O argumento correto é combinatório.

A literatura cobre cada uma das restrições do nosso problema isoladamente. Nenhum trabalho cobre a interseção. A matriz de combinatorialidade documentada em [comparacao-trabalhos-relacionados.md](comparacao-trabalhos-relacionados.md) torna isso explícito: dos 15 trabalhos relevantes do Eixo A e C, os três que mais cobrem (W4 DanceTrack, W12 jiu-jitsu scoring, W15 continual learning drone racing) atingem no máximo quatro das seis restrições. O Kanshigan é a primeira pipeline a operar sob a interseção completa.

As seis restrições combinadas são:

1. **Eventos sub-segundo com decisão crítica.** Rounds inteiros duram menos de 1 s. Uma janela de erro de poucos frames compromete a classificação. A literatura cobre isolado em action spotting [W8] e bola/shuttlecock [W9, W10, W11], mas sempre relaxando outra restrição (qualidade homogênea, sem oclusão entre alvos similares).

2. **Aparência uniforme extrema, sem marcadores.** DanceTrack [W4] é o benchmark mais próximo, mas dançarinos ainda têm cor de roupa distinta entre objetos. SoccerNet [W7] tem aparência similar mas com número de camisa e cor de time. SSL-Vision [W13] resolve assumindo marcadores fiduciais que não estão disponíveis em vídeo de torneio existente. Robôs de Sumô 3kg são caixas pretas visualmente idênticas, sem textura, sem cor distintiva.

3. **Vídeo de qualidade heterogênea.** SoccerNet [W5, W7, W8] usa broadcast profissional padronizado. SportsMOT [W5] também. O Kanshigan precisa lidar com gravações de torneio amador brasileiro (handheld, sem padronização) e All Japan Robot Sumo Tournament (broadcast) no mesmo pipeline. Drone racing [W15] chega perto com mudança de cena, mas não cobre diferença de qualidade de captura.

4. **Ausência de marcadores fiduciais.** SSL-Vision [W13] e BattleBots em estação de pilotagem dependem deles. O Kanshigan analisa vídeos como eles foram gravados originalmente.

5. **Movimento não-linear.** Rotações, ricochetes e colisões geram trajetórias que quebram o filtro de Kalman padrão. OC-SORT [W3] endereça isso, mas isolado de C2 (aparência uniforme). Jiu-jitsu [W12] tem contato corpo a corpo, mas resolve com pose estimation que não se aplica a chassis rígido.

6. **Análise post-match não embarcada.** RoboCup [W13] roda em tempo real para controlar o robô. Action spotting [W8] e jiu-jitsu scoring [W12] são offline mas dependem de anotação humana densa. O Kanshigan extrai métricas a partir de vídeo gravado, sem anotação humana além do pequeno gold set.

A interseção destas seis restrições caracteriza uma classe de problemas mais ampla que Sumô de Robôs: drone racing sem fiduciais, outras categorias de combate de robôs (ant fights, antweight, lightweight), micro-mouse competitions, qualquer cenário de combate ou disputa rápida entre alvos rígidos visualmente idênticos. A contribuição metodológica e arquitetural do Kanshigan é transferível para essa classe, com Sumô como o veículo onde as restrições aparecem em estado puro e forçadas.

## Fontes Marcadas como Mais Fracas (Reavaliar)

- **W14 (FST.ai taekwondo, arXiv 2025):** preprint sem revisão por pares confirmada. Citado por descrever pipeline completa de scoring automático em combate, gap raro de cobrir. Validei o link, paper acessível.
- **W15 (Continual Learning Gate Detection, arXiv 2024):** preprint. Inclui valor real para o problema de iluminação heterogênea entre torneios.
- **W6 (Deep HM-SORT, arXiv 2024):** preprint mas com resultados validados em dois benchmarks públicos (SportsMOT e SoccerNet-Tracking). Aceitável como referência de SOTA recente.
- **Orbitron BattleBots (não numerado):** apenas wiki da Fandom como registro disponível. Não incluído na tabela formal por falta de paper, mas mencionado nas observações; vale citar no texto do artigo como evidência prática de CV em combate de robôs.

## Verificação de Links

Confirmei acesso a arxiv.org/abs/2110.06864 (W1), arxiv.org/abs/2206.14651 (W2), arxiv.org/abs/2304.05170 (W5), arxiv.org/abs/2308.16651 (W7), arxiv.org/abs/2406.12081 (W6) e arxiv.org/abs/2408.00714 (W18) com extração bem sucedida de metadados. Os demais links seguem o padrão de URL canônica de cada base (arXiv, openaccess.thecvf.com, dl.acm.org, cs.cmu.edu, springer.com) que foram retornados pela busca como resultados de primeira página.
