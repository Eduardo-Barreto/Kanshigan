# Comparações de arquitetura e validação do anotador

## Contexto

Reflexão sobre o paper frente aos critérios de avaliação (contribuição, validação,
metodologia, posicionamento) expôs duas lacunas que esta entrada fecha.

## Lacuna 1: a pergunta de pesquisa não estava sendo respondida

A pergunta de pesquisa fala em comparar *arquiteturas de detecção* e *algoritmos de
rastreamento*, mas o paper comparava só detector com fine-tuning vs COCO (efeito de
dado de treino, não de arquitetura) e usava um único tracker. Faltava a comparação que
a própria pergunta promete. Entregamos as duas:

**Detecção: YOLOv8s vs YOLO26n.** Treinados na mesma divisão, mesma semente (42), mesmo
`imgsz=640`. Avaliados no gold held-out por fonte:

| Detector | Fonte | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|
| YOLOv8s (11,1 M par., 28,4 GFLOPs) | BR | 0.965 | 0.767 | 0.990 | 0.964 |
| YOLOv8s | JP | 0.976 | 0.695 | 0.991 | 0.915 |
| YOLO26n (2,4 M par., 5,2 GFLOPs) | BR | 0.968 | 0.772 | 0.963 | 0.955 |
| YOLO26n | JP | 0.993 | 0.691 | 0.986 | 0.976 |
| COCO zero-shot | BR | 0.026 | 0.017 | 0.032 | 0.750 |

(Números BR após corrigir o gold, ver abaixo. O YOLO26n iguala ou supera o YOLOv8s nas
duas fontes.)

Resultado: o YOLO26n, com ~1/5 dos parâmetros, iguala o YOLOv8s no BR e o supera no JP.
A arquitetura compacta basta para o domínio e melhora a viabilidade sem perder
acurácia. Boa resposta para o eixo acurácia-vs-viabilidade da pergunta.

**Rastreamento: OC-SORT vs ByteTrack.** Sobre as mesmas detecções, contra o gold de
identidades BR:

| Tracker | MOTA | IDF1 | ID switches |
|---|---|---|---|
| OC-SORT | 0.881 | 0.933 | 1 |
| ByteTrack | 0.898 | 0.947 | 0 |

ByteTrack fica ligeiramente à frente neste round (sem troca de identidade), mas um
único round não decide entre eles. O honesto é dizer que ambos sustentam identidade no
caso típico, e a comparação robusta precisa de mais rounds gold com identidade.

## Lacuna 2: SAM 3 como anotador não estava medido

A metodologia dizia que o gold validava o anotador (SAM vs gold), mas o número nunca
fora calculado (`results/E1_sam3_vs_gold` estava vazio). Implementamos
`evaluate_annotator` (concordância por IoU entre as caixas propostas e o gold) e
`annotate_gold_images.py`, que roda o SAM 3 nos próprios quadros do gold (evita o
problema de alinhar a numeração do gold, que é um subconjunto renumerado).

No gold BR: precisão 0.98, recall 0.94, F1 0.96, IoU médio 0.91. Forte: quando o SAM
propõe uma caixa, ela é quase idêntica à do humano, e a revisão mexeu em poucos
quadros.

No gold JP, achado importante: o predictor de **imagem** quadro a quadro detecta os
robôs pretos pequenos em só ~22% dos quadros (precisão 0.95, IoU 0.89 quando acha, mas
recall baixo). Não é contradição com a história do C3: lá usamos o predictor de
**vídeo**, cuja propagação temporal mais o limiar 0.15 recuperam os dois robôs (202
quadros de treino). Ou seja, a ferramenta certa para o JP é o predictor de vídeo, não o
de imagem. Reportamos o BR como validação limpa do anotador e o JP com essa ressalva.

## Outros ajustes da rodada

- **Tabela do conjunto de dados:** os números apareciam sem unidade. Cabeçalho virou
  "Quadros BR" / "Quadros JP" e a legenda explica que Clips é o total de rounds.
- **"Pré-banca" fora do paper:** trocado por "estado inicial do projeto".
- **round_start por artefato do SG:** o evento disparava no quadro 1 por ruído de borda
  do Savitzky-Golay. Agora exige movimento sustentado e pula os primeiros quadros; no
  gold passou a disparar no quadro 20 (~0.33 s), o primeiro movimento real.
- **Números de cinemática:** velocidade de pico medida no gold, 2,9 m/s (A) e 2,7 m/s
  (B), entram no texto para sustentar o claim de extração de velocidade.
- `train.py` agora parametriza o nome da run, para treinar arquiteturas diferentes sem
  sobrescrever.
- **Gold BR: 4 quadros com coordenadas negativas.** Os quadros 56-59 tinham caixas que
  extrapolavam o recorte do dohyo (robô parte fora), gerando coordenadas negativas que o
  YOLO descartava (avaliava 55/59). A causa estava em `box_native_to_crop_yolo`, que não
  recortava a caixa ao retângulo do crop. Corrigido: a caixa é clampada ao crop antes de
  normalizar, e caixas totalmente fora são descartadas. O gold passou a avaliar os 59
  quadros (3 viram quadros de fundo, com o robô fora do recorte). Os números BR caíram
  um pouco por incluir os quadros finais difíceis (clash/blur), o que é mais honesto. O
  SAM-vs-gold BR melhorou (F1 0.92 → 0.96), pois o gold clampado casa com o que o SAM vê
  dentro do recorte.

## Status

- Paper recompila em 11 páginas, sem erros, sem travessões.
- Pergunta de pesquisa respondida nas duas metades (detector e tracker comparados).
- SAM 3 validado como anotador (BR F1 0.92).
- Pendente: mais rounds gold com identidade para comparação robusta de trackers;
  rastreador com aparência (Deep HM-SORT); base de eventos anotada para a cinemática.
