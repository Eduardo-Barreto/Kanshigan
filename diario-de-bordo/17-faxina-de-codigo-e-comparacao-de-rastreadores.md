# Faxina de código e comparação de rastreadores

## Contexto

Rodada de melhoria da pipeline com três frentes: limpar o código morto, cobrir com
testes a lógica que ainda não tinha rede de proteção e fechar a pendência de
rastreador com aparência que a entrada 16 deixou aberta. Tudo sobre o mesmo gold BR
(`gold_zb01`, 385 quadros a 60 fps), sem retreinar nada.

## Parte 1: faxina de código (anti-slop)

Varredura por símbolo definido e nunca usado. Cinco trechos eram código morto, vivos
só no arquivo onde nasceram:

- `Calibration.major_axis_px` (propriedade sem nenhum chamador).
- `metrics.spatial_heatmap` (mapa de calor que ninguém plotava).
- `dohyo.has_match` mais as constantes `MATCH_MIN_SCORE` e `MATCH_MIN_AREA_RATIO`,
  que só existiam para essa função. O segmentador usa `detect_calibration` direto,
  com os próprios limiares, então `has_match` nunca foi chamado.
- `crop.box_crop_to_native` (a volta de recorte para nativo, sem uso; a ida,
  `box_native_to_crop_yolo`, é a usada de fato).
- `import cv2` em `crop.py`: o recorte é fatiamento NumPy puro, o OpenCV nunca entrava.

`Calibration.radius_cm` parecia morto, mas `test_dohyo.py` o trava. Mantido.

Confirmação por `pyflakes` (zero avisos) e pela suíte (23 testes seguiam verdes). Nada
de comportamento mudou: foram só remoções.

## Parte 2: cobertura de testes

A lógica pura de avaliação e de geometria de caixa não tinha teste. Três arquivos
novos, de 23 para 40 testes:

- `test_crop.py`: trava o conserto da entrada 16. Aquele bug das coordenadas negativas
  (caixa de robô que vaza a borda do dohyo) agora tem regressão explícita: nenhuma
  saída de `box_native_to_crop_yolo` pode ser negativa, e a caixa que extrapola é
  recortada ao retângulo do crop antes de normalizar.
- `test_evaluate.py`: `_iou` (caixas idênticas, disjuntas, meia sobreposição),
  `_load_mot`, `evaluate_annotator` (concordância perfeita, FN por robô perdido, FP por
  caixa sobrando) e `evaluate_events` (casa dentro da tolerância, rejeita fora, recall
  zero quando o evento falta).
- `test_schema.py`: `TrackPoint.center_px` e `Track.frames`.

Regra do diário: cada bug corrigido vira um teste para não voltar. O do recorte
estava sem o seu; agora está coberto.

## Parte 3: comparação de rastreadores com detecção fixa

A entrada 16 admitiu duas lacunas no eixo de rastreamento: um round só não decide entre
trackers, e faltava um rastreador com aparência. Fechamos a segunda e deixamos o
método pronto para a primeira.

**Desenho controlado.** A pergunta é sobre a escolha do *rastreador*, então o detector
tem que ficar fixo. O script novo `tracker_comparison.py` roda o YOLO uma vez sobre o
gold, guarda as detecções por quadro (706 caixas em 385 quadros) e replica essas
*mesmas* detecções em cada tracker. Qualquer diferença de métrica é do tracker, não do
detector. O boxmot não traz o "Deep HM-SORT" que a entrada 16 citou, então usamos os
rastreadores com aparência que ele oferece (DeepOCSORT e BoT-SORT, ambos com ReID
OSNet) contra os de movimento (OC-SORT e ByteTrack).

**Validação do harness.** O OC-SORT saiu em MOTA 0.881, IDF1 0.933, 1 troca de ID,
idêntico ao número da entrada 16. Isso confirma que o caminho de detecção e avaliação é
o mesmo; as outras três linhas são comparáveis entre si por construção.

| Rastreador | Tipo | MOTA | IDF1 | Trocas de ID | FPS (só tracker) |
|---|---|---|---|---|---|
| OC-SORT | movimento | 0.881 | 0.933 | 1 | 3183 |
| ByteTrack | movimento | 0.890 | 0.938 | 1 | 3448 |
| DeepOCSORT | aparência | 0.856 | 0.919 | 1 | 81 |
| BoT-SORT | aparência | 0.890 | 0.938 | 1 | 94 |

**Achado.** A aparência não ajuda aqui. O BoT-SORT empata com o ByteTrack e o
DeepOCSORT fica atrás dos dois de movimento. Todos seguram identidade no caso típico
(no máximo 1 troca). O custo, porém, é grande: o passo de ReID derruba o throughput de
mais de 3000 fps para 80 a 94 fps, cerca de 35 a 40 vezes mais lento, sem ganho de
acurácia.

**Por quê.** O ReID assume alvos visualmente distintos. Os dois robôs de Sumô são
caixas pretas pequenas e quase idênticas, então a aparência quase não carrega sinal
discriminativo; sobra o movimento, que os trackers baratos já modelam bem. É uma
resposta limpa para o eixo acurácia-vs-viabilidade da pergunta: para este domínio, o
rastreador de movimento é a escolha certa, e pagar por aparência só piora a viabilidade.

Figura em `results/figures/tracker_comparison.png` (dois painéis: acurácia quase
empatada à esquerda, throughput em escala log à direita). Números crus em
`results/E4_tracker_comparison/comparison.json`.

## Status

- Pipeline sem código morto; `pyflakes` limpo.
- 40 testes verdes (eram 23); o bug de recorte da entrada 16 agora tem regressão.
- Quatro rastreadores comparados sobre detecções idênticas. Aparência não supera
  movimento neste domínio e custa ~40x em throughput.
- Pendente: ainda só um round gold com identidade. A comparação robusta entre os
  quatro (e a decisão movimento-vs-aparência) precisa de mais rounds gold anotados.
