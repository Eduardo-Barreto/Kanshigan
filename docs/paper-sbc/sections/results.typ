= Resultados Parciais

Esta seção reporta os resultados medidos sobre footage real até a pré-banca. Todos
os números são medidos, não projetados. O conjunto de avaliação é pequeno (um round
held-out, anotado e revisado manualmente), então os valores indicam ordem de
grandeza e viabilidade, não um modelo definitivo (ver limitações na @sec-discussao).
As medições de hardware usaram uma RTX 4070 Laptop de 8 GB.

== Conjunto de dados

O conjunto vem de partidas autônomas de 3 kg da IRONCup 2025, recortadas em rounds
individuais (rastreamento e eventos só fazem sentido dentro de um round). A divisão
é por round/clip, não por quadro, para evitar vazamento entre quadros vizinhos quase
idênticos. Para treino e validação mantemos apenas os quadros em que o anotador
produziu rótulos completos (os dois robôs), descartando quadros incompletos que
ensinariam o detector a ignorar um robô. O gold é um round inteiro mantido fora do
treino e revisado manualmente quadro a quadro.

#figure(
  caption: [Composição do conjunto, após filtrar quadros de rótulo completo. Treino e validação anotados via SAM 3; gold revisado manualmente.],
  table(
    columns: 4,
    align: (left, center, center, left),
    stroke: 0.4pt,
    table.header([*Subconjunto*], [*Clips*], [*Quadros*], [*Anotação*]),
    [Treino], [6], [431], [SAM 3, quadros completos],
    [Validação], [1], [57], [SAM 3, quadros completos],
    [Gold (teste)], [1], [59], [SAM 3 + revisão manual],
  ),
)

== Detecção

O detector com fine-tuning no domínio (E2) atinge mAP\@0.5 de 0.938 na validação e
0.871 no round gold held-out, com recall de 0.91, contra 0.003 do YOLOv8s
pré-treinado em COCO sem fine-tuning (E3). A queda de duas ordens de grandeza no
baseline confirma que o domínio exige treino específico: os robôs não correspondem a
nenhuma classe COCO. O recall alto no gold mostra que o detector encontra ambos os
robôs de forma consistente, dado treino com rótulos completos.

#figure(
  caption: [Acurácia do detector. E2 é o experimento principal; E3 é o baseline sem fine-tuning. mAP no round gold held-out, revisado manualmente.],
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Config.*], [*mAP\@.5 (val)*], [*mAP\@.5 (gold)*], [*mAP\@.5:.95 (gold)*], [*Recall (gold)*]),
    [E2 YOLOv8s fine-tuned], [0.938], [0.871], [0.527], [0.907],
    [E3 YOLOv8s COCO], [---], [0.003], [0.001], [0.364],
  ),
) <tab-detector>

== Viabilidade

A pipeline completa (decodificação, detecção do dohyo, YOLO, OC-SORT, métricas e
eventos) roda a 93 quadros por segundo em batch 1, com pico de 82 MB de memória de
GPU alocada pelo detector. Em contraste, o SAM 3 usado como anotador roda a cerca de
2 quadros por segundo sobre entrada decimada de 480×270, com cerca de 7 GB de VRAM ---
confirmando a decisão de usá-lo apenas como anotador (E1), nunca na inferência final.

#figure(
  caption: [Viabilidade em RTX 4070 Laptop 8 GB. Pipeline de inferência vs SAM 3 como anotador.],
  table(
    columns: 3,
    align: (left, center, center),
    stroke: 0.4pt,
    table.header([*Componente*], [*FPS*], [*VRAM*]),
    [Pipeline E2 (inferência)], [93], [82 MB],
    [SAM 3 (anotador, E1)], [~2], [~7 GB],
  ),
) <tab-viability>

== Rastreamento, métricas e eventos

#figure(
  image("/results/figures/gold_zb01_trajectories.png", width: 65%),
  caption: [Trajetórias dos dois robôs no round gold, no referencial centrado no dohyo (círculo de raio 77 cm). A projeção por quadro cancela o movimento da câmera de mão.],
) <fig-traj>

Qualitativamente, o rastreamento mantém identidades A e B consistentes ao longo do
round held-out (@fig-traj), com o detector cinemático projetando posição e
velocidade no referencial do dohyo (velocidade máxima observada da ordem de 2 m/s,
coerente com a modalidade). A detecção de início de round dispara de forma
confiável; a detecção de ring-out e de primeiro contato ainda requer calibração dos
limiares sobre o gold, conforme previsto no protocolo. As métricas quantitativas de
rastreamento (IDF1, HOTA, ID switches) dependem de um gold com identidades anotadas
quadro a quadro, próximo passo desta linha; serão reportadas na sequência. A troca de identidade em oclusão prolongada entre robôs idênticos
é a limitação esperada do rastreador motion-only, discutida a seguir.
