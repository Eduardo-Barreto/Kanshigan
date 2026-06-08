= Resultados Parciais

Esta seção reporta os resultados medidos sobre footage real até a pré-banca. Todos
os números são medidos, não projetados. O conjunto de avaliação é pequeno (dois rounds
held-out, um por fonte, anotados e revisados manualmente), então os valores indicam
ordem de grandeza e viabilidade, não um modelo definitivo (ver limitações na
@sec-discussao).
As medições de hardware usaram uma RTX 4070 Laptop de 8 GB.

== Conjunto de dados

O conjunto é multi-fonte, atendendo à restrição de qualidade heterogênea (C3): footage
brasileiro da IRONCup 2025 (câmera de mão, ângulo oblíquo) e footage de torneio
regional japonês (câmera fixa cenital). Os vídeos são recortados em rounds individuais
(rastreamento e eventos só fazem sentido dentro de um round). A divisão é por
round/clip, não por quadro, para evitar vazamento entre quadros vizinhos quase
idênticos. Para treino e validação mantemos apenas os quadros em que o anotador
produziu rótulos completos (os dois robôs), descartando quadros incompletos que
ensinariam o detector a ignorar um robô. O gold tem um round por fonte, mantido fora
do treino e revisado manualmente quadro a quadro.

#figure(
  caption: [Composição do conjunto multi-fonte, após filtrar quadros de rótulo completo. Treino e validação anotados via SAM 3; gold revisado manualmente.],
  table(
    columns: 5,
    align: (left, center, center, center, left),
    stroke: 0.4pt,
    table.header([*Subconjunto*], [*Clips*], [*BR*], [*JP*], [*Anotação*]),
    [Treino], [14], [423], [202], [SAM 3, quadros completos],
    [Validação], [2], [59], [15], [SAM 3, quadros completos],
    [Gold (teste)], [2], [59], [57], [SAM 3 + revisão manual],
  ),
)

== Detecção

O detector com fine-tuning no domínio (E2), treinado nas duas fontes e operando sobre
o recorte do dohyo, atinge mAP\@0.5 de 0.985 no gold brasileiro e 0.976 no gold
japonês held-out, contra 0.026 do YOLOv8s pré-treinado em COCO sem fine-tuning (E3) ---
duas ordens de grandeza abaixo, confirmando que o domínio exige treino específico. Que
um único detector alcance mAP acima de 0.97 em ambas as fontes, apesar das câmeras
opostas (mão oblíqua vs cenital fixa), é a evidência central de que a pipeline atende à
heterogeneidade de qualidade (C3). O recorte no dohyo foi decisivo: sem ele, no quadro
inteiro, o detector caía para recall 0.91 e precisão 0.71 (perdia robôs em movimento e
gerava falso positivo no fundo); ampliar os robôs e remover o fundo levou ambos a 0.98.

#figure(
  caption: [Acurácia do detector multi-fonte sobre o recorte do dohyo, por fonte, no gold held-out revisado manualmente. E3 é o baseline COCO sem fine-tuning (gold BR).],
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Config. / fonte*], [*mAP\@.5*], [*mAP\@.5:.95*], [*Precisão*], [*Recall*]),
    [E2 fine-tuned --- gold BR], [0.985], [0.781], [0.990], [0.982],
    [E2 fine-tuned --- gold JP], [0.976], [0.695], [0.991], [0.915],
    [E3 COCO --- gold BR], [0.026], [0.017], [0.031], [0.745],
  ),
) <tab-detector>

#figure(
  image("/results/figures/qualitative_br_jp.png", width: 100%),
  caption: [Saída da pipeline em footage real das duas fontes: arena (amarelo) e robôs A (verde) e B (laranja) detectados e rastreados. À esquerda, BR (câmera de mão); à direita, torneio JP (câmera cenital fixa).],
) <fig-qualitative>

== Viabilidade

A pipeline completa (decodificação, detecção do dohyo, YOLO, OC-SORT, métricas e
eventos) roda a 133 quadros por segundo em batch 1, com pico de 82 MB de memória de
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
    [Pipeline E2 (inferência)], [133], [82 MB],
    [SAM 3 (anotador, E1)], [~2], [~7 GB],
  ),
) <tab-viability>

== Rastreamento, métricas e eventos

#figure(
  image("/results/figures/gold_zb01_trajectories.png", width: 65%),
  caption: [Trajetórias dos dois robôs no round gold, no referencial centrado no dohyo (círculo de raio 77 cm). A projeção por quadro cancela o movimento da câmera de mão.],
) <fig-traj>

Contra um gold com identidades anotadas e revisadas manualmente, o OC-SORT atinge
IDF1 de 0.93 e MOTA de 0.88, com uma única troca de identidade no round. O detector
cinemático projeta posição e velocidade no referencial do dohyo (@fig-traj),
com velocidade máxima da ordem de 3 m/s, coerente com a modalidade.

#figure(
  caption: [Rastreamento contra o gold com identidades, no round held-out.],
  table(
    columns: 4,
    align: (left, center, center, center),
    stroke: 0.4pt,
    table.header([*Tracker*], [*MOTA*], [*IDF1*], [*ID switches*]),
    [OC-SORT], [0.88], [0.93], [1],
  ),
) <tab-tracking>

A única troca de identidade ocorre na aproximação entre os dois robôs idênticos: é a
limitação esperada do rastreador motion-only, discutida a seguir. A detecção de
início de round dispara de forma confiável; a de ring-out e primeiro contato ainda
requer calibração dos limiares sobre o gold, conforme previsto no protocolo.
