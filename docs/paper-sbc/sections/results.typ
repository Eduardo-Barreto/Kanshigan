= Resultados Parciais <sec-resultados>

Esta seção reporta os resultados medidos sobre footage real neste estado inicial do
projeto. Todos os números são medidos, não projetados. O conjunto de avaliação é
pequeno (dois rounds held-out, um por fonte, anotados e revisados manualmente), então
os valores indicam ordem de grandeza e viabilidade, não um modelo definitivo (ver
limitações na @sec-discussao). As medições de hardware usaram uma RTX 4070 Laptop de
8 GB.

== Conjunto de dados

O conjunto é multi-fonte, atendendo à condição de qualidade heterogênea (C4): footage
brasileiro da IRONCup 2025 (câmera de mão, ângulo oblíquo) e footage de torneio
regional japonês (câmera fixa cenital). Apenas essas duas fontes compõem treino,
validação e gold; o footage de Sumô RC e a final de mundial aparecem somente como
avaliação qualitativa out-of-distribution na @sec-discussao. Os vídeos são recortados em rounds individuais
(rastreamento e eventos só fazem sentido dentro de um round). A divisão é por
round/clip, não por quadro, para evitar vazamento entre quadros vizinhos quase
idênticos. Para treino e validação mantemos apenas os quadros em que o anotador
produziu rótulos completos (os dois robôs), descartando quadros incompletos que
ensinariam o detector a ignorar um robô. O gold tem um round por fonte, mantido fora
do treino e revisado manualmente quadro a quadro.

#figure(
  caption: [Composição do conjunto, em número de quadros por fonte (BR e JP), contando
  apenas quadros de rótulo completo (os dois robôs). A coluna Clips é o total de rounds
  (BR + JP) no subconjunto. A divisão é por round, não por quadro. Treino e validação
  são anotados pelo SAM 3; o gold é um round held-out por fonte, anotado pelo SAM 3 e
  revisado manualmente quadro a quadro.],
  table(
    columns: 5,
    align: (left, center, center, center, left),
    stroke: 0.4pt,
    table.header([*Subconjunto*], [*Clips*], [*Quadros BR*], [*Quadros JP*], [*Anotação*]),
    [Treino], [14], [423], [202], [SAM 3],
    [Validação], [2], [59], [15], [SAM 3],
    [Gold (teste)], [2], [59], [57], [SAM 3 + revisão manual],
  ),
) <tab-dataset>

== Detecção: comparação de arquiteturas

Para responder à parte de detecção da pergunta de pesquisa, comparamos duas
arquiteturas com fine-tuning no domínio, sobre o recorte do dohyo: o YOLOv8s
(11,1 M de parâmetros, 28,4 GFLOPs) e o YOLO26n, compacto (2,4 M de parâmetros,
5,2 GFLOPs). Ambos atingem mAP\@0.5 acima de 0.96 nas duas fontes (@tab-detector). O
YOLO26n, com cerca de um quinto dos parâmetros, fica em 0.968 contra 0.965 no gold BR
e 0.993 contra 0.976 no gold JP; com um único round held-out por fonte, diferenças
dessa ordem não são distinguíveis de ruído de amostragem, então o que o conjunto
sustenta é equivalência prática: a arquitetura compacta basta para o domínio e melhora
a viabilidade sem custo mensurável de acurácia. O YOLOv8s
pré-treinado em COCO sem fine-tuning (E3) fica em 0.026, duas ordens de grandeza
abaixo, o que confirma que o domínio exige treino específico.

Um detector acima de 0.96 nas duas fontes, apesar das câmeras opostas (mão oblíqua e
cenital fixa), sustenta a condição de qualidade heterogênea (C4). O recorte no dohyo
foi decisivo: sem ele, no quadro inteiro, o detector caía para recall 0.91 e precisão
0.71 (perdia robôs em movimento e gerava falso positivo no fundo); ampliar os robôs e
remover o fundo levou precisão a 0.99 e recall a 0.96.

#figure(
  caption: [Acurácia das duas arquiteturas de detector, com fine-tuning sobre o recorte
  do dohyo, no gold held-out por fonte. COCO é o YOLOv8s sem fine-tuning (baseline zero-shot,
  gold BR). YOLOv8s: 11,1 M parâmetros, 28,4 GFLOPs; YOLO26n: 2,4 M, 5,2 GFLOPs.],
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Detector / fonte*], [*mAP\@.5*], [*mAP\@.5:.95*], [*Precisão*], [*Recall*]),
    [YOLOv8s (gold BR)], [0.965], [0.767], [0.990], [0.964],
    [YOLOv8s (gold JP)], [0.976], [0.695], [0.991], [0.915],
    [YOLO26n (gold BR)], [0.968], [0.772], [0.963], [0.955],
    [YOLO26n (gold JP)], [0.993], [0.691], [0.986], [0.976],
    [COCO zero-shot (gold BR)], [0.026], [0.017], [0.032], [0.750],
  ),
) <tab-detector>

== Escala do treino JP: mais dado muda o erro, não o mAP <sec-escala-jp>

O treino JP usa 8 dos 26 trechos que a segmentação encontra no vídeo de origem. Para
medir se a quantidade de anotação é o gargalo, anotamos com o SAM 3 e revisamos
manualmente, quadro a quadro, outros 14 rounds do mesmo vídeo (448 quadros), o que
leva o treino JP de 202 para 650 quadros, e retreinamos o YOLOv8s com os mesmos
hiperparâmetros. O gold não muda. Como uma diferença de um ponto de mAP num gold de um
round não se distingue de ruído, repetimos as duas configurações com três sementes
(@tab-escala).

#figure(
  caption: [YOLOv8s treinado sem e com os 14 rounds JP adicionais, três sementes
  (42, 0 e 1) por configuração, nos golds que não mudam. mAP como média ± desvio
  padrão; as contagens são por semente, com confiança mínima 0.25 e casamento por
  IoU ≥ 0.5 contra o gold (112 robôs no BR, 82 no JP). "FP fora" são detecções fora
  da arena que não casam com nenhuma caixa do gold.],
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Gold / treino*], [*mAP\@.5*], [*mAP\@.5:.95*], [*Robôs perdidos*], [*FP fora*]),
    [BR / original], [0.968 ± 0.006], [0.786 ± 0.018], [4, 4, 4], [0, 0, 0],
    [BR / + 14 rounds JP], [0.957 ± 0.032], [0.728 ± 0.047], [3, 4, 3], [0, 1, 1],
    [JP / original], [0.963 ± 0.019], [0.669 ± 0.023], [5, 5, 10], [17, 20, 9],
    [JP / + 14 rounds JP], [0.978 ± 0.009], [0.717 ± 0.020], [0, 2, 1], [54, 37, 47],
  ),
) <tab-escala>

O mAP\@.5 não separa as configurações em nenhum dos golds: os intervalos se sobrepõem.
O que muda é a anatomia do erro. No gold JP, o detector com mais dado perde menos
robôs nas três sementes (de 5 a 10 perdidos para 0 a 2, em 82) e ajusta melhor as
caixas: o mAP\@.5:.95 das três sementes novas (0.703 a 0.740) fica acima das três
originais (0.652 a 0.695). Entre os robôs recuperados estão casos na borda do quadro
e borrados pelo movimento (@fig-escala).

O ganho tem custo na outra fonte. No gold BR, o mAP\@.5:.95 cai nas três sementes
(0.674 a 0.757 contra 0.767 a 0.804), e uma delas degrada de forma visível (mAP\@.5
0.920, recall 0.79), fragmentando um robô em caixas parciais. Os quadros novos são
todos JP e invertem a proporção do treino (423 BR para 650 JP); o desbalanceamento
entre fontes é a explicação provável, mas não a testamos isoladamente.

As detecções fora da arena triplicam no gold JP. Nos quadros inspecionados não são
alucinações: são o robô já ejetado (@fig-escala). O treino JP o rotula, tanto no conjunto original quanto nos rounds novos (cerca
de 17% das caixas ficam fora da elipse da arena nos dois), e o gold JP não (1 de 82).
O aumento é consistente com um detector que aprende melhor o que o treino já pedia. É
uma divergência de convenção de rótulo entre treino e gold, anterior a este experimento,
que precisa ser unificada antes de ser lida como erro do detector. Os falsos positivos dentro da arena, por sua vez, são dominados pela
semente (de 0 a 21 no JP, de 1 a 70 no BR) e não sustentam conclusão.

Três sementes indicam direção, não significância. Por isso mantemos como detector
reportado o da @tab-detector, treinado no conjunto original. Ficam dois resultados.
Primeiro, neste experimento, anotar mais de uma única fonte não foi ganho gratuito:
melhorou a fonte ampliada e cobrou da outra. Segundo, o desvio entre sementes (0.019 no mAP\@.5 do gold
JP) é da mesma ordem da diferença entre as duas arquiteturas na @tab-detector (0.017),
o que é compatível com a leitura de equivalência prática feita ali.

#figure(
  image("/results/figures/jp_data_scaling_cases.png", width: 100%),
  caption: [Dois quadros do gold JP, semente 42. Branco: gold; vermelho: detector
  original; verde: detector com os 14 rounds adicionais; o número é a confiança. Em
  cima, o original marca riscos do dohyo e perde o robô na borda. Embaixo, perde o
  robô borrado; o novo o encontra e também marca, no canto inferior, o robô já
  ejetado, que o gold não rotula.],
) <fig-escala>

== Validação do anotador: SAM 3 vs gold

Se o SAM 3 anota o treino, ele precisa concordar com o que um humano aceita. Medimos
isso rodando o SAM 3 sobre os mesmos quadros do gold revisado e comparando suas caixas
às do gold por IoU (@tab-annotator). No gold BR, o SAM 3 atinge precisão 0.98, recall
0.94 e F1 0.96, com IoU médio 0.91 nas caixas casadas: quando propõe uma caixa, ela é
espacialmente quase idêntica à do humano, e a revisão manual mexeu em poucos quadros.
No gold JP, em vista cenital, as caixas propostas seguem precisas (precisão 0.95, IoU
0.89), mas o predictor de imagem quadro a quadro detecta menos os robôs pretos
pequenos; é o predictor de vídeo, com propagação temporal e limiar reduzido, que
recupera os dois robôs na anotação de treino (@sec-discussao).

Os 14 rounds revisados da @sec-escala-jp dão uma segunda medida, maior e do próprio
predictor de vídeo: sobre 448 quadros e 878 caixas revisadas, ele atinge precisão
0.80, recall 0.71 e IoU médio 0.94 nas caixas casadas, e a revisão alterou 318 dos 448
quadros. A leitura é a mesma do gold BR na localização (a caixa proposta quase
coincide com a humana) e diferente na cobertura: o anotador deixa cerca de três em
cada dez robôs sem caixa, e é a revisão que os recupera. Como a revisão parte da
proposta do SAM 3, a concordância é um limite superior.

#figure(
  caption: [Concordância do SAM 3 como anotador com o gold revisado por humano, por IoU.
  No gold JP, o número é do predictor de imagem quadro a quadro; a anotação de treino
  usa o predictor de vídeo (propagação temporal), medido na última linha contra os 14
  rounds revisados (448 quadros).],
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Fonte*], [*Precisão*], [*Recall*], [*F1*], [*IoU médio*]),
    [Gold BR], [0.98], [0.94], [0.96], [0.91],
    [Gold JP (por quadro)], [0.95], [0.22], [0.36], [0.89],
    [JP, 14 rounds revisados (vídeo)], [0.80], [0.71], [0.75], [0.94],
  ),
) <tab-annotator>

#figure(
  image("/results/figures/qualitative_br_jp.png", width: 82%),
  caption: [Saída da pipeline em footage real das duas fontes: arena (amarelo) e robôs A (verde) e B (laranja) detectados e rastreados. À esquerda, BR (câmera de mão); à direita, torneio JP (câmera cenital fixa).],
) <fig-qualitative>

== Viabilidade

A pipeline completa (decodificação, detecção do dohyo, YOLO, OC-SORT, métricas e
eventos) roda acima de 100 quadros por segundo em batch 1 sobre o round gold BR (até
cerca de 130 aquecida; menos na partida fria, com o carregamento do modelo e a
inicialização do contexto CUDA dentro da medição). O custo de memória é pequeno: o
pico do processo (`torch.cuda.max_memory_reserved`) fica em cerca de 101 MB com o
YOLOv8s e 90 MB com o YOLO26n, longe dos 8 GB da GPU (@tab-viability). O SAM 3 como
anotador roda a cerca de 2 quadros por segundo sobre entrada decimada de 480×270, com
cerca de 7 GB de VRAM, o que confirma usá-lo apenas como anotador (E1), fora da
inferência.

#figure(
  caption: [Viabilidade em RTX 4070 Laptop 8 GB, no round gold. Pesos do detector, pico de VRAM alocada (tensores) e pico de VRAM reservada (processo). O pipeline de inferência ocupa cerca de 0,1 GB; o SAM 3, como anotador offline, cerca de 7 GB.],
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Componente*], [*FPS*], [*Pesos*], [*VRAM alocada*], [*VRAM reservada*]),
    [Pipeline E2 (YOLOv8s)], [>100], [44,5 MB], [82 MB], [101 MB],
    [Pipeline E2 (YOLO26n)], [>100], [10 MB], [66 MB], [90 MB],
    [SAM 3 (anotador, E1)], [~2], [---], [---], [~7 GB],
  ),
) <tab-viability>

== Rastreamento: comparação de algoritmos

Para a parte de rastreamento da pergunta de pesquisa, comparamos quatro rastreadores
sobre as *mesmas* detecções (detector YOLO26n fixo, round held-out), contra um gold
com identidades anotadas e revisadas manualmente (@tab-tracking). Dois são motion-only
(OC-SORT e ByteTrack) e dois usam aparência via ReID (DeepOCSORT e BoT-SORT). Fixar as
detecções isola o efeito do rastreador: qualquer diferença é dele, não do detector.

Os quatro sustentam identidade no caso típico, com no máximo uma troca, sempre na
aproximação entre os dois robôs idênticos. A aparência não traz ganho de acurácia: o
BoT-SORT empata com o ByteTrack (MOTA 0.89, IDF1 0.94) e o DeepOCSORT fica atrás dos
dois motion-only. O custo, porém, é grande: o passo de ReID derruba o throughput do
rastreador de mais de 3000 quadros por segundo para menos de 100, cerca de 35 a 40
vezes mais lento, sem retorno em acurácia (o porquê estrutural está na @sec-discussao).
A um único round held-out, a diferença entre os quatro é pequena demais para eleger um
vencedor isolado; o que se sustenta é a dominância do motion-only sobre o de aparência.

#figure(
  caption: [Quatro rastreadores sobre detecções idênticas (detector fixo), contra o gold com identidades no round held-out. FPS medido só no estágio de rastreamento, isolando o custo próprio do rastreador.],
  table(
    columns: 6,
    align: (left, left, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Tracker*], [*Tipo*], [*MOTA*], [*IDF1*], [*ID switches*], [*FPS*]),
    [OC-SORT], [movimento], [0.88], [0.93], [1], [3183],
    [ByteTrack], [movimento], [0.89], [0.94], [1], [3448],
    [DeepOCSORT], [aparência], [0.86], [0.92], [1], [81],
    [BoT-SORT], [aparência], [0.89], [0.94], [1], [94],
  ),
) <tab-tracking>

A pipeline também projeta posição e velocidade no referencial do dohyo e detecta
eventos por regras determinísticas, mas a validação quantitativa dessas saídas é
trabalho futuro, não resultado desta fase: das três classes de evento, só o início de
round dispara de forma confiável; ring-out e primeiro contato exigem um conjunto de
calibração de limiares separado do gold, que ainda não existe; e a velocidade em
centímetros não tem ground-truth físico e sofre com o movimento da câmera de mão
(@sec-discussao).
