= Resultados <sec-resultados>

O conjunto de referência é pequeno: um round por fonte, e um único round com identidades. Por isso tratamos diferenças pequenas como ruído e dizemos, em cada caso, o que a amostra sustenta.

== Validação do anotador (E1)

O SAM 3 anota o treino, então precisa concordar com o que a revisão humana aceita. Comparamos suas caixas com as do conjunto de referência por IoU ≥ 0,5 (@tab-annotator). No round BR, o modo de imagem com o conceito "robot" atinge F1 de 0,96, com IoU médio de 0,91 nas caixas casadas; o modo de vídeo com o conceito "toy", que anotou o treino BR, não foi medido nesse round. No JP, em vista cenital, o modo de imagem quase não acha os robôs pretos e pequenos (revocação de 0,16). O modo de vídeo, que propaga as máscaras no tempo, recupera os dois robôs: revocação de 0,94 e F1 de 0,91 no round de referência JP.

Os 14 rounds JP extras de E5 dão uma medida maior do modo de vídeo. Em 448 quadros e 878 caixas revisadas, ele atinge F1 de 0,75. As caixas que ele propõe quase coincidem com as humanas (IoU de 0,94), mas cerca de três em cada dez robôs ficam sem caixa, e a revisão alterou 71% dos quadros. A revisão, portanto, não foi acabamento: foi o que tornou esses rounds utilizáveis. Os rótulos do treino reportado não passaram por essa revisão, e a qualidade deles segue desconhecida.

#figure(
  caption: [Concordância do SAM 3 com a revisão humana (IoU ≥ 0,5). Toda revisão partiu da proposta do SAM 3, então os valores são limites superiores. Referência BR: modo de imagem com o conceito "robot"; as demais linhas usam "toy" com limiar 0,15. O modo de vídeo anotou o treino das duas fontes. Os 14 rounds foram anotados depois de uma correção que ordena as caixas pela distância ao centro da arena, e não medem os rótulos JP originais.],
  table(
    columns: 6,
    align: (left, left, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Conjunto*], [*Modo*], [*Precisão*], [*Revocação*], [*F1*], [*IoU médio*]),
    [Referência BR], [imagem], [0,98], [0,94], [0,96], [0,91],
    [Referência JP], [imagem], [1,00], [0,16], [0,27], [0,92],
    [Referência JP], [vídeo], [0,88], [0,94], [0,91], [0,98],
    [14 rounds JP], [vídeo], [0,80], [0,71], [0,75], [0,94],
  ),
) <tab-annotator>

== Detecção (E2 e E3) <sec-deteccao>

Os dois detectores com ajuste fino atingem mAP\@0,5 acima de 0,96 nas duas fontes (@tab-detector). O YOLO26n, com um quinto dos parâmetros, fica em 0,968 contra 0,965 no BR e em 0,993 contra 0,976 no JP. Para dar escala a essas diferenças, o YOLOv8s treinado com três sementes varia 0,019 (desvio padrão) no JP (@sec-escala-jp). A diferença entre arquiteturas é da mesma ordem. O conjunto sustenta, portanto, equivalência prática, não superioridade: a capacidade extra do YOLOv8s não rende acurácia mensurável neste domínio. O YOLO26n foi treinado com uma única semente.

O ajuste fino no domínio, por outro lado, é decisivo. Avaliamos o YOLOv8s com pesos COCO (E3) aceitando qualquer classe COCO como robô: ele atinge 0,757 no BR e 0,359 no JP. Na avaliação padrão de classe única, que compara só a classe 0 do COCO (pessoa), fica em 0,026 e 0,062. O detector genérico acha parte dos robôs como algum objeto, mas falha justamente na vista cenital. O treino no domínio leva os dois casos acima de 0,96.

#figure(
  caption: [Detecção no round de referência de cada fonte, sobre o recorte da arena. Precisão e revocação no limiar de confiança de F1 máximo, como reporta o Ultralytics. YOLOv8s (COCO): sem ajuste fino, com qualquer classe aceita como robô.],
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Detector*], [*Fonte*], [*mAP\@0,5*], [*mAP\@0,5:0,95*], [*Precisão*], [*Revocação*]),
    [YOLOv8s], [BR], [0,965], [0,767], [0,990], [0,964],
    [YOLOv8s], [JP], [0,976], [0,695], [0,991], [0,915],
    [YOLO26n], [BR], [0,968], [0,772], [0,963], [0,955],
    [YOLO26n], [JP], [0,993], [0,691], [0,986], [0,976],
    [YOLOv8s (COCO)], [BR], [0,757], [0,570], [0,764], [0,693],
    [YOLOv8s (COCO)], [JP], [0,359], [0,252], [0,398], [0,463],
  ),
) <tab-detector>

A precisão alta da @tab-detector depende do limiar. Com confiança fixa de 0,25, a que alimenta o rastreador, o YOLOv8s marca 30 falsos positivos nos 57 quadros JP (precisão de 0,72). Dezessete deles caem fora da arena e são, quase todos, o robô já fora do _dohyo_, que o treino JP rotula e a referência JP não. O filtro da arena que a pipeline aplica remove essas caixas e leva a precisão a 0,86; restam 13 falsos positivos dentro da arena. No YOLO26n, o filtro leva a precisão de 0,86 a 0,98; no BR, não muda nada. A parte fora da arena vem da convenção de anotação JP, não do detector.

== Rastreamento (E4)

Aplicamos os quatro rastreadores às mesmas detecções de cada detector, no round BR com identidades (@tab-tracking). As métricas usam só os 59 quadros anotados#footnote[Seis das 118 caixas de identidade ficam fora do recorte, porque a referência de identidade foi gerada antes de uma correção do recorte; nenhum rastreador pode acertá-las. Avaliando só as 112 caixas visíveis, o OC-SORT sobre o YOLOv8s sobe de MOTA 0,881 para 0,929 e de HOTA 0,810 para 0,853, sem alterar a ordem entre rastreadores. Nas trajetórias cruas, antes da convenção A/B, três combinações chegam a duas trocas e o IDF1 do OC-SORT sobre o YOLOv8s cai de 0,933 para 0,925; a ordem entre rastreadores não muda.].

#figure(
  caption: [Grade detector × rastreador no round BR com identidades (59 quadros a 10 Hz, 118 caixas). Trocas: trocas de identidade. FPS: pipeline completa sobre o recorte, em GPU, com decodificação; média de três execuções, desvio de no máximo 3,4 FPS.],
  table(
    columns: 8,
    align: (left, left, left, center, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Detector*], [*Rastreador*], [*Tipo*], [*MOTA*], [*IDF1*], [*HOTA*], [*Trocas*], [*FPS*]),
    [YOLOv8s], [OC-SORT], [movimento], [0,881], [0,933], [0,810], [1], [155],
    [YOLOv8s], [ByteTrack], [movimento], [0,898], [0,947], [0,824], [0], [151],
    [YOLOv8s], [DeepOCSORT], [aparência], [0,881], [0,933], [0,810], [1], [57],
    [YOLOv8s], [BoT-SORT], [aparência], [0,907], [0,952], [0,833], [0], [61],
    [YOLO26n], [OC-SORT], [movimento], [0,881], [0,933], [0,796], [1], [128],
    [YOLO26n], [ByteTrack], [movimento], [0,890], [0,938], [0,804], [1], [130],
    [YOLO26n], [DeepOCSORT], [aparência], [0,856], [0,919], [0,774], [1], [54],
    [YOLO26n], [BoT-SORT], [aparência], [0,890], [0,938], [0,804], [1], [57],
  ),
) <tab-tracking>

As oito combinações sustentam a identidade, com no máximo uma troca em 118 caixas. As diferenças de acurácia cabem em uma ou duas caixas: o BoT-SORT sobre o YOLOv8s lidera por uma margem que um único round não separa de ruído. A diferença de custo, ao contrário, é grande e não depende do tamanho da amostra. O passo de ReID derruba a pipeline inteira de 128 a 155 para 54 a 61 FPS. Medido sem o detector, o passo de rastreamento fica de 31 a 35 vezes mais lento com ReID. Rastreadores só de movimento entregam a mesma identidade por uma fração do custo.

*Por que a aparência não ajuda?* Esperávamos que a resposta fosse C2: robôs parecidos demais para o ReID separar. Medimos essa premissa e ela não se sustenta neste round. Extraímos descritores de 111 recortes do round (os visíveis com lado de ao menos 12 px) e comparamos a similaridade cosseno entre recortes do mesmo robô e de robôs diferentes. Com o OSNet que os rastreadores usam, a média intra-robô é 0,85 e a inter-robô 0,54, com área sob a curva ROC (AUC) de 0,94 na separação; o vizinho mais próximo acerta a identidade em 97% dos recortes. Até um histograma de cor separa os dois robôs (AUC de 0,92). A explicação que os dados sustentam é outra: o movimento sozinho já resolve este round, e não sobra erro de identidade para a aparência corrigir. Sobre o YOLOv8s, o DeepOCSORT produz trajetórias quase idênticas às do OC-SORT (697 de 698 linhas iguais): o ReID quase nunca muda uma associação. Este round, portanto, não estressa C2, e um confronto entre robôs de fato indistinguíveis segue sem medição.

== Custo de execução <sec-custo>

A pipeline completa, da decodificação aos eventos, roda em tempo real sobre o vídeo nativo em GPU de notebook (@tab-viability). Em 1080p a 60 FPS, ela processa 97 FPS com o YOLOv8s e 89 com o YOLO26n, diferença que as execuções individuais não separam. No recorte, o YOLOv8s é mais rápido (155 contra 128 FPS), apesar de ter quase cinco vezes mais parâmetros: na GPU, o custo dominante não é o tamanho da rede. Na CPU do mesmo notebook, sem GPU, a ordem se inverte: o YOLO26n roda a 40 FPS e o YOLOv8s a 21. A análise pós-partida segue viável na CPU, abaixo do tempo real. Em vídeo 4K com codificação HEVC (H.265), a decodificação vira o gargalo e a pipeline cai para 16 FPS. Um processo isolado ocupa de 250 a 272 MiB de memória de GPU, contexto CUDA incluído, menos de 4% dos 8 GB da placa; com ReID, até 334 MiB. Durante a anotação, registramos no diário de bordo que o SAM 3 processava cerca de 2 FPS e ocupava cerca de 7 GB, o que justifica mantê-lo fora da inferência.

#figure(
  caption: [Custo da pipeline completa (decodificação, arena, recorte, detecção, OC-SORT, cinemática e eventos) no round BR de referência; média e desvio de três execuções aquecidas; a linha 4K usa um round RC. Notebook com Intel Core Ultra 9 185H e RTX 4070 Laptop de 8 GB. Memória: pico medido pelo NVML, a biblioteca de monitoramento da NVIDIA, na primeira execução de um processo com um só modelo, contexto CUDA incluído.],
  table(
    columns: 4,
    align: (left, left, center, center),
    stroke: 0.4pt,
    table.header([*Entrada*], [*Dispositivo*], [*YOLOv8s*], [*YOLO26n*]),
    [1080p nativo], [GPU], [97,0 ± 7,0 FPS], [88,9 ± 2,2 FPS],
    [1080p nativo], [CPU], [21,3 ± 1,7 FPS], [39,6 ± 2,3 FPS],
    [Recorte (1558 × 506)], [GPU], [154,5 ± 2,2 FPS], [127,7 ± 1,2 FPS],
    [4K HEVC (RC)], [GPU], [16,1 ± 2,6 FPS], [16,5 ± 1,4 FPS],
    [Memória de GPU], [GPU], [250 MiB], [272 MiB],
    [Pesos do detector], [disco], [22,5 MB], [5,4 MB],
  ),
) <tab-viability>

== Escala do treino JP (E5) <sec-escala-jp>

O treino JP usa 8 dos 26 rounds que a segmentação encontra no vídeo de origem. Para medir se a quantidade de anotação limita o detector, anotamos com o SAM 3 e revisamos à mão outros 14 rounds do mesmo vídeo (448 quadros). O treino JP passou de 202 para 650 quadros. Retreinamos o YOLOv8s com os mesmos hiperparâmetros e três sementes, sem mudar os conjuntos de referência (@tab-escala).

#figure(
  caption: [YOLOv8s sem e com os 14 rounds JP extras, sementes 42, 0 e 1. mAP como média ± desvio. Contagens por semente, com confiança mínima de 0,25 e IoU ≥ 0,5 (112 robôs no BR, 82 no JP). Perdidos: falsos negativos. FP dentro e FP fora: falsos positivos dentro e fora da arena, antes do filtro da pipeline.],
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Avaliado / treino*], [*mAP\@0,5*], [*mAP\@0,5:0,95*], [*Perdidos*], [*FP dentro*], [*FP fora*]),
    [BR / original], [0,968 ± 0,006], [0,786 ± 0,018], [4, 4, 4], [1, 28, 1], [0, 0, 0],
    [BR / + 14 JP], [0,957 ± 0,032], [0,728 ± 0,047], [3, 4, 3], [1, 70, 1], [0, 1, 1],
    [JP / original], [0,963 ± 0,019], [0,669 ± 0,023], [5, 5, 10], [13, 1, 0], [17, 20, 9],
    [JP / + 14 JP], [0,978 ± 0,009], [0,717 ± 0,020], [0, 2, 1], [0, 21, 1], [54, 37, 47],
  ),
) <tab-escala>

O mAP\@0,5 não separa as configurações: os intervalos se sobrepõem nas duas fontes. O que muda é o tipo de erro. Na referência JP, o detector com mais dado perde menos robôs nas três sementes (de 5 a 10 para 0 a 2, em 82), inclusive robôs borrados e na borda do quadro, e ajusta melhor as caixas (mAP\@0,5:0,95 de 0,703 a 0,740, contra 0,652 a 0,695). O ganho tem custo na outra fonte. Na referência BR, o mAP\@0,5:0,95 cai nas três sementes, e uma delas fragmenta um robô em caixas parciais. Os quadros novos invertem a proporção do treino (423 BR para 650 JP); o desbalanceamento é a explicação provável, mas não a testamos isoladamente. Os falsos positivos dentro da arena variam mais entre sementes que entre configurações: com a semente 0, um robô se parte em duas ou três caixas parciais nas duas configurações. Como os rounds novos também foram revisados à mão e incluem 18 quadros com um único robô, que o treino original excluía, o experimento não separa o efeito de mais dado do efeito de dado revisado. As detecções fora da arena triplicam no JP e são, nos quadros inspecionados, o robô já fora do _dohyo_, a mesma divergência de convenção da @sec-deteccao. Ampliar a anotação de uma única fonte, portanto, teve custo: melhorou a fonte ampliada e piorou a outra. Mantemos como detector reportado o treinado no conjunto original.

== Avaliação sem rótulos (E6) <sec-sem-rotulos>

Fora dos dois rounds de referência não há rótulo. O domínio, porém, impõe uma restrição a toda saída correta: durante o round, há exatamente dois robôs no _dohyo_. Contamos a fração de quadros que viola essa restrição, entre o início e o fim do round que a própria pipeline marca, e ancoramos o indicador no round BR, onde a acurácia é conhecida (IDF1 de 0,93 e 6,3% de violação). O indicador conta robôs; não confere se as caixas estão certas. Nos clipes avaliados, ele coincide com a avaliação qualitativa registrada antes da medição: 44% num vídeo BR de baixa resolução e 100% num trecho da grande final do All Japan Robot-Sumo Tournament 2025. Dois clipes de execuções anteriores, cujo vídeo de origem não recuperamos, ficam em 0% (round JP de câmera fixa) e 54% (outro trecho da final).

A mesma pipeline, sem retreino, rodou nos 23 rounds de Sumô de rádio-controle (RC) do acervo da equipe ThundeRatz, uma categoria fora do treino, com robôs de outro formato e vídeo de celular. A violação mediana é de 27%. Dez dos 23 rounds ficam no regime do round de referência ou melhor, três deles com janelas curtas, abaixo de 60 quadros, em que a taxa é instável; sete passam de 50%. É um indício de que o detector transfere para outra categoria em parte dos rounds, não uma medida de acurácia.

Na grande final japonesa, transmissão oficial com arena azul, placar sobreposto e colisões violentas, o modo de falha é visível (@fig-worlds). No movimento lento do início, o detector acha os dois robôs, e a arena de cor nova é detectada. No movimento rápido, o borrão extremo impede as duas detecções, e a segunda metade do round fica quase sem detecção. O SAM 3, sobre o recorte e com propagação temporal, ainda sustenta um dos robôs. O limite não é a semelhança entre robôs, que ali carregam bandeiras distintas, mas a falta de dados de treino com esse borrão.

#figure(
  image("/results/figures/worlds_model_vs_sam.png", width: 62%),
  caption: [Nosso detector (esquerda, quadro nativo) e SAM 3 (direita, recorte da arena) nos mesmos instantes do primeiro round da grande final do All Japan Robot-Sumo Tournament 2025, fora da distribuição de treino; clipe distinto dos dois trechos auditados. Em cima, movimento lento: ambos acham os dois robôs. Embaixo, movimento rápido: o detector perde os dois; o SAM 3 mantém um.],
) <fig-worlds>

O estágio de arena também foi medido sem rótulos. Como o _dohyo_ é fixo e tem diâmetro conhecido, um ajuste correto produz escala estável ao longo do round. Comparamos o seletor por escore com o de maior contorno branco, sobre os mesmos quadros de 30 clipes, considerando empate uma diferença menor que 0,01 no coeficiente de variação da escala. Os dois empatam em 17, o escore vence em 9 e perde em 4. Quando vence, a diferença pode ser de duas ordens de grandeza: num vídeo BR de câmera de mão, o coeficiente de variação vai de 0,44 para 0,002. O escore é indiferente com um único candidato brilhante e decisivo quando o fundo oferece concorrente, como na câmera de mão. A mesma medida mostra quanto a escala oscila. O coeficiente de variação é 0,001 na câmera fixa JP, de 0,002 a 0,064 nos três vídeos BR e 0,18 de mediana nos 24 clipes RC (os 23 rounds e um clipe de demonstração), chegando a 0,38. A oscilação mistura erro de ajuste com zoom e movimento real da câmera; o erro de escorço dentro do quadro não foi medido. Sob câmera de celular, a escala em centímetros não é confiável sem retificação, e toda velocidade herda esse erro.
