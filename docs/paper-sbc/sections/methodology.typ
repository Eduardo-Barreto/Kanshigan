= Materiais e Métodos <sec-metodologia>

== Desenho do estudo

A pesquisa é aplicada, quantitativa e experimental. Não propomos detector nem rastreador novo: comparamos componentes existentes sobre vídeo real de torneio, contra um conjunto de referência revisado por humanos. A pergunta de pesquisa se decompõe em dois fatores manipulados. O primeiro é a arquitetura do detector: o YOLOv8s @jocher2023yolo (11,1 M de parâmetros, 28,4 GFLOPs) contra o compacto YOLO26n @jocher2026yolo26 (2,4 M, 5,2 GFLOPs), treinados com divisão, semente e protocolo idênticos. O segundo é o algoritmo de rastreamento: quatro rastreadores aplicados às mesmas detecções, de modo que a diferença medida seja do rastreador, não do detector. As variáveis dependentes são a acurácia e o custo de execução, porque a pergunta busca o equilíbrio entre os dois.

Os experimentos se organizam em seis pontos (@tab-experimentos). O custo de execução é medido sobre as configurações de E2 e E4.

#figure(
  caption: [Pontos experimentais e o que cada um isola.],
  table(
    columns: (auto, 1fr, 1fr),
    align: (center, left, left),
    stroke: 0.4pt,
    table.header([*ID*], [*Configuração*], [*Pergunta*]),
    [E1], [SAM 3 contra o conjunto de referência], [O anotador concorda com a revisão humana?],
    [E2], [YOLOv8s e YOLO26n com ajuste fino no domínio], [Quanto a capacidade do detector importa?],
    [E3], [YOLOv8s com pesos COCO, sem ajuste fino], [O domínio exige treino específico?],
    [E4], [Quatro rastreadores × dois detectores, detecções fixas], [A aparência melhora a identidade? A que custo?],
    [E5], [YOLOv8s com e sem 14 rounds JP extras, três sementes], [Mais anotação melhora o detector?],
    [E6], [Avaliação sem rótulos e ablação da arena], [Onde a pipeline falha fora do conjunto de referência?],
  ),
) <tab-experimentos>

== Pipeline

A pipeline segue o paradigma de rastreamento por detecção (_tracking-by-detection_): um detector localiza cada robô por uma caixa delimitadora em cada quadro, e um rastreador liga as caixas no tempo (@fig-pipeline). A inferência roda sobre todos os quadros, na taxa nativa do vídeo, porque os eventos decisivos duram poucos quadros. Cada estágio grava uma saída inspecionável, o que permite avaliar os componentes em separado.

#figure(
  image("/docs/paper-sbc/figures/pipeline.svg", width: 100%),
  caption: [Estágios da pipeline. A primeira fileira prepara o quadro; a segunda extrai a informação. O ramo tracejado é a anotação offline com o SAM 3, que constrói o conjunto de treino e fica fora da inferência.],
) <fig-pipeline>

*Arena e calibração.* O _dohyo_ tem geometria fixada por regulamento: disco escuro de 154 cm de diâmetro com borda branca de 5 cm @fujisoft2024rules. Por isso a arena é detectada por visão clássica, sem modelo treinado. Limiarizamos a luminância para isolar a borda branca, limpamos o resultado com morfologia e ajustamos elipses aos contornos. Um escore que combina tamanho, centralidade e razão de aspecto escolhe a elipse; nos quadros em que o ajuste falha, reusamos a última elipse válida. Um círculo visto em perspectiva oblíqua projeta-se como elipse, cujo eixo maior corresponde ao diâmetro real. A razão entre o diâmetro real e o eixo maior em pixels fixa a escala em centímetros por pixel de cada quadro, sem calibração de câmera nem marcadores. A escala isotrópica ignora o escorço da perspectiva, limitação que E6 quantifica.

*Recorte e detecção.* Os robôs ocupam pequena fração do quadro. Reduzido à entrada de 640 px do detector, o quadro inteiro deixa os robôs com poucos pixels, que o borrão de movimento apaga, e o fundo (plateia, mãos de operadores) gera falsos positivos. Recortamos então um retângulo fixo por clipe, com margem de 20% em torno da elipse mediana de 12 quadros amostrados. No round BR de referência, o recorte descarta 62% da área do quadro e amplia os robôs em cerca de 1,2 vez na entrada do detector; as caixas voltam às coordenadas nativas por um deslocamento. O detector YOLO roda sobre o recorte. Descartamos detecções com centro fora da elipse da arena ampliada em 15% e mantemos no máximo dois robôs por quadro.

*Rastreamento e identidade.* Comparamos dois rastreadores de movimento e dois de aparência, implementados na biblioteca BoxMOT. O OC-SORT @cao2023ocsort corrige o filtro de Kalman com as observações recentes, o que favorece trajetórias não lineares (C1). O ByteTrack @zhang2022bytetrack associa também as detecções de baixa confiança, comuns sob borrão. O DeepOCSORT @maggiolino2023deepocsort e o BoT-SORT @aharon2022botsort somam um descritor de reidentificação por aparência (ReID), o OSNet @zhou2019osnet treinado em pedestres, sem ajuste ao domínio. Os dois últimos testam se a aparência acrescenta informação quando os alvos são quase idênticos (C2). Os rastreadores emitem identificadores arbitrários. Chamamos de A o robô mais à esquerda no primeiro quadro com os dois. Mantemos só as duas trajetórias mais longas, o que deixa lacunas quando uma trajetória se fragmenta.

*Cinemática e eventos.* A pipeline projeta cada robô no referencial do _dohyo_, em centímetros e relativo ao centro da arena. Como a arena é fixa no mundo, a posição relativa cancela a translação da câmera de mão. Um filtro de Savitzky-Golay @savitzky1964smoothing suaviza a trajetória antes de derivar velocidade e aceleração, porque diferenças finitas diretas amplificariam o ruído de detecção. Regras determinísticas marcam três eventos. O início do round é o primeiro quadro em que qualquer um dos robôs supera 15 cm/s por três quadros seguidos, ignorando os dois primeiros quadros, onde o filtro tem artefato de borda. O primeiro contato exige, no mesmo quadro, distância abaixo de 30 cm e variação de velocidade acima de 40 cm/s. A saída do _dohyo_ (_ring-out_) ocorre quando o centro de um robô ultrapassa o raio da arena. Os limiares são valores iniciais fixados a priori, não calibrados. Este artigo não valida a cinemática nem os eventos; @sec-discussao discute o que essa validação exige.

== Dados e anotação

Não existe conjunto público do domínio. Construímos o nosso a partir de duas fontes de vídeo de torneio com regimes de câmera opostos (C4). A fonte brasileira (BR) reúne partidas da IRONCup 2025 filmadas com câmera de mão, em ângulo oblíquo, a 60 quadros por segundo. A fonte japonesa (JP) é um vídeo de 412 s de um torneio regional, gravado por câmera cenital fixa. Segmentamos os vídeos em rounds, cada um num clipe, e dividimos treino, validação e teste por round, não por quadro: quadros vizinhos são quase idênticos, e dividir por quadro vazaria informação. O teste é, portanto, reservado por round, não por torneio. O round de teste JP sai do mesmo vídeo do treino JP, e o BR, do mesmo torneio do treino BR.

Anotar caixas à mão custaria horas por round. Usamos o SAM 3 @carion2025sam3 como anotador semiautomático. Seguimos a prática de usar um modelo grande, de vocabulário aberto, como anotador, e um detector compacto, treinado nessas anotações, na inferência @liu2024groundingdino @ravi2024sam2. O SAM 3 propaga máscaras em vídeo a partir de um conceito textual; usamos o conceito "toy", escolhido num teste inicial com dez conceitos. Convertemos cada máscara em caixa pela maior componente conexa e escalamos as caixas para a resolução nativa. O anotador recebe o recorte da arena reduzido a 960 px de largura e a 10 quadros por segundo. Para caber em 8 GB de memória de GPU, processa janelas de 60 quadros e reinicia o _prompt_ em cada uma. Os robôs JP, pequenos e pretos em vista cenital, recebiam escore baixo para o conceito e ficavam abaixo do limiar padrão. Baixar o limiar para 0,15 e descartar caixas fora da arena recuperou os dois robôs. Os rótulos de treino não foram revisados à mão; mantivemos só os quadros com os dois robôs. Os rótulos JP de treino foram gerados antes de uma correção do filtro de caixas que, em rounds anotados depois, trocava um robô por um tripé fora da arena; não medimos esse efeito no treino original.

O conjunto de referência (_gold_) tem um round por fonte, fora do treino. O SAM 3 propôs as caixas, e nós as revisamos à mão em cada quadro amostrado a 10 Hz: 59 quadros com 112 caixas no BR e 57 quadros com 82 caixas no JP. Como a revisão parte da proposta do SAM 3, a concordância medida em E1 é um limite superior. No round BR, uma referência de identidade marca qual robô é A e qual é B nos mesmos 59 quadros (118 caixas, incluindo 6 que ficam fora do recorte), o que permite avaliar o rastreamento. A @tab-dataset resume a composição.

#figure(
  caption: [Composição do conjunto, em quadros anotados (amostrados a 10 Hz). A coluna Rounds soma BR e JP. Treino e validação incluem só quadros com os dois robôs rotulados, para não ensinar o detector a ignorar um deles; o conjunto de referência inclui todos os quadros amostrados do round.],
  table(
    columns: 5,
    align: (left, center, center, center, left),
    stroke: 0.4pt,
    table.header([*Subconjunto*], [*Rounds*], [*Quadros BR*], [*Quadros JP*], [*Anotação*]),
    [Treino], [14], [423], [202], [SAM 3],
    [Validação], [2], [59], [15], [SAM 3],
    [Referência (teste)], [2], [59], [57], [SAM 3 + revisão manual],
  ),
) <tab-dataset>

== Treino e avaliação

Cada detector parte dos pesos COCO @lin2014coco da sua arquitetura e treina com a mesma divisão, entrada de 640 px, aumento de dados padrão do Ultralytics 8.4.55, lote de 16, até 100 épocas com parada antecipada após 20 épocas sem melhora na validação, e semente 42. Congelamos os detectores reportados antes da avaliação final. O recorte da arena, porém, decidimos durante o desenvolvimento, observando falhas do detector no quadro inteiro.

Na detecção, medimos a precisão média (mAP) com limiar de sobreposição (IoU) de 0,5 e a média entre 0,5 e 0,95, além de precisão e revocação no ponto de F1 máximo, como reporta o Ultralytics. No rastreamento, medimos o MOTA @bernardin2008clear, que soma falsos positivos, falsos negativos e trocas de identidade; o IDF1 @ristani2016idf1, que mede por quanto tempo cada robô mantém a identidade certa; e o HOTA @luiten2021hota, que equilibra detecção e associação. No custo, medimos quadros por segundo e pico de memória de GPU em um notebook com RTX 4070 Laptop de 8 GB. Reportamos cada número por fonte e sempre medido, nunca projetado; salvo indicação, cada um vem de um arquivo em `results/`.

Hiperparâmetros, versões (PyTorch 2.12, Ultralytics 8.4.55, BoxMOT 19.0.0) e scripts ficam versionados no repositório. O protocolo de E1 a E4 foi registrado num diário de bordo antes da execução; E5 e E6 vieram depois, em resposta à banca de qualificação. O diário documenta também os desvios, as falhas e as correções.
