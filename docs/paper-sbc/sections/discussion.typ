= Discussão <sec-discussao>

== O que os resultados respondem

A pergunta pedia a combinação de detector e rastreador que melhor equilibra acurácia e viabilidade. Nos dados medidos, a acurácia não separa as combinações, e o custo separa. Os dois detectores com ajuste fino empatam dentro da variação entre sementes; os quatro rastreadores empatam dentro de uma ou duas caixas. A escolha, então, é decidida pelo custo. Entre rastreadores, os só de movimento vencem com folga: o ReID reduz a pipeline a cerca de 40% da velocidade sem ganho mensurável. Entre detectores, o vencedor depende do hardware: em GPU os dois ficam próximos no vídeo nativo, e o YOLOv8s é mais rápido no recorte; em CPU, o YOLO26n quase dobra a velocidade.

O que de fato moveu a acurácia foi o ajuste fino no domínio, não a arquitetura: ele levou o mAP\@0,5 de 0,36 para mais de 0,97 na vista cenital, enquanto multiplicar os parâmetros por quase cinco não mudou o resultado. O recorte da arena também pesou durante o desenvolvimento, ao tirar do quadro o fundo que gerava falsos positivos, mas esse ganho não é reprodutível a partir dos artefatos salvos. A lição transferível é que, para alvos rígidos numa arena conhecida, dados do domínio rendem mais que capacidade de modelo.

Dois resultados não seguem o raciocínio que a literatura sugeria. Primeiro, o OC-SORT, desenhado para movimento não linear (C1), não superou o ByteTrack neste round. Com 59 quadros a 10 Hz, a referência registra pouco das colisões em que a correção do OC-SORT faria diferença; C1 continua sem teste adequado. Segundo, o ReID não ajudou, como no DanceTrack @sun2022dancetrack, mas por outro motivo: lá os alvos são parecidos; aqui são distinguíveis, e o movimento basta. A afirmação que os dados sustentam é estreita. Quando o movimento mantém a identidade, a aparência não paga seu custo. Segue em aberto se ela compensa quando os robôs são indistinguíveis e o movimento falha, como em oclusões longas.

== Ameaças à validade

*Amostra.* A detecção é avaliada em um round por fonte, e o rastreamento em um único round BR, com 59 quadros amostrados a 10 Hz. Trocas de identidade curtas entre quadros amostrados não aparecem na métrica. Não reportamos intervalos de confiança porque quadros de um mesmo round são fortemente correlacionados; as três sementes de E5 são a única estimativa de variância, e só para o YOLOv8s. A grade de rastreamento usa a semente 42; com a semente 0, o mesmo detector marca 28 falsos positivos dentro da arena no round BR, e não medimos o efeito disso no rastreamento.

*Independência entre treino e teste.* A divisão é por round, não por torneio. O round BR de referência é uma semifinal da IRONCup 2025 entre dois robôs que também aparecem no treino; um deles está em todos os rounds BR de treino e validação. O round JP sai do mesmo vídeo contínuo que o treino JP. Os números medem generalização para rounds novos de eventos conhecidos. A generalização para eventos novos só aparece na avaliação sem rótulos (@sec-sem-rotulos), e ali ela falha sob borrão extremo.

*Rótulos.* Os rótulos de treino não foram revisados, e a revisão dos 14 rounds JP alterou 71% dos quadros. O conjunto de referência partiu da proposta do SAM 3, e a revisão humana tende a aceitar caixas plausíveis, o que favorece o anotador em E1 e pode favorecer detectores treinados nas mesmas propostas.

*Resolução temporal.* No clipe BR, 48% dos pares de quadros consecutivos são quase idênticos, compatível com uma gravação de 30 FPS entregue a 60. Se confirmado, a resolução temporal efetiva do BR é a metade da nominal.

*Decisões de desenvolvimento.* O recorte da arena foi adotado depois de observar, no round BR de referência, a precisão do detector em quadro inteiro (cerca de 0,71). Os pesos daquele detector não foram preservados.

== Cinemática e eventos

A pipeline calcula posição, velocidade e eventos, mas este trabalho não os valida, por três razões. Primeiro, a escala isotrópica chega a um coeficiente de variação de 0,38 dentro de uma partida filmada com celular (@sec-sem-rotulos); velocidades em centímetros por segundo herdam esse erro até que a arena seja retificada por homografia. Segundo, as regras de evento dependem do detector: no round BR, o início do round dispara no quadro 20 com o YOLOv8s e no quadro 310 com o YOLO26n. Terceiro, não existe conjunto anotado de eventos separado do teste, e calibrar os limiares no próprio teste vazaria a avaliação. Enquanto essa validação não for feita, a saída de eventos serve como sugestão para revisão humana, não como medida.

== Trabalhos futuros

O passo de maior valor é ampliar o conjunto de referência com identidade, em mais rounds e nas duas fontes, incluindo confrontos entre robôs de fato parecidos e colisões anotadas em todos os quadros. Isso testaria C1 e C2 de verdade. O Studio, aplicação de anotação do repositório, já registra identidade, eventos e limites de round. Seguem: retificar a arena por homografia e validar a escala contra medidas conhecidas; anotar eventos num conjunto de calibração separado; reequilibrar as fontes e unificar a convenção do robô já fora do _dohyo_; incluir vídeo de transmissão com borrão no treino; e comparar detectores de outra família, como o RT-DETR @zhao2024rtdetr. A longo prazo, a pipeline pode sustentar uma plataforma aberta de análise de combate de robôs.

= Conclusão

Este trabalho perguntou quais combinações de detector e rastreador equilibram melhor acurácia e viabilidade ao rastrear alvos visualmente semelhantes, em movimento rápido e não linear, em vídeo não calibrado e sem marcadores. Usamos o Sumô de Robôs autônomo de 3 kg como domínio e construímos o Kanshigan, uma pipeline aberta de rastreamento por detecção, com detectores treinados em anotações do SAM 3 e avaliados contra rounds revisados à mão.

A resposta, nas condições medidas, é um detector YOLO com ajuste fino no recorte da arena combinado a um rastreador só de movimento, como o OC-SORT ou o ByteTrack: o YOLOv8s ou o YOLO26n quando há GPU, o YOLO26n quando só há CPU. A acurácia vem do ajuste fino no domínio, não da arquitetura, e a aparência não paga seu custo quando o movimento já mantém a identidade. A pipeline roda em tempo real sobre vídeo 1080p numa GPU de notebook, com menos de 300 MiB de memória, e a 40 FPS só na CPU.

O alcance é limitado por um round de teste por fonte e um único round com identidade, ambos de eventos presentes no treino. Cinemática e eventos são produzidos, mas não validados. O código, os scripts e resultados que geram cada tabela e o diário de bordo que registra cada decisão e cada erro estão em #link("https://github.com/Eduardo-Barreto/Kanshigan")[github.com/Eduardo-Barreto/Kanshigan].
