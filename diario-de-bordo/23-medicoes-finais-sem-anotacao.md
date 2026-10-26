# Medições finais sem anotação nova

## Contexto

A banca pediu um conjunto de números para o artigo final. A regra desta rodada: só computação. Nenhum rótulo novo, nenhum rótulo alterado, nenhum retreino do detector reportado. Tudo roda na máquina com a RTX 4070 Laptop (8 GB) e o Intel Core Ultra 9 185H, com torch 2.12.0+cu126, Ultralytics 8.4.55 e boxmot 18.0.0. As saídas ficam em `results/E6_final_measurements/`, com um README que liga cada arquivo ao comando que o gerou. O código novo mora em `experiments/pos-banca/`: `final_tracking.py`, `final_viability.py`, `final_detection.py`, `appearance_separability.py`, `round_durations.py`, `rc_rounds.py` e três scripts de lote. A lógica não trivial ganhou 17 testes (`tests/test_final_measurements.py`; a suíte do pós-banca passou de 7 para 24).

Esta entrada registra o que foi medido, o que mudou de leitura e o que não deu para medir.

## O gold de identidade, contado

A suspeita era que o gold de identidade BR fosse menor do que o texto sugere. Confirmada. O `identity.txt` tem 59 quadros e 118 caixas, nos quadros nativos 6, 12, ..., 354 de um clip de 385 quadros a 60 fps: um quadro a cada 6, ou seja, 10 Hz. Conferimos o alinhamento por PSNR: a imagem `gold_zb01_0001` casa com o quadro nativo 6 (38 dB contra 32 dB do vizinho), e o mesmo vale nos índices 10, 30 e 50.

Dois achados laterais apareceram nessa conferência:

- O `identity.txt` foi gerado antes do conserto do recorte do diário 16. Seis caixas ficam inteiramente fora do quadro do clip recortado (robôs empurrados para trás do dohyo, no fim do round) e quatro ficam parcialmente fora. O `gold_crop/labels` já tinha sido corrigido para 112 caixas; o `identity.txt`, não. O `evaluate.py` conta essas seis caixas como robô perdido, e nenhum rastreador consegue acertá-las.
- O clip BR parece ter quadros repetidos: 185 de 384 pares consecutivos têm diferença média de cinza abaixo de 0,5, contra 0 no clip JP. Os pares 59/60, 179/180 e 299/300 são idênticos no PSNR. É compatível com um vídeo de 30 fps entregue a 60 fps, mas não verificamos a gravação original. Se for isso, a resolução temporal real do BR é metade da declarada.

## Rastreamento: grade 2 x 4, com HOTA

O `tracker_comparison.py` do diário 17 fixava o YOLO26n. O `final_tracking.py` repete o desenho para os dois detectores: cada detector roda uma vez, as detecções ficam em cache e alimentam os quatro rastreadores. O HOTA vem das classes oficiais do TrackEval (último commit, `12c8791`), alimentadas quadro a quadro com IoU, só nos quadros do gold, como o `evaluate.py` faz. Para confiar no encaixe, o mesmo dado passa pelo CLEAR e pelo Identity do TrackEval: MOTA, IDF1 e trocas de ID batem com o motmetrics nas 32 combinações (8 rastreamentos, 2 variantes de trilha, 2 de gold). A linha do YOLO26n reproduz o `E4_tracker_comparison` dígito a dígito.

Números como o artigo os mede (trilhas após `assign_ab`, gold como está):

| Detector | Rastreador | MOTA | IDF1 | HOTA | DetA | AssA | Trocas | FPS do rastreador |
|---|---|---|---|---|---|---|---|---|
| YOLOv8s | OC-SORT | 0,881 | 0,933 | 0,810 | 0,799 | 0,824 | 1 | 3145 |
| YOLOv8s | ByteTrack | 0,898 | 0,947 | 0,824 | 0,815 | 0,836 | 0 | 3405 |
| YOLOv8s | DeepOCSORT | 0,881 | 0,933 | 0,810 | 0,799 | 0,824 | 1 | 100 |
| YOLOv8s | BoT-SORT | 0,907 | 0,952 | 0,833 | 0,823 | 0,847 | 0 | 98 |
| YOLO26n | OC-SORT | 0,881 | 0,933 | 0,796 | 0,783 | 0,816 | 1 | 3004 |
| YOLO26n | ByteTrack | 0,890 | 0,938 | 0,804 | 0,791 | 0,822 | 1 | 3253 |
| YOLO26n | DeepOCSORT | 0,856 | 0,919 | 0,774 | 0,762 | 0,793 | 1 | 96 |
| YOLO26n | BoT-SORT | 0,890 | 0,938 | 0,804 | 0,790 | 0,825 | 1 | 96 |

FPS do rastreador é a média de três repetições, com detecções prontas. O LocA fica em 0,91 (YOLOv8s) e 0,89 (YOLO26n).

O que se lê:

- A coincidência do diário 17 se explica. Lá o OC-SORT sobre o YOLO26n saiu "idêntico ao número da entrada 16", que era YOLOv8s. É coincidência real: MOTA e IDF1 empatam, mas o HOTA separa (0,810 contra 0,796) porque as caixas do YOLO26n são menos justas.
- Com o YOLOv8s, o BoT-SORT fica ligeiramente à frente de todos. A conclusão do diário 17 ("aparência não ajuda") dependia do detector. O certo é: as diferenças entre rastreadores cabem em uma ou duas caixas de 118 e em no máximo uma troca de ID, num round só.

**As seis caixas fora do recorte.** Recortando as caixas do gold ao quadro do clip (sem mexer no arquivo) e descartando as que não têm área visível, o gold fica com 112 caixas, o mesmo número do `gold_crop`. O OC-SORT sobre o YOLOv8s vai de MOTA 0,881 e IDF1 0,933 para 0,929 e 0,959, e o HOTA de 0,810 para 0,853. A ordenação entre rastreadores não muda. As seis caixas eram, sim, contadas como erro do rastreador, e o número antigo subestima o rastreamento em cerca de 5 pontos de MOTA. Guardamos as duas versões no JSON.

**Sem o `assign_ab`.** O pós-filtro mantém as duas trilhas mais longas e renomeia para A e B. Avaliando os ids crus do rastreador (4 ou 5 ids por execução), o OC-SORT sobe em MOTA (0,881 para 0,890, porque as trilhas curtas descartadas tinham acertos) e cai em IDF1 (0,933 para 0,925, com 2 trocas em vez de 1). O BoT-SORT e o ByteTrack sobre o YOLO26n não mudam. O `assign_ab` não está escondendo erro grande.

## Viabilidade: decodificação, carga e processamento separados

O `infer.py` inicia o cronômetro depois de decodificar o vídeo e antes de carregar o modelo. O artigo diz que o FPS inclui decodificação; não incluía. O `final_viability.py` refaz os mesmos estágios com as mesmas funções e cronometra cada um: decodificação, carga do modelo e processamento (calibração do dohyo, recorte, detecção, rastreamento, cinemática e eventos). Cada configuração roda num processo novo: uma partida fria e três aquecidas. O contexto CUDA é criado antes do cronômetro também na partida fria, porque a primeira sincronização o inicializa.

A outra correção veio do coordenador da auditoria: o "acima de 100 FPS" foi medido no clip já recortado (1558 x 506). O mesmo round em resolução nativa (1920 x 1080, cortado do bruto com `ffmpeg -c copy`, com o recorte dentro do cronômetro) é a entrada honesta.

GPU, OC-SORT, média e desvio de três execuções aquecidas:

| Entrada | Detector | Ponta a ponta com decodificação | Equivalente ao `infer.py` | Só processamento | Partida fria (ponta a ponta) |
|---|---|---|---|---|---|
| Recortado 1558x506 | YOLOv8s | 154,5 ± 2,2 | 170,6 ± 0,9 | 172,5 ± 1,4 | 127,8 |
| Recortado 1558x506 | YOLO26n | 127,7 ± 1,2 | 139,0 ± 1,4 | 140,4 ± 1,3 | 111,7 |
| Nativo 1920x1080 | YOLOv8s | 97,0 ± 7,0 | 112,2 ± 1,6 | 112,7 ± 1,6 | 74,2 |
| Nativo 1920x1080 | YOLO26n | 88,9 ± 2,2 | 101,3 ± 0,3 | 102,0 ± 0,3 | 79,3 |
| RC 4K (IMG_1594, 334 quadros) | YOLOv8s | 16,1 ± 2,6 | 38,4 ± 0,4 | 38,5 ± 0,4 | 11,3 |
| RC 4K (IMG_1594, 334 quadros) | YOLO26n | 16,5 ± 1,4 | 37,1 ± 0,5 | 37,2 ± 0,5 | 14,4 |

A decodificação custa 0,2 s no clip recortado, 0,4 a 0,9 s no nativo H.264 e 10 a 20 s no 4K HEVC, onde vira o gargalo: o processamento roda a 38 FPS e a pipeline inteira a 16. A carga do modelo aquecida é de 20 a 30 ms; fria, até 150 ms (até 0,7 s com o modelo de ReID dos rastreadores de aparência).

Leitura: no round nativo a 1080p, a pipeline com decodificação fica perto de 90 a 100 FPS aquecida e 74 a 79 FPS fria. A afirmação defensável é "acima de 60 FPS em 1080p, em tempo real para vídeo de 60 fps", não "acima de 100 FPS". O YOLO26n, com um quinto dos parâmetros, é mais lento na GPU que o YOLOv8s (128 contra 155 FPS no recorte). Ele só vence na CPU.

CPU (`CUDA_VISIBLE_DEVICES` vazio, 8 threads do torch), OC-SORT, ponta a ponta com decodificação:

| Entrada | YOLO26n | YOLOv8s |
|---|---|---|
| Recortado | 45,3 ± 0,6 | 23,4 ± 1,4 |
| Nativo 1080p | 39,6 ± 2,3 | 21,3 ± 1,7 |

Em CPU de notebook, sem GPU, o YOLO26n roda a cerca de 40 FPS no nativo: análise pós-partida viável em hardware de consumo, mas abaixo de 60 FPS.

**Rastreador dentro da pipeline.** Com o recorte e a GPU, ponta a ponta com decodificação:

| Rastreador | YOLOv8s | YOLO26n |
|---|---|---|
| OC-SORT | 154,5 ± 2,2 | 127,7 ± 1,2 |
| ByteTrack | 150,7 ± 2,2 | 129,9 ± 3,4 |
| DeepOCSORT | 56,8 ± 0,9 | 53,5 ± 1,2 |
| BoT-SORT | 61,3 ± 1,2 | 57,2 ± 1,2 |

Com aparência, a pipeline inteira cai para perto de 60 FPS, já abaixo do tempo real no recorte. No nativo, cairia mais.

**VRAM.** A primeira medição deu números que cresciam a cada execução aquecida (pico alocado do YOLOv8s de 78 para 204 MiB). A causa é o desenho da medição, não a pipeline: as três execuções no mesmo processo mantêm vivos os modelos das execuções anteriores. O número de uma pipeline isolada é o da partida fria. Ele reproduz o diário 18: YOLOv8s com 78,4 MiB alocados (82,2 MB) e 96 MiB reservados (100,7 MB); YOLO26n com 63,4 MiB e 86 MiB. O NVML, que inclui o contexto CUDA e é o que o `nvidia-smi` mostra, dá 250 MiB para o processo do YOLOv8s e 272 MiB para o do YOLO26n. Com ReID, 334 e 300 MiB. Pico de qualquer execução: 424 MiB. Tudo abaixo de 6% da GPU.

## Detecção: duas convenções de limiar e o pós-filtro do dohyo

O artigo reporta precisão e recall do `model.val` do Ultralytics. Esses números saem do limiar de confiança que maximiza o F1 médio na curva PR (NMS a 0,001, depois `smooth(f1).argmax()`), não de um limiar fixo. O `data_scaling.py` e o rastreador usam confiança 0,25. O `final_detection.py` mede as duas convenções e acrescenta um pós-filtro: descartar predições com centro fora da elipse do dohyo, ajustada pelo próprio `dohyo.detect_calibration` em cada imagem do gold (margem 1,0, o anel; e 1,15, a do `infer.py`). Só as predições são filtradas.

| Detector / gold | mAP50 | mAP50-95 | P / R (val, F1 máx.) | P / R (conf 0,25) | P / R (0,25, filtro m1,0) | P / R (0,25, filtro m1,15) |
|---|---|---|---|---|---|---|
| YOLOv8s / BR | 0,965 | 0,767 | 0,990 / 0,964 | 0,991 / 0,964 | 0,991 / 0,964 | 0,991 / 0,964 |
| YOLOv8s / JP | 0,976 | 0,695 | 0,991 / 0,915 | 0,720 / 0,939 | 0,854 / 0,927 | 0,856 / 0,939 |
| YOLO26n / BR | 0,968 | 0,772 | 0,963 / 0,955 | 0,923 / 0,964 | 0,923 / 0,964 | 0,923 / 0,964 |
| YOLO26n / JP | 0,993 | 0,691 | 0,986 / 0,976 | 0,859 / 0,963 | 0,987 / 0,951 | 0,975 / 0,963 |

O mAP e o P/R do `val` reproduzem a tabela do artigo. A conf 0,25, a precisão do YOLOv8s no JP cai para 0,72: 30 falsos positivos em 57 quadros. O filtro tira 17 deles (os de fora da arena, quase todos o robô já ejetado, que o gold JP não rotula) e a precisão sobe para 0,85; restam 13 falsos positivos dentro do dohyo. No YOLO26n, o filtro leva a precisão de 0,86 para 0,99. A pipeline já aplica esse filtro (margem 1,15) e mantém as duas melhores caixas, então a coluna filtrada é a mais próxima do que o rastreador recebe. Uma caixa do gold JP fica fora da elipse; por isso o recall cai um pouco com a margem 1,0. No BR, o filtro não muda nada. Em 2 dos 59 quadros do gold BR a elipse não foi ajustada e o filtro caiu na elipse inscrita no recorte.

**O baseline COCO (E3).** Os 0,026 de mAP50 se reproduzem, mas só na convenção padrão: com o gold de uma classe, o Ultralytics só casa a classe 0 do COCO, "pessoa". Sem classe (`single_cls=True`, qualquer objeto COCO conta como robô), o YOLOv8s pré-treinado chega a mAP50 0,757 no BR e 0,359 no JP. O "duas ordens de grandeza abaixo" mede, em parte, um descasamento de rótulo, não só a falta de treino no domínio. A afirmação honesta é que o detector genérico acha parte dos robôs como algum objeto COCO (0,76 no BR), e o treino no domínio leva a 0,97.

## Eventos: o round_start do gold

O `results/E2_yolo_oc_vs_gold/gold_zb01.json` ainda tinha `round_start` no quadro 1, o artefato de borda do Savitzky-Golay que o diário 16 diz ter corrigido. O conserto existia no `events.py` (pular 2 quadros de borda e exigir 3 de movimento sustentado); o JSON é que nunca foi regerado. Com o `infer.py` atual (YOLOv8s, OC-SORT), o `round_start` cai no quadro 20 (333 ms), como o diário 16 descreve. No mesmo round em resolução nativa, cai no 24. Não há ring-out: o round termina em `timeout_manual_review` no quadro 334.

Um achado que o diário 16 não tinha: com o YOLO26n, o mesmo round dispara `round_start` no quadro 310, quase no fim. A regra de eventos é sensível ao detector, e só foi conferida com o YOLOv8s.

A âncora da auditoria mudou, então a linha de base foi regerada (`results/audit/baseline.csv`; a anterior está em `E6_final_measurements/audit/baseline_before_E6.csv`). Regeramos também os clips cujo vídeo de origem está na máquina: atena (`/home/barreto/tcc/atenavsbullbassauro.mp4`, mesmos 598 quadros), o mundial (`data/raw/jp/worlds_...`) e o rc1591 (`IMG_1591.MOV` reduzido a 1080p). Para `demo_jp` e `w_replay` não achamos o vídeo nativo de origem, então ficam os JSONs antigos.

| Clip | Eventos disparados | Violação antes | Violação agora |
|---|---|---|---|
| `demo_jp` (JSON antigo) | round_start 55, ring_out 177, round_end 177 | 0,000 | 0,000 |
| `rc1591` | round_start 196, round_end 704 (timeout) | 0,008 | 0,077 |
| `gold_zb01` | round_start 20, round_end 334 (timeout) | 0,060 | 0,064 |
| `atena` | round_start 323, round_end 597 (timeout) | 0,444 | 0,444 |
| `w_replay` (JSON antigo) | round_start 120, round_end 241 (timeout) | 0,541 | 0,541 |
| `worlds` | sem round_start, round_end 2021 (timeout) | 1,000 | 1,000 |

O gold passa de 6,0% para 6,4% de violação: a janela agora começa no 20 e não no 1. O rc1591 piorou bastante, de 0,8% para 7,7%. Não sabemos com que entrada e parâmetros o JSON antigo foi gerado (o diário 20 rodou em outra máquina), então tratamos o número novo como o reprodutível.

## Aparência: a premissa C2 não se sustenta neste round

O artigo explica o empate entre rastreadores dizendo que os dois robôs são caixas pretas quase idênticas e que a aparência quase não carrega sinal. Isso nunca tinha sido medido. O `appearance_separability.py` recorta as 111 caixas visíveis do gold de identidade (56 de A, 55 de B, lado mínimo de 12 px), extrai embeddings e compara a similaridade cosseno de pares do mesmo robô com a de pares de robôs diferentes.

| Extrator | Intra (média ± dp) | Inter (média ± dp) | AUC | AUC (pares a 0,5 s ou mais) | Sobreposição | Vizinho mais próximo |
|---|---|---|---|---|---|---|
| OSNet x0.25 MSMT17 (o ReID do boxmot) | 0,849 ± 0,124 | 0,542 ± 0,037 | 0,940 | 0,934 | 0,109 | 0,973 |
| ResNet-50 ImageNet | 0,737 ± 0,141 | 0,477 ± 0,065 | 0,912 | 0,904 | 0,124 | 0,973 |
| Histograma HSV 8x8x8 | 0,837 ± 0,190 | 0,452 ± 0,070 | 0,924 | 0,917 | 0,148 | 0,955 |

A hipótese caiu. Os dois robôs deste round são distinguíveis pela aparência, pelos três extratores, inclusive por um histograma de cor. O vizinho mais próximo, excluindo quadros a menos de meio segundo, acerta a identidade em 97% das caixas. O ReID do boxmot separa melhor que o extrator genérico, então não é um modelo inadequado.

A explicação que os dados sustentam para o empate é outra: o movimento sozinho já resolve este round (0 ou 1 troca de ID em 118 caixas), e não sobra erro para a aparência corrigir. O round é Galena contra Gyarados, robôs de equipes diferentes. A afirmação "robôs indistinguíveis" pode valer em outros confrontos, mas não foi medida em nenhum.

Ressalvas: o recorte inclui fundo do dohyo, e pares do mesmo robô em posições próximas compartilham fundo; a identidade do gold veio de ligação por centróide com revisão humana; e é um round só.

## Duração dos rounds

A frase "a maioria dos rounds termina em menos de um segundo" foi testada de três jeitos, do mais frouxo ao mais rigoroso (`round_durations.py`):

1. **Janelas do `segment_rounds.py`.** JP: 26 janelas, mediana 4,0 s, mínimo 3,0 s. BR: 8 janelas, mediana 10,7 s, máximo 33,4 s. Nenhuma abaixo de 1 s, mas a janela inclui a espera antes do hajime e margem. As marcas manuais do `br01` (formato antigo, quadros nativos) dão 2,7, 3,1, 1,3 e 6,4 s.
2. **Instantes da pipeline a 60 fps** (YOLOv8s + OC-SORT) em todos os 34 clips nativos: movimento (o `round_start`), contato (primeiro quadro em que as caixas dos dois robôs se intersectam) e saída (primeiro quadro em que o centro de um robô sai da elipse do dohyo daquele quadro).
3. **Os mesmos instantes nos rótulos revisados** (10 Hz). Os quadros rotulados são só os que o SAM achou e a revisão manteve, e o início do round muitas vezes não está rotulado, então não medimos "movimento" pelos rótulos.

Cada fonte isolada tem ruído. Num clip, a caixa de um robô começa fora da elipse; noutro, o robô ejetado não é rotulado; noutro, a pipeline perde o robô. Por isso o número principal é o do subconjunto em que pipeline e rótulos concordam no instante da saída com folga de 0,1 s: 11 rounds, 10 deles JP. Ali:

| Medida | n | Mediana | Abaixo de 1 s |
|---|---|---|---|
| Movimento até a saída (pipeline) | 8 | 1,09 s | 50% |
| Contato até a saída (pipeline) | 7 | 1,00 s | 43% |
| Contato até a saída (rótulos) | 10 | 0,75 s | 60% |

Sem o filtro de concordância, no JP: movimento até a saída com mediana de 1,40 s (n = 11, 36% abaixo de 1 s) e contato até a saída com mediana de 0,75 s nos rótulos (n = 14, 57%). O movimento até o contato tem mediana de 0,5 s. No BR, só dois rounds têm movimento e saída detectados nessa ordem pela pipeline (4,2 e 24,1 s), e as janelas são longas.

A frase do artigo não se sustenta como escrita. O que os dados sustentam: na fonte JP (câmera cenital fixa, um único torneio), do primeiro movimento à saída do dohyo o round dura cerca de 1 s de mediana; a fase de contato até a saída dura de 0,75 a 1 s de mediana, abaixo de 1 s em pouco mais da metade dos casos; no BR, os rounds duram vários segundos. Uma formulação defensável: "a fase decisiva, do primeiro contato à saída do dohyo, dura tipicamente cerca de um segundo".

## RC: os 23 rounds, enfim

O diário 20 deixou pendente a pipeline nos rounds RC por falta de pesos e GPU. Os vídeos não estão em `data/raw`, mas em `/home/barreto/tcc/videos/3kg RC/`. O `rc_rounds.py` descarta as duplicatas por md5 (as mesmas três do diário 20), segmenta, corta e roda o `infer.py` (YOLOv8s + OC-SORT). Os 4K foram reduzidos a 1920 px de largura, porque o `infer.py` guarda todos os quadros na memória e um round 4K de 15 s passaria de 20 GB. A segmentação reproduziu os 23 rounds de 22 vídeos.

Auditoria estrutural (`audit/rc_rounds.csv`), com o gold como âncora (6,4% de violação):

- Violação mediana por round: 26,8%; ponderada por quadro: 29,8%.
- 10 dos 23 rounds ficam no regime do gold ou melhor (violação até 6,4%); 7 ficam acima de 50%.
- 6 rounds terminam por ring-out detectado; os outros caem em `timeout_manual_review`.
- 2 saltos impossíveis (teleportes), ambos no IMG_1584, que também tem 97,5% de violação.
- Em 3 rounds a janela de auditoria tem menos de 60 quadros (round_start tarde ou ring-out cedo), o que torna a taxa instável ali.

O RC deixa de ser só figura qualitativa: cerca de metade dos rounds opera no regime do gold e a outra metade, não.

## SAM 3: predictor de vídeo no gold JP e o E1 reavaliado

Rodamos o SAM 3 num diretório de dados temporário (`KANSHIGAN_DATA_ROOT`), para não gravar nada nas pastas de rótulo. Avaliação pelo `evaluate_annotator` (IoU de 0,5).

| Anotador | Gold | Precisão | Recall | F1 | IoU médio |
|---|---|---|---|---|---|
| Predictor de vídeo, "toy", limiar 0,15 | JP | 0,875 | 0,939 | 0,906 | 0,975 |
| Predictor de imagem, "robot" | BR | 0,981 | 0,938 | 0,959 | 0,912 |
| Predictor de imagem, "toy", limiar 0,15 | JP | 1,000 | 0,159 | 0,274 | 0,917 |

O BR reproduz o diário 16 exatamente. O predictor de vídeo no JP fecha a issue #14 do lado computacional: recall de 0,94 contra 0,16 do predictor de imagem. Ressalva forte: o gold JP foi pré-anotado pelo próprio predictor de vídeo e depois revisado, e o IoU de 0,975 mostra isso. A concordância é um limite superior. O prompt original do E1 no JP não está registrado; usamos o do anotador JP ("toy", 0,15) e saiu recall 0,16, perto dos ~22% do diário 16.

## Proveniência dos golds

- **O gold BR divide evento e robôs com o treino.** Pelos títulos no YouTube, o `gold_zb01` é "Galena vs. Gyarados, semifinal, IRONCup 2025". Dois rounds de treino são Galena contra Gyarados em outras partidas do mesmo torneio (`tr_B_r1`, e `tr_fz_r1`, a grande final), e a Galena está em todos os rounds BR de treino e validação. O gold BR é um round inédito, não um evento nem um robô inéditos.
- **O gold JP sai do mesmo vídeo do treino JP.** O `gold_jp25` é um round do `Z_MC3r_c2P0.mp4`, o mesmo vídeo contínuo, de câmera fixa, de onde vêm os 8 rounds de treino, a validação e os candidatos.
- **O ganho do recorte (0,71 para 0,99) foi medido no gold BR**, comparando um detector treinado e avaliado em quadro inteiro com o detector atual, treinado no recorte. O "antes" vem do diário 14 (commit `e7e1cd4`); não sobrou peso, run nem JSON desse modelo, então o número não é reproduzível a partir dos artefatos salvos.

## O que não deu

- **Reproduzir o "antes" do recorte.** Sem os pesos do detector de quadro inteiro, e retreinar foge da regra desta rodada.
- **Achar a origem de `demo_jp` e `w_replay`.** Ficaram com os JSONs antigos na linha de base. O `w_replay` só existe na versão decimada do SAM.
- **Separar o CUDA init da partida fria.** O contexto é criado antes do cronômetro; a partida fria mede o primeiro modelo e o primeiro autotune, não a criação do contexto.
- **Confirmar os quadros duplicados do BR na fonte.** Fica como indício.

## Lições

- Um conserto sem regerar o artefato não conserta o artigo. O `round_start` do gold ficou meses no quadro 1 num JSON versionado, e a correção do recorte nunca chegou ao `identity.txt`.
- O cronômetro define o número. Mover a decodificação para dentro e trocar a entrada recortada pela nativa leva o "acima de 100 FPS" para perto de 90 aquecido e 75 frio.
- Uma premissa plausível precisa de medida. "Os robôs são indistinguíveis" parecia óbvio e caiu com um histograma de cor.
- O baseline também tem convenção. O 0,026 do COCO media, em parte, a classe errada.

## Pendências para o artigo

- Trocar a afirmação de FPS pela medida em 1080p nativo com decodificação, e reportar CPU.
- Reescrever o parágrafo de aparência (C2): o empate entre rastreadores não vem de robôs indistinguíveis.
- Declarar o gold de identidade: 59 quadros a 10 Hz, 118 caixas, seis fora do recorte; reportar HOTA e a versão visível.
- Declarar que gold BR e treino BR dividem evento e robôs, e que gold JP e treino JP dividem o vídeo.
- Trocar "a maioria dos rounds termina em menos de um segundo" pela frase da fase de contato.
- Reportar o COCO sem classe ao lado do 0,026, ou explicar a convenção.
- Regerar o `identity.txt` com o recorte é mudança de rótulo: fica para decisão do autor.
