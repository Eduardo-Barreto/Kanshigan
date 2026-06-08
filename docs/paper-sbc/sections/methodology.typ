= Metodologia

A pipeline opera em sete estágios sobre o vídeo nativo, sem decimar quadros, para
preservar a resolução temporal que os eventos sub-segundo exigem.

+ *Decodificação* quadro a quadro em resolução e taxa nativas.
+ *Detecção do dohyo* por visão clássica: limiarização de luminância para isolar a
  borda branca (tawara), morfologia e ajuste de elipse ao maior contorno
  plausível, selecionado por um escore que combina tamanho, centralidade e razão
  de aspecto. A câmera de mão se move, então a calibração é feita por quadro, com
  reuso da última detecção válida em quadros de falha.
+ *Calibração espacial*: a elipse detectada é a imagem oblíqua de um círculo de
  154 cm de diâmetro; seu eixo maior fixa a escala centímetro-por-pixel.
+ *Recorte no dohyo*: o quadro é cortado na elipse detectada antes da detecção. Os
  robôs ocupam fração pequena do quadro; alimentar o quadro inteiro a um detector de
  640 px os encolhe abaixo do que sobrevive ao borrão de movimento, e o fundo (plateia,
  mãos) gera falso positivo. O recorte amplia os robôs cerca de três vezes e remove o
  fundo. As caixas voltam a coordenadas nativas por um deslocamento.
+ *Detecção dos robôs* com YOLOv8s @jocher2023yolo sobre o recorte, filtrando
  detecções fora da elipse e mantendo no máximo dois robôs por quadro.
+ *Rastreamento* com OC-SORT @cao2023ocsort, escolhido por lidar com o movimento
  não linear de colisões e giros. As detecções viram trajetórias contínuas.
+ *Extração de métricas* no referencial do dohyo, em centímetros. Expressar cada
  robô relativo ao centro da arena cancela o movimento da câmera, pois o dohyo é
  fixo no mundo. Velocidade e aceleração vêm de um filtro de Savitzky-Golay, que
  diferencia e suaviza em uma só passagem.
+ *Detecção de eventos* por regras determinísticas e inspecionáveis (descritas
  abaixo), em vez de um classificador treinado, dado o tamanho do conjunto de
  avaliação.

== Identidade dos robôs

OC-SORT emite identificadores arbitrários. Fixamos a convenção de que o robô A é o
mais à esquerda no primeiro quadro em que ambos são rastreados, e mantemos apenas
as duas trajetórias mais longas, robusto a fragmentação de identidade.

== Detector cinemático de eventos

Trabalhando no referencial centrado no dohyo, os eventos viram testes simples sobre
as séries cinemáticas. O *início do round* é o primeiro quadro em que ambos os
robôs superam um limiar de velocidade. O *primeiro contato* é o primeiro quadro em
que ambos sofrem variação brusca de velocidade simultânea e estão próximos. O
*ring-out* é o primeiro quadro em que a distância do centro de um robô ao centro do
dohyo excede o raio --- um teste invariante ao movimento da câmera. O vencedor é o
robô que não sofreu ring-out; rounds sem ring-out são marcados para resolução
manual. Os limiares ficam em arquivo de configuração versionado e são reportados
tanto calibrados no conjunto gold quanto em valores padrão de literatura, para
mostrar que a detecção não depende exclusivamente do ajuste.

== Conjunto de dados e protocolo de anotação

Sem dataset público do domínio, construímos o conjunto a partir de footage real de
torneios. Os clips de treino e validação são anotados de forma semiautomática: o
SAM 3 @carion2025sam3 propaga máscaras a partir de um prompt textual, as máscaras
viram caixas pela maior componente conexa, e as caixas são escaladas para a
resolução nativa em formato YOLO. Para caber em GPU de 8 GB, o SAM 3 processa cada
clip em janelas curtas, encerrando a sessão entre janelas para liberar memória ---
rounds de Sumô são curtos o bastante para uma janela cobrir um round inteiro.

A divisão treino/validação é feita por clip, não por quadro: quadros vizinhos de um
mesmo round são quase idênticos, então dividir por quadro vazaria informação entre
os conjuntos. O conjunto gold (teste) é anotado e revisado manualmente, fica fora
do treino e sustenta três medições independentes: SAM 3 contra gold (validação do
anotador), modelo treinado contra gold (generalização) e modelo contra SAM 3
(deriva).

== Protocolo de treino

O detector parte dos pesos COCO do YOLOv8s, treina com `imgsz=640`, aumento padrão
do Ultralytics, semente fixa em 42 e parada antecipada no platô da validação. Os
hiperparâmetros são versionados no repositório, não escondidos em comentários, para
reprodutibilidade.
