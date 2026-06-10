= Metodologia <sec-metodologia>

#figure(
  image("/docs/paper-sbc/figures/pipeline.svg", width: 100%),
  caption: [Estágios da pipeline. A primeira fileira prepara o quadro (decodificação a
  recorte); a segunda extrai a informação (detecção a eventos). O ramo tracejado é a
  anotação offline com o SAM 3, que constrói o conjunto de treino do detector e fica
  fora da inferência.],
) <fig-pipeline>

A pipeline segue o paradigma de *tracking-by-detection* discutido na seção
anterior e opera em oito estágios sobre o vídeo nativo, sem decimar quadros, para
preservar a resolução temporal que os eventos sub-segundo exigem (@fig-pipeline).
Os quatro primeiros estágios preparam o quadro: decodificação em resolução e taxa
nativas, detecção da arena, calibração espacial e recorte. Os quatro últimos
extraem a informação: detecção dos robôs, rastreamento, métricas cinemáticas e
eventos. Separar a pipeline em estágios com saídas inspecionáveis permite avaliar
cada componente isoladamente contra o conjunto gold, como a seção de resultados
faz. As subseções a seguir detalham cada etapa e o porquê de cada decisão.

== Detecção da arena e calibração espacial

A arena é detectada por visão clássica, sem modelo treinado. A escolha é
deliberada: o dohyo tem geometria e aparência fixadas por regulamento (disco
escuro com borda branca, a tawara), então um procedimento determinístico resolve
o problema sem exigir dados anotados, e suas falhas são diagnosticáveis por
inspeção. O procedimento limiariza a luminância para isolar a borda branca,
aplica operações morfológicas para limpar o resultado e ajusta uma elipse ao
maior contorno plausível, selecionado por um escore que combina tamanho,
centralidade e razão de aspecto. Como a câmera de mão se move, a detecção é
refeita a cada quadro, com reuso da última detecção válida nos quadros em que o
ajuste falha.

A elipse alimenta a calibração espacial. Um círculo visto em perspectiva oblíqua
projeta-se como elipse, e o eixo maior dessa elipse corresponde ao diâmetro do
círculo, conhecido por regulamento (154 cm na categoria de 3 kg). A razão entre
os dois fixa a escala centímetro-por-pixel de cada quadro, convertendo todas as
medições posteriores do espaço de pixels para o referencial físico da arena, sem
exigir calibração prévia da câmera nem marcadores externos.

== Recorte e detecção dos robôs

Antes da detecção, o quadro é cortado na elipse da arena. Os robôs ocupam fração
pequena do quadro inteiro; alimentá-lo direto a um detector de 640 px os encolhe
abaixo do que sobrevive ao borrão de movimento, e o fundo (plateia, mãos de
operadores) gera falso positivo. O recorte amplia os robôs cerca de três vezes e
remove o fundo; as caixas detectadas voltam a coordenadas nativas por um
deslocamento.

Sobre o recorte roda um detector YOLO com fine-tuning no domínio, filtrando
detecções fora da elipse e mantendo no máximo dois robôs por quadro. Comparamos
duas arquiteturas, o YOLOv8s @jocher2023yolo e o YOLO26n compacto
@jocher2026yolo26, para medir quanto a capacidade do modelo importa no domínio.

== Rastreamento e identidade

As detecções viram trajetórias contínuas por um rastreador que associa caixas
entre quadros. A restrição C2 (dois robôs visualmente quase idênticos) sugere que
a aparência teria pouco a oferecer, mas tratamos isso como hipótese a testar, não
como premissa. Comparamos quatro rastreadores sobre as mesmas detecções: dois
motion-only, OC-SORT @cao2023ocsort e ByteTrack @zhang2022bytetrack, e dois com
aparência via ReID, DeepOCSORT @maggiolino2023deepocsort e BoT-SORT
@aharon2022botsort. Os pares motion-only representam estratégias distintas diante
do movimento não linear de colisões e giros (C1): o OC-SORT corrige o filtro de
Kalman com as observações recentes, e o ByteTrack recupera detecções de baixa
confiança, frequentes durante o borrão de movimento. Os dois com ReID medem
diretamente se a aparência adiciona algo sob C2; a seção de resultados mostra que
não.

Os rastreadores emitem identificadores arbitrários, que não dizem qual robô é
qual. Fixamos a convenção de que o robô A é o mais à esquerda no primeiro quadro
em que ambos são rastreados, e mantemos apenas as duas trajetórias mais longas.
Como só existem dois alvos em cena, qualquer fragmento curto de trajetória é
ruído de identidade, e descartá-lo torna a convenção tolerante à fragmentação.

== Métricas e eventos cinemáticos

As métricas são extraídas no referencial do dohyo, em centímetros. Expressar cada
robô relativo ao centro da arena cancela o movimento da câmera, pois o dohyo é
fixo no mundo: quando a câmera de mão treme, arena e robôs se deslocam juntos na
imagem, e a posição relativa permanece. Velocidade e aceleração não podem vir de
diferenças finitas diretas, que amplificam o ruído de detecção quadro a quadro;
usamos um filtro de Savitzky-Golay @savitzky1964smoothing, que ajusta polinômios
locais por mínimos quadrados e, com isso, suaviza e diferencia a série em uma só
passagem.

Os eventos são detectados por regras determinísticas e inspecionáveis sobre as
séries cinemáticas, em vez de um classificador treinado. Com um conjunto de
avaliação pequeno, um classificador aprenderia os poucos exemplos disponíveis sem
garantia de generalizar; regras explícitas, além de não exigirem treino, tornam
cada decisão da pipeline auditável. O *início do round* é o primeiro quadro em que ambos os
robôs superam um limiar de velocidade. O *primeiro contato* é o primeiro quadro
em que ambos sofrem variação brusca de velocidade simultânea e estão próximos. O
*ring-out* é o primeiro quadro em que a distância do centro de um robô ao centro
do dohyo excede o raio, um teste invariante ao movimento da câmera. O vencedor é
o robô que não sofreu ring-out; rounds sem ring-out são marcados para resolução
manual. Os limiares ficam em arquivo de configuração versionado e são reportados
tanto calibrados no conjunto gold quanto em valores padrão de literatura, para
mostrar que a detecção não depende exclusivamente do ajuste.

== Conjunto de dados e protocolo de anotação

Sem dataset público do domínio, construímos o conjunto a partir de footage real de
torneios. Anotar caixas quadro a quadro manualmente custaria horas por round; a
anotação semiautomática reduz o trabalho humano a revisão. O SAM 3 @carion2025sam3,
na implementação oficial de referência,
propaga máscaras de segmentação a partir de um prompt textual, as máscaras
viram caixas pela maior componente conexa, e as caixas são escaladas para a
resolução nativa em formato YOLO. Para caber em GPU de 8 GB, o SAM 3 processa cada
clip em janelas curtas, encerrando a sessão entre janelas para liberar memória.
Rounds de Sumô são curtos o bastante para uma janela cobrir um round inteiro.

A divisão treino/validação é feita por clip, não por quadro: quadros vizinhos de um
mesmo round são quase idênticos, então dividir por quadro vazaria informação entre
os conjuntos. O conjunto gold (teste) é anotado e revisado manualmente, fica fora
do treino e sustenta duas medições independentes: o SAM 3 contra o gold (validação do
anotador, por IoU) e o modelo treinado contra o gold (generalização).

== Protocolo de treino

Cada detector parte dos pesos COCO da sua arquitetura e treina com a mesma divisão,
`imgsz=640`, aumento padrão do Ultralytics (versão 8.4.55), semente fixa em 42 e parada antecipada no
platô da validação, para que a comparação entre arquiteturas isole o efeito do modelo.
Os hiperparâmetros são versionados no repositório, para reprodutibilidade.
