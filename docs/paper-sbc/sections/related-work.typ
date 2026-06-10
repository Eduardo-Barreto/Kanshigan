= Trabalhos Relacionados <sec-related>

Nenhum trabalho publicado trata da análise automatizada de partidas de Sumô de
Robôs, então a revisão se organiza pelos quatro corpos de literatura mais
próximos do problema: o rastreamento multiobjeto genérico, que fornece os
algoritmos de base; o rastreamento em esportes, que enfrenta alvos parecidos e
movimento rápido; a visão aplicada a combate e a robôs de competição, os
domínios mais análogos ao nosso; e os foundation models de segmentação, que
mudaram o custo de construir um dataset. Dessa análise emergem seis condições
(C1 a C6) que caracterizam o domínio; a @tab-matrix as consolida e situa o
recorte científico deste trabalho.

== Rastreamento multiobjeto por tracking-by-detection

O paradigma dominante em rastreamento multiobjeto é o *tracking-by-detection*:
um detector localiza os alvos quadro a quadro e um algoritmo de associação liga
as detecções no tempo. O SORT @bewley2016sort estabeleceu a formulação mínima:
um filtro de Kalman com modelo de velocidade constante prediz onde cada alvo
estará no próximo quadro, e a associação resolve um problema de atribuição por
sobreposição (IoU) entre predições e detecções. Mesmo sem usar aparência, o
SORT alcançou MOTA 34.0 no benchmark MOT15 com associação a 260 Hz, dezenas de
vezes mais rápido que os concorrentes da época, e fixou o desenho básico que as
variantes posteriores refinam.

O ByteTrack @zhang2022bytetrack refinou a etapa de associação: em vez de
descartar detecções de baixa confiança, associa primeiro as de alta confiança e
usa as restantes para recuperar alvos parcialmente visíveis, o que reduz buracos
de trajetória durante oclusão e borrão. Com isso atingiu MOTA 80.3, IDF1 77.3 e
HOTA 63.1 no MOT17, estado da arte na publicação. O OC-SORT @cao2023ocsort
atacou outra fragilidade: o filtro de Kalman com velocidade constante acumula
erro quando o alvo manobra ou some atrás de oclusão; o método recalcula os
parâmetros do filtro retroativamente a partir das observações que cercam a
oclusão, em vez de confiar na predição às cegas, mantendo a associação puramente
por movimento e alcançando HOTA 63.2 no MOT17, equivalente ao ByteTrack.

Esses métodos motion-only são o ponto de partida natural para o Sumô: robôs que
colidem, giram e ricocheteiam quebram exatamente a hipótese de velocidade
constante que o OC-SORT relaxa. Dessa análise sai a primeira característica do
nosso problema: *C1, movimento não linear*, em que colisões e mudanças bruscas de
direção são o evento central da modalidade, recorrente em todo round. Os benchmarks em que
esses métodos reportam resultados (MOT17, MOT20) são, porém, de pedestres
filmados por câmeras profissionais, alvos com roupas e aparências distintas; o
que acontece quando os alvos são indistinguíveis é o tema da próxima subseção.

== Rastreamento em esportes: alvos parecidos, movimento rápido

O DanceTrack @sun2022dancetrack foi construído para isolar essa pergunta: são
100 vídeos de dança em grupo, com dançarinos uniformizados e coreografias de
movimento diverso, em que a associação não pode se apoiar na aparência. O
resultado central do benchmark é a degradação dos rastreadores que dominavam o
MOT17: o ByteTrack cai de HOTA 63.1 no MOT17 para 47.7 no DanceTrack, e mesmo o
OC-SORT, projetado para movimento não linear e líder entre os métodos
motion-only, fica em 55.1. O paralelo com o Sumô é direto: dois robôs de combate
são caixas rasteiras propositalmente escuras e parecidas. Disso deriva *C2,
aparência uniforme entre alvos*: a identidade precisa vir do movimento, não do
visual.

O SportsMOT @cui2023sportsmot estende a análise a esportes reais, com 240
sequências de basquete, vôlei e futebol (cerca de 15 vezes mais quadros que o
MOT17) coletadas de jogos olímpicos, NCAA e NBA. O dataset caracteriza o desafio
como movimento de velocidade alta e variável, com arrancadas e paradas que
desafiam os modelos de movimento; mesmo o melhor método avaliado, o MixSort
proposto pelos autores, fica em HOTA 66.2 no basquete, o esporte mais difícil.
No Sumô, a escala temporal é mais extrema: o round inteiro frequentemente cabe
em menos de um segundo, e os eventos que decidem a partida (arrancada, contato,
ring-out) duram poucos quadros. Disso deriva *C3, eventos sub-segundo com
decisão crítica*, que nenhum benchmark esportivo força: perder dez quadros em um
rally de vôlei custa pouco; no Sumô, perde o round inteiro.

O SoccerNet-Tracking @cioppa2022soccernet completa o quadro com a escala de
broadcast profissional: 200 sequências de 30 segundos e um tempo completo de 45
minutos de futebol televisionado, multicâmera e em alta resolução. O vídeo
disponível de torneios de Sumô é o oposto: câmera de mão de espectador, ângulo
oblíquo e iluminação de ginásio, ao lado de footage cenital fixa de torneios
japoneses. Disso deriva *C4, vídeo de qualidade heterogênea*: a pipeline precisa
funcionar sobre o material que existe, não sobre o material que um broadcast
produziria.

== Visão em combate e em robôs de competição

Dois domínios chegam mais perto do Sumô de Robôs. No combate humano, o scoring
automático de jiu-jitsu de Hudovernik e Skočaj @hudovernik2022jiujitsu detecta
posições de luta a partir de vídeo de câmera fixa e pontua o combate por regras
da modalidade, combinando estimativa de pose dos atletas com pistas visuais para
resolver o contato corpo a corpo e a oclusão severa. O trabalho confirma que a
análise post-match de combate por vídeo é viável (*C5, análise post-match*: a
avaliação roda sobre o vídeo gravado, depois da luta, como no nosso caso), mas a
técnica central não transfere: robôs de Sumô são chassis rígidos sem
articulações, e não existe pose a estimar.

Nos robôs de competição, o SSL-Vision @zickler2010sslvision é o sistema de visão
compartilhado da RoboCup Small Size League: câmeras fixas sobre o campo e
padrões coloridos no topo de cada robô, que codificam identidade e orientação,
permitem localização global em tempo real, confiável a ponto de o sistema ser
obrigatório na liga desde 2010. É a prova de que rastrear robôs rápidos é
tratável quando se controla a instrumentação. O Sumô competitivo nega as duas
premissas: o regulamento não prevê marcadores, o acervo de vídeo existente não
os tem, e não há câmera calibrada fixa em torneio. Disso deriva *C6, ausência de
marcadores fiduciais*, que descarta a família de soluções instrumentadas e exige
detecção pela aparência natural dos robôs.

== Foundation models como anotadores

Construir um detector supervisionado para um domínio sem dataset público esbarra
no gargalo clássico da anotação. Modelos de fundação de vocabulário aberto
mudaram esse custo: o Grounding DINO @liu2024groundingdino detecta objetos a
partir de descrição textual sem treino no domínio, alcançando 52.5 AP no COCO em
regime zero-shot, e a família SAM @ravi2024sam2 @carion2025sam3 segmenta e
propaga máscaras em vídeo a partir de prompts, com o SAM 3 aceitando conceitos
textuais diretamente. O custo computacional desses modelos, porém, os mantém
distantes de inferência prática em vídeo longo, o que consolidou uma divisão de
papéis: o foundation model anota, e um modelo compacto treinado nessas anotações
roda a inferência. Adotamos essa divisão, com a validação do anotador contra
revisão humana descrita na metodologia.

== O recorte científico em aberto

As seis condições derivadas acima, numeradas na ordem em que emergiram da
análise, caracterizam o domínio: *C1* movimento não linear (da hipótese de
velocidade constante que colisões quebram); *C2* aparência uniforme entre alvos
(do resultado do DanceTrack); *C3* eventos sub-segundo com decisão crítica (da
escala temporal que nenhum benchmark esportivo força); *C4* vídeo de qualidade
heterogênea (do contraste com as condições de broadcast); *C5* análise
post-match (do regime de uso, como no scoring de jiu-jitsu); e *C6* ausência de
marcadores fiduciais (da premissa que o SSL-Vision exige e o Sumô nega).

Nem todas pesam igual para a ciência. *C5* descreve o regime de uso, não uma
dificuldade técnica: a análise é post-match, sobre vídeo gravado, como no scoring
de jiu-jitsu. *C6* também não mede dificuldade: satisfeita por quase todos, é o
critério que exclui as soluções instrumentadas, como o SSL-Vision. *C3* e *C4* descrevem a instância. O que permanece em aberto, com
maior valor científico transferível, é o par *C1+C2*: rastrear alvos de aparência
uniforme sob movimento não linear. O DanceTrack isola esse par e mostra a queda dos
melhores rastreadores, mas em captura controlada; sob vídeo não calibrado e sem
marcadores, e julgado pela acurácia e pelo custo de rodar em hardware
de consumo, o melhor equilíbrio segue sem resposta. É esse o recorte que a
pergunta deste trabalho ataca.

#figure(
  caption: [Cobertura das seis condições do domínio (#sym.circle.filled cobre, #sym.circle.filled.tiny parcial, #sym.circle não força). C1 movimento não linear; C2 aparência uniforme; C3 eventos sub-segundo; C4 vídeo heterogêneo; C5 análise post-match; C6 ausência de fiduciais. Mapa de condições, não placar de desempenho: o par C1+C2 sob o regime não controlado (C4, C6) é a célula em aberto.],
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center, center),
    stroke: 0.4pt,
    table.header(
      table.cell(rowspan: 2)[*Trabalho*],
      table.cell(colspan: 3, align: center)[*Desafio técnico*],
      table.cell(colspan: 3, align: center)[*Regime de dados e uso*],
      [*C1*], [*C2*], [*C3*], [*C4*], [*C5*], [*C6*],
    ),
    [ByteTrack @zhang2022bytetrack], [#sym.circle.filled.tiny], [#sym.circle.filled.tiny], [#sym.circle], [#sym.circle], [#sym.circle], [#sym.circle.filled],
    [OC-SORT @cao2023ocsort], [#sym.circle.filled], [#sym.circle.filled.tiny], [#sym.circle], [#sym.circle], [#sym.circle], [#sym.circle.filled],
    [DanceTrack @sun2022dancetrack], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle], [#sym.circle], [#sym.circle], [#sym.circle.filled],
    [SportsMOT @cui2023sportsmot], [#sym.circle.filled.tiny], [#sym.circle.filled], [#sym.circle], [#sym.circle], [#sym.circle], [#sym.circle.filled],
    [SoccerNet-Tracking @cioppa2022soccernet], [#sym.circle.filled.tiny], [#sym.circle.filled], [#sym.circle.filled.tiny], [#sym.circle], [#sym.circle.filled.tiny], [#sym.circle.filled],
    [Jiu-jitsu scoring @hudovernik2022jiujitsu], [#sym.circle.filled], [#sym.circle.filled.tiny], [#sym.circle.filled.tiny], [#sym.circle], [#sym.circle.filled], [#sym.circle.filled],
    [SSL-Vision @zickler2010sslvision], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle], [#sym.circle], [#sym.circle],
    [*Sumô de Robôs (este trabalho)*], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle.filled],
  ),
) <tab-matrix>

Os trabalhos mais próximos cobrem plenamente no máximo três das seis condições, e
nenhum reúne o par C1+C2 ao regime não controlado do Sumô: os rastreadores
genéricos e os benchmarks esportivos operam sobre captura controlada, e o
SSL-Vision depende da instrumentação que o Sumô não tem. Essa lacuna define uma
classe de problemas mais ampla (drone racing sem fiduciais, outras categorias de
combate de robôs, disputas rápidas entre alvos rígidos idênticos) para a qual a
contribuição é transferível, com o Sumô como veículo em que as condições aparecem
em estado puro.
