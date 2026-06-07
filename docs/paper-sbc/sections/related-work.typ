= Trabalhos Relacionados

A visão computacional aplicada a esportes convergiu fortemente para o paradigma
*tracking-by-detection*: um detector localiza os alvos quadro a quadro e um
rastreador associa as detecções ao longo do tempo. O par YOLO mais variantes de
SORT tornou-se baseline padrão em rastreamento multiobjeto @zhang2022bytetrack, e
o mesmo padrão se replica em benchmarks esportivos como SportsMOT e SoccerNet
@cui2023sportsmot @soccernet2023. Os trabalhos recentes que superam o ByteTrack
agregam compensação de movimento de câmera @aharon2022botsort, associação
observation-centric @cao2023ocsort ou fusão de características profundas
@deephmsort2024.

Quatro eixos são relevantes para o Kanshigan. No rastreamento de aparência
uniforme, o DanceTrack @sun2022dancetrack estabeleceu que rastreadores baseados só
em IoU falham quando os alvos são visualmente semelhantes, cenário análogo às
"caixas pretas" idênticas do Sumô. No movimento não linear, o OC-SORT
@cao2023ocsort foi projetado para trajetórias que quebram o filtro de Kalman padrão
e atinge estado da arte no DanceTrack. Em combate corpo a corpo, o scoring
automático em jiu-jitsu @jiujitsu2022 resolve oclusão severa com estimativa de
pose, inviável para chassis rígidos. Em robôs de competição, o SSL-Vision
@zickler2010sslvision depende de marcadores fiduciais coloridos, indisponíveis em
vídeo de torneio existente. Por fim, foundation models como Grounding DINO
@liu2024groundingdino e SAM 2 @ravi2024sam2 são adotados como anotadores
semiautomáticos, não como inferência final, por seu custo computacional ---
decisão que adotamos.

== A interseção que define o problema

O diferencial do Kanshigan não é o nicho "Sumô de Robôs", e sim a combinação de
restrições que o problema impõe. A literatura cobre cada restrição isoladamente;
nenhum trabalho cobre a interseção. Definimos seis características:
*C1* eventos sub-segundo com decisão crítica;
*C2* aparência uniforme entre alvos, sem marcadores;
*C3* vídeo de qualidade heterogênea (broadcast e amador);
*C4* ausência de marcadores fiduciais;
*C5* movimento não linear (colisões, giros, ricochetes);
*C6* análise post-match, não embarcada.

#figure(
  caption: [Cobertura das seis restrições do problema por trabalho. "S" cobre, "p" parcial, "n" não força a restrição.],
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    stroke: 0.4pt,
    table.header([*Trabalho*], [*C1*], [*C2*], [*C3*], [*C4*], [*C5*], [*C6*]),
    [ByteTrack @zhang2022bytetrack], [n], [p], [n], [S], [p], [n],
    [OC-SORT @cao2023ocsort], [n], [p], [n], [S], [S], [n],
    [DanceTrack @sun2022dancetrack], [n], [S], [n], [S], [S], [n],
    [SportsMOT @cui2023sportsmot], [n], [S], [n], [S], [p], [n],
    [SoccerNet Tracking @soccernet2023], [p], [S], [n], [S], [p], [p],
    [Jiu-jitsu scoring @jiujitsu2022], [p], [p], [n], [S], [S], [S],
    [SSL-Vision @zickler2010sslvision], [S], [S], [n], [n], [S], [n],
    [*Kanshigan*], [*S*], [*S*], [*S*], [*S*], [*S*], [*S*],
  ),
) <tab-matrix>

Como mostra a @tab-matrix, os trabalhos mais próximos cobrem no máximo quatro das
seis restrições, e cada um deixa restrições críticas de fora. O Kanshigan é a
primeira pipeline a operar sob a interseção completa. Essa interseção caracteriza
uma classe de problemas mais ampla --- drone racing sem fiduciais, outras
categorias de combate de robôs, disputas rápidas entre alvos rígidos idênticos ---
para a qual a contribuição é transferível, com o Sumô como veículo em que as
restrições aparecem em estado puro.
