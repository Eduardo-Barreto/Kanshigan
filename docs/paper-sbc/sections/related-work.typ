= Trabalhos Relacionados <sec-related>

A visão computacional já aparece no Sumô de Robôs, mas embarcada no robô, não como análise externa: o robô detecta o oponente por infravermelho, ultrassom ou câmera própria durante a luta. Até onde nossa busca alcança, a análise pós-partida, feita por um observador externo a partir de vídeo de terceira pessoa, não tem pipeline, conjunto de dados nem _benchmark_ publicado#footnote[Busca de junho de 2026 em mecanismo de busca web e no arXiv, com quatro strings que combinam "robot sumo", "sumo robot" e "combat robot" a termos de visão, detecção, rastreamento, métricas e arbitragem. Strings, achados e categorização estão versionados no repositório. Não cobrimos bases autenticadas como IEEE Xplore e Scopus.]. Organizamos a revisão em três linhas próximas do problema e delas derivamos seis condições (C1 a C6) que caracterizam o domínio (@tab-matrix).

== Rastreamento multiobjeto e esportes

O paradigma dominante em rastreamento multiobjeto é o rastreamento por detecção. O SORT @bewley2016sort fixou a formulação mínima: um filtro de Kalman de velocidade constante prediz cada alvo no quadro seguinte, e uma atribuição por sobreposição (IoU) liga as detecções, sem usar aparência. O ByteTrack @zhang2022bytetrack passou a associar também as detecções de baixa confiança, o que reduz buracos de trajetória sob oclusão e borrão (HOTA 63,1 no MOT17). O OC-SORT @cao2023ocsort atacou outra fragilidade: o Kalman de velocidade constante acumula erro quando o alvo manobra ou some. O método recalcula o filtro a partir das observações em torno da oclusão e atinge HOTA 63,2 no MOT17. Robôs que colidem, giram e ricocheteiam quebram justamente a hipótese de velocidade constante. Daí a primeira condição, *C1, movimento não linear*: colisões e mudanças bruscas de direção ocorrem em todo round.

Os benchmarks do MOT17 filmam pedestres de roupas distintas, que a aparência ajuda a separar. O DanceTrack @sun2022dancetrack isola o caso contrário: 100 vídeos de dança em grupo, com dançarinos uniformizados e movimento diverso. Nele, os rastreadores que dominavam o MOT17 degradam: o ByteTrack cai para HOTA 47,7 e o OC-SORT fica em 55,1 (valores da tabela do DanceTrack). Sucessores que reforçam pistas geométricas, como o C-BIoU @yang2023cbiou e o Hybrid-SORT @yang2024hybridsort, chegam a 60,6 e 62,2. Dois robôs de combate são caixas escuras e parecidas, o que define *C2, aparência uniforme entre alvos*: a identidade precisa vir do movimento.

O SportsMOT @cui2023sportsmot leva o problema a 240 vídeos de basquete, vôlei e futebol, com velocidade alta e variável. O método dos autores atinge HOTA 74,1, e o basquete, de mais contato físico, é o esporte mais difícil. O Deep-EIoU @huang2024deepeiou e o Deep HM-SORT @deephmsort2024 combinam caixas expandidas com aparência para jogadores. No Sumô de Robôs a escala temporal é mais extrema: o contato que decide o round dura poucos quadros. Daí *C3, eventos decisivos de poucos quadros*: perder dez quadros num rali de vôlei custa pouco, mas no Sumô esses quadros podem conter o empurrão que decide a partida. O SoccerNet-Tracking @cioppa2022soccernet, por fim, usa a câmera principal de transmissões profissionais de futebol, em alta resolução. O vídeo de torneios de Sumô é o oposto: câmera de mão de espectador, ângulo oblíquo e iluminação de ginásio, ao lado de câmera cenital fixa no Japão. Daí *C4, vídeo de qualidade heterogênea*.

== Alvos idênticos fora dos esportes

O rastreamento de alvos indistinguíveis é um campo maduro na biologia. O idtracker.ai @romeroferrero2019idtracker identifica até 100 animais sem marcadores, com acerto acima de 99,9%. O TRex @walter2021trex rastreia centenas de indivíduos, e o SLEAP @pereira2022sleap rastreia poses de vários animais. Esses sistemas, porém, aprendem a identidade a partir do próprio vídeo. Exigem gravações longas, de câmera fixa e com fundo estável, em que todos os alvos apareçam separados ao menos uma vez. O Sumô de Robôs nega essas premissas: um round dura segundos, os robôs passam boa parte dele em contato e a câmera brasileira está na mão. Não há tempo nem condição de captura para aprender a aparência de cada robô.

== Combate e robôs de competição

Dois domínios chegam mais perto do Sumô de Robôs. No combate humano, #cite(<hudovernik2022jiujitsu>, form: "prose") pontuam lutas de jiu-jitsu a partir de vídeo de câmera fixa, com estimativa de pose para resolver o contato entre corpos. O trabalho mostra que analisar combate em vídeo gravado é viável. Esse regime define *C5, análise pós-partida*. A técnica, porém, não se aplica a chassis rígidos, que não têm pose a estimar.

Nos robôs de competição, o SSL-Vision @zickler2010sslvision, sistema oficial da RoboCup Small Size League desde 2010, localiza robôs rápidos com câmeras fixas e padrões coloridos no topo de cada robô. Rastrear robôs rápidos é tratável quando se controla a instrumentação. O Sumô de Robôs nega essa premissa: o regulamento não prevê marcadores, o acervo de vídeo não os tem e não há câmera calibrada. Daí *C6, ausência de marcadores fiduciais*, que descarta as soluções instrumentadas e exige detectar os robôs pela aparência natural.

== O recorte em aberto

Nem todas as condições pesam igual. C5 descreve o regime de uso, não uma dificuldade técnica. C6 também não mede dificuldade: quase todos os trabalhos da @tab-matrix operam sem marcadores, e a condição serve para excluir as soluções instrumentadas. C3 e C4 descrevem o caso do Sumô de Robôs, não o problema geral. O que permanece em aberto, com valor transferível, é o par *C1+C2* sob vídeo não controlado (C4, C6): rastrear alvos de aparência uniforme em movimento não linear. O DanceTrack isola esse par, mas com câmera estável e alvos inteiros no quadro. A biologia o resolve com gravações longas de câmera fixa. Sob vídeo de torneio, curto e não calibrado, e julgado pela acurácia e pelo custo de rodar em hardware de consumo, o melhor equilíbrio segue sem resposta. É esse o recorte que a pergunta deste trabalho ataca.

#figure(
  caption: [Cobertura das seis condições do domínio (#sym.circle.filled cobre, #sym.circle.filled.tiny parcial, #sym.circle não força). C1 movimento não linear; C2 aparência uniforme; C3 eventos decisivos de poucos quadros; C4 vídeo heterogêneo; C5 análise pós-partida; C6 ausência de marcadores. Em C5 e C6, #sym.circle.filled indica que o trabalho opera nesse regime. É um mapa de condições, não um placar de desempenho; a última linha descreve o domínio-alvo, não um resultado.],
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
    [idtracker.ai @romeroferrero2019idtracker], [#sym.circle.filled.tiny], [#sym.circle.filled], [#sym.circle], [#sym.circle], [#sym.circle.filled], [#sym.circle.filled],
    [Jiu-jitsu @hudovernik2022jiujitsu], [#sym.circle.filled], [#sym.circle.filled.tiny], [#sym.circle.filled.tiny], [#sym.circle], [#sym.circle.filled], [#sym.circle.filled],
    [SSL-Vision @zickler2010sslvision], [#sym.circle.filled], [#sym.circle], [#sym.circle.filled], [#sym.circle], [#sym.circle], [#sym.circle],
    [_Sumô de Robôs (domínio-alvo)_], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle.filled], [#sym.circle.filled],
  ),
) <tab-matrix>

Hipotetizamos que a resposta vale para outras disputas rápidas entre alvos rígidos parecidos, como outras categorias de combate de robôs; a @sec-sem-rotulos traz um primeiro indício com o Sumô de rádio-controle.
