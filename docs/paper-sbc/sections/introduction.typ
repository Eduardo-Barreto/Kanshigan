= Introdução

O Sumô de Robôs é uma modalidade de competição robótica em que dois robôs
disputam, sobre uma arena circular chamada dohyo, quem empurra o oponente para
fora. A modalidade nasceu no Japão em 1989, com o torneio que se tornou o All
Japan Robot-Sumo Tournament, organizado pela Fujisoft e reconhecido como a
competição de sumô de robôs mais longeva do mundo @guinness_robotsumo. Na
categoria principal, robôs autônomos de até 3 kg competem sobre um dohyo de
154 cm de diâmetro, sem qualquer intervenção humana durante o round
@fujisoft2024rules. No Brasil, a modalidade tem calendário consolidado: a
RoboCore Experience é o maior evento de robótica de combate da América Latina
@robocore_experience, e torneios sancionados como a IRONCup, do Inatel, reúnem
as categorias de sumô a cada edição @ironcup2026. Além do circuito competitivo,
a modalidade é adotada como estratégia de ensino de engenharia, pela combinação
de projeto mecânico, eletrônica e programação em um problema fechado e
mensurável @carbone2022robotsumo.

A categoria autônoma de 3 kg cria um problema particular de análise. Os robôs
partem com aceleração altíssima e, no footage de torneio que compõe o conjunto
de dados deste trabalho, a maioria dos rounds termina em menos de um segundo:
decisões de vitória, momento do primeiro contato e trajetórias acontecem em
poucas dezenas de quadros de vídeo, rápido demais para observação a olho nu.
Ainda assim, toda a análise de desempenho na modalidade é feita por observação
humana, sem ferramentas automatizadas, bases de dados estruturadas ou métricas
padronizadas; equipes ajustam estratégia e hardware com base em memória e
repetição manual de vídeo, sem medições objetivas de velocidade, aceleração,
tempo de reação ou padrões espaciais de movimento.

Esse vazio contrasta com o estado da análise esportiva para atletas humanos. A
visão computacional transformou a análise de desempenho em esportes como
futebol, basquete e tênis, com sistemas de rastreamento de jogadores e bola que
alimentam estatísticas, decisões de arbitragem e treinamento
@thomas2017computer @naik2022comprehensive. Essa infraestrutura, porém, foi
construída sobre premissas que não valem para o combate de robôs: atletas
distinguíveis pela aparência, transmissões profissionais multicâmera e eventos
em escala de segundos. Para o Sumô de Robôs, com alvos rígidos quase idênticos,
vídeo heterogêneo de torneio e desfechos sub-segundo, não encontramos na
literatura pipeline, dataset ou benchmark publicado.

Medições objetivas dariam às equipes respostas que hoje faltam: se o robô reage
antes do oponente, com que velocidade chega ao primeiro contato, por onde costuma
ser empurrado para fora. Para a visão computacional, a modalidade é um domínio de
teste com condições que raramente aparecem juntas: eventos críticos sub-segundo,
alvos sem marcadores e visualmente uniformes, vídeo de qualidade variável.

Este trabalho apresenta o Kanshigan, uma pipeline de visão computacional de
código aberto para extração automatizada de métricas de desempenho a partir do
vídeo de partidas de Sumô de Robôs autônomos de 3 kg. O domínio instancia, em
estado concentrado, um problema de rastreamento mais geral, e é sobre ele que a
pesquisa se orienta: quais combinações de detector e rastreador melhor equilibram
acurácia e viabilidade prática para rastrear múltiplos alvos visualmente
semelhantes, em movimento rápido e não linear, a partir de vídeo não calibrado e
sem marcadores? Por
viabilidade prática entendemos rodar em hardware de consumo, com throughput e
memória medidos e sem instrumentação especial: a análise só serve às equipes se
rodar no computador que elas têm.

A pergunta busca a combinação superior em acurácia e viabilidade, e a respondemos
comparando arquiteturas de detector e algoritmos de rastreamento sobre footage
real de torneios, contra um conjunto de avaliação revisado por humanos. Esta é
uma entrega de meio de percurso, e a reportamos como tal: a componente de
detecção é respondida nas duas fontes; a de rastreamento, medida em um round com
identidades anotadas, é preliminar e será fechada na versão final, com mais
rounds gold.

As contribuições deste artigo são: (i) a caracterização do Sumô de Robôs por
seis condições que a literatura enfrenta apenas em separado, e o recorte da
questão científica ao subconjunto de fato em aberto, rastrear alvos de aparência
uniforme sob movimento não linear no equilíbrio entre acurácia e viabilidade;
(ii) a primeira pipeline aberta de detecção, rastreamento e extração de métricas
para a modalidade, reprodutível em hardware de consumo; e (iii) um conjunto de
dados anotado, representativo da categoria competitiva atual, derivado de
torneios reais.
