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
padronizadas, lacuna que a revisão da @sec-related confirma; equipes ajustam
estratégia e hardware com base em memória e repetição manual de vídeo, sem
medições objetivas de velocidade, aceleração, tempo de reação ou padrões
espaciais de movimento.

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

Medições objetivas beneficiariam diretamente as equipes, que hoje não conseguem
responder com dados perguntas básicas de projeto: se o robô reage antes do
oponente, com que velocidade chega ao primeiro contato, por onde costuma ser
empurrado para fora. Para a comunidade de visão computacional, a modalidade
oferece um domínio de teste com restrições que raramente aparecem em conjunto,
como eventos críticos sub-segundo, alvos sem marcadores e visualmente uniformes
e vídeo de qualidade variável.

Este trabalho apresenta o Kanshigan, uma pipeline de visão computacional de
código aberto para extração automatizada de métricas de desempenho a partir do
vídeo de partidas de Sumô de Robôs autônomos de 3 kg. A pergunta que orienta a
pesquisa é: que combinação de arquitetura de detecção e de algoritmo de
rastreamento é suficiente, em acurácia e em viabilidade prática, para extrair
métricas de desempenho no domínio do Sumô de Robôs? A pergunta é de
suficiência, não de superioridade: com o conjunto de avaliação desta fase,
diferenças finas entre métodos não são distinguíveis de ruído, mas é possível
estabelecer se cada componente atende ao domínio. Para
respondê-la, comparamos arquiteturas de detector e algoritmos de rastreamento
sobre footage real de torneios, contra um conjunto de avaliação revisado por
humanos. Nesta entrega de meio de percurso, a componente de detecção da pergunta
é respondida nas duas fontes; a de rastreamento, medida em um único round com
identidades anotadas, é resultado preliminar por desenho e será fechada na
versão final, com mais rounds gold.

As contribuições deste artigo são: (i) a formulação do problema do Sumô de
Robôs pela interseção de seis restrições que a literatura cobre apenas
isoladamente; (ii) a primeira pipeline
aberta de detecção, rastreamento e extração de métricas para a modalidade,
reprodutível em hardware de consumo; e (iii) um conjunto de dados anotado,
representativo da categoria competitiva atual, derivado de torneios reais.

O restante do artigo se organiza assim: a @sec-related analisa os trabalhos
relacionados e deriva as seis restrições que definem o problema; a
@sec-metodologia descreve a pipeline e o protocolo experimental; a
@sec-resultados reporta os resultados parciais; e a @sec-discussao discute
limitações e trabalhos futuros.
