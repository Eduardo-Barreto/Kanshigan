= Introdução

O Sumô de Robôs é uma modalidade de competição em que dois robôs autônomos
disputam, sobre uma arena circular chamada dohyo, quem empurra o oponente para
fora. Na categoria de 3 kg, os robôs partem com aceleração altíssima e a maioria
dos rounds termina em menos de um segundo. Essa dinâmica extrema torna a análise
de desempenho difícil até para juízes experientes: decisões de vitória, momento de
primeiro contato e trajetória dos robôs acontecem rápido demais para observação a
olho nu.

Apesar disso, toda a análise de partidas hoje é subjetiva. Não há ferramentas
automatizadas, bases de dados estruturadas nem métricas padronizadas para a
modalidade. Equipes ajustam estratégia e hardware com base em memória e repetição
manual de vídeo, sem medições objetivas de velocidade, aceleração, tempo de reação
ou padrões espaciais de movimento.

Este trabalho apresenta o Kanshigan, uma pipeline de visão computacional de código
aberto para extração automatizada de métricas de desempenho a partir do vídeo de
partidas de Sumô de Robôs autônomos de 3 kg. A pergunta que orienta a pesquisa é:
quais escolhas de arquitetura de detecção e de algoritmo de rastreamento compõem
uma pipeline que melhor equilibra acurácia e viabilidade prática para essa
extração de métricas no domínio do Sumô de Robôs?

A pipeline detecta a arena por visão clássica, calibra a escala em centímetros a
partir do diâmetro conhecido do dohyo, detecta os robôs com um detector treinado,
rastreia ambos com identidade consistente e deriva posição, velocidade, aceleração
e eventos (início do round, primeiro contato, ring-out). Para construir a base de
treino sem um dataset público do domínio, usamos o SAM 3 como anotador
semiautomático com revisão humana, deixando o foundation model fora da inferência
final por seu custo computacional.

As contribuições deste artigo são: (i) a formulação do problema do Sumô de Robôs
pela interseção de seis restrições que a literatura cobre apenas isoladamente,
sustentando o diferencial científico; (ii) a primeira pipeline aberta de detecção,
rastreamento e extração de métricas para a modalidade, reprodutível em hardware de
consumo; e (iii) um conjunto de dados anotado, representativo da categoria
competitiva atual, derivado de torneios reais. Reportamos resultados parciais
medidos sobre footage real, com honestidade quanto às limitações de um conjunto de
avaliação pequeno.
