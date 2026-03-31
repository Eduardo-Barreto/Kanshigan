= Introdução

Competições de Sumô de Robôs autônomos (3kg) são disputas de robótica móvel de altíssima velocidade onde rounds inteiros duram frações de segundo. Com aproximadamente 80.000 competidores ativos em mais de 30 países, o esporte possui uma comunidade global expressiva que participa de torneios como o All Japan Robot Sumo Tournament @fujisoft-about e competições regionais como a RoboCore (Brasil) e RoboGames (EUA) @robogames-rules.

Apesar dessa escala, toda a análise de desempenho é inteiramente manual e subjetiva. Não existem ferramentas automatizadas, bancos de dados estruturados ou métricas padronizadas para o domínio. A ausência de dados quantitativos afeta três grupos diretamente:

- *Equipes de robótica competitiva* não têm dados reais de combate para embasar decisões de design, projetando novos robôs com base em intuição e memória.
- *Organizadores de competições* não possuem documentação histórica estruturada nem ferramentas de análise objetiva.
- *Espectadores* acompanham transmissões ao vivo sem qualquer estatística de desempenho que enriqueça a experiência de visualização. Em contraste com esportes tradicionais (futebol, baseball, basquete), onde estatísticas em tempo real são parte fundamental da transmissão, o Sumô de Robôs oferece zero dados ao espectador.

Na literatura de visão computacional, a análise esportiva tem avançado significativamente em domínios como futebol @giancola2018soccernet, esportes de combate e tênis de mesa. No entanto, nenhum trabalho aborda análise externa automatizada de partidas de Sumô de Robôs. O domínio apresenta desafios técnicos únicos: velocidade extrema (rounds em frações de segundo), objetos pequenos e visualmente similares, oclusão mútua durante contato, e qualidade de vídeo heterogênea variando de broadcasts profissionais a gravações amadoras.

== Problema de pesquisa

Como a escolha de arquitetura de visão computacional influencia a acurácia e viabilidade prática da classificação automática de partidas de Sumô de Robôs?
