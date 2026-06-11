# Roteiro: apresentação pré-banca (10 minutos)

Apresentação: `index-banca.html` (abrir no navegador, navegar com setas ou espaço). 22 slides. Os tempos-alvo de entrada estão neste roteiro, ao lado de cada slide.

A fala abaixo foi escrita pra ser dita, não lida. Cada slide tem uma **frase-síntese**: se o tempo apertar, fale só ela.

O fio da apresentação: a banca precisa sair sabendo (1) qual problema a gente investiga e qual lacuna ele abre, (2) que a metodologia está definida e já rodou com resultado medido, (3) que o artigo existe inteiro, (4) pra onde ele vai e por quê. A apresentação inteira percorre esses quatro pontos nessa ordem, sem anunciá-los.

---

## Slide 1: Título (0:00, 20s)

**Visual:** nome do projeto, autores, orientador, instituição.

**Frase-síntese:** o Kanshigan mede partidas de Sumô de Robôs a partir do vídeo.

> Bom dia. Eu sou o Eduardo e vou apresentar o Kanshigan: uma pipeline de visão computacional que mede partidas de Sumô de Robôs a partir do vídeo da luta. Essa é uma entrega de meio de percurso, então tudo que eu mostrar já foi medido. Nada aqui é projeção.

## Slide 2: Visão computacional hoje (0:20, 30s)

**Visual:** GIF clássico de tracking (ByteTrack em pedestres, MOT20), tela quase cheia.

**Frase-síntese:** rastrear o mundo virou rotina.

> Isso aqui a visão computacional já faz há anos. Caixa em cada pessoa, identidade que persiste, trajetória, tudo em tempo real, numa estação de trem lotada. É um problema maduro, com benchmark, métrica e ranking.

## Slide 3: O que eu quero medir (0:50, 30s)

**Visual:** vídeo da final do mundial, sem tracking, tela quase cheia. Deixar rodar.

**Frase-síntese:** quero isso, mas aqui.

> E eu quero fazer exatamente isso neste cenário. Final do mundial japonês, categoria 3 kg autônomo. Repara na velocidade: o round acaba em menos de um segundo. [deixar o round correr] Isso que vocês viram foi uma luta inteira.

## Slide 4: Hoje é no olho (1:20, 20s)

**Visual:** frase grande, dois chips: "nenhuma métrica", "replay manual no celular".

**Frase-síntese:** ninguém mede nada disso hoje.

> E hoje, tudo isso é analisado no olho. Quem venceu, quem rampou quem, com que velocidade: decidido de memória e no replay de celular. Eu compito nessa categoria, e é assim em todo torneio.

## Slide 5: Rastreadores genéricos → C1 (1:40, 30s)

**Visual:** primeiras páginas de ByteTrack e OC-SORT à esquerda; derivação à direita; sela com C1.

**Frase-síntese:** o Kalman assume velocidade constante; robô de Sumô quebra isso.

> Então fui ver o que a literatura resolve. A base é o tracking-by-detection: um detector acha os alvos e um filtro de Kalman prevê onde cada um estará no próximo quadro. ByteTrack e OC-SORT são o estado da arte disso. Mas o Kalman assume velocidade constante. Robô de Sumô colide, gira e ricocheteia: a previsão quebra. Essa é a primeira característica do nosso domínio, C1, movimento não linear.

## Slide 6: DanceTrack → C2 (2:10, 30s)

**Visual:** página do DanceTrack, GIF de dançarinos uniformizados, sela C2.

**Frase-síntese:** alvos iguais derrubam os melhores rastreadores.

> A segunda vem do DanceTrack. Dançarinos de uniforme, movimento brusco. Quando os alvos se parecem, os melhores rastreadores despencam: o ByteTrack cai de 63 pra 48 de HOTA. O paralelo com a gente é direto: dois robôs de combate são duas caixas pretas. A identidade vem do movimento, não do visual. C2, aparência uniforme.

## Slide 7: Esportes → C3 e C4 (2:40, 30s)

**Visual:** SportsMOT à esquerda, SoccerNet à direita, uma sela em cada metade.

**Frase-síntese:** no Sumô o round cabe num segundo, e o vídeo é de câmera de mão.

> Os benchmarks de esporte chegam perto, mas dois detalhes mudam tudo. No vôlei, perder dez quadros de um rally custa pouco. No Sumô, o round inteiro cabe num segundo: perdeu dez quadros, perdeu o round. C3, eventos sub-segundo. E o SoccerNet trabalha com broadcast profissional. Nosso material é câmera de mão de espectador, em ginásio. C4, vídeo heterogêneo.

## Slide 8: Jiu-jitsu → C5 (3:10, 25s)

**Visual:** cartão de citação do paper, derivação à direita, sela C5.

**Frase-síntese:** análise de combate gravado funciona, mas pose humana não transfere.

> No combate humano, tem um trabalho que pontua jiu-jitsu a partir do vídeo gravado. Prova que análise de combate depois da luta é viável: é o nosso regime de uso, C5, análise post-match. Só que a técnica dele é estimativa de pose. Robô é chassi rígido. Não tem pose pra estimar.

## Slide 9: SSL-Vision → C6 (3:35, 25s)

**Visual:** página do SSL-Vision, foto do robô com padrão colorido, sela C6.

**Frase-síntese:** marcador resolve o problema, e o Sumô não permite marcador.

> Nos robôs de competição, a RoboCup resolve isso desde 2009: padrão colorido no topo de cada robô, câmera fixa calibrada. Rastrear robô rápido fica fácil quando você controla a instrumentação. O Sumô nega tudo isso. O regulamento não prevê marcador e o acervo de vídeo que existe não tem. C6, sem fiduciais.

## Slide 10: A matriz e a lacuna (4:00, 45s) [TEM BUILD: apertar seta uma vez a mais]

**Visual:** matriz trabalhos × C1-C6. Ao avançar, C1 e C2 acendem, o resto esmaece e aparece a coluna "regime de captura" mostrando o porém de cada linha, mais a barra com a lacuna.

**Frase-síntese:** quem acerta o par C1+C2 só acerta em captura controlada; em vídeo real, está em aberto.

> Juntando tudo: seis condições caracterizam o domínio, e cada vizinho cobre no máximo três delas. [avançar o build] O recorte científico está nestas duas: rastrear alvos de aparência idêntica sob movimento não linear. E reparem que alguns trabalhos pintam bem essas duas colunas. Mas olhem o regime de cada um. O DanceTrack acerta o par em estúdio, com captura controlada. A RoboCup acerta com marcador colado no robô e câmera calibrada. O jiu-jitsu, com câmera fixa e pose humana. Tira o estúdio, o marcador e a câmera fixa, e cobra que rode em hardware comum: ninguém respondeu. Essa é a lacuna que a nossa pesquisa ataca. E ela não é exclusiva do Sumô: o mesmo par aparece em drone racing e em qualquer disputa rápida entre alvos iguais. O Sumô é só o caso mais puro.

## Slide 11: A pergunta de pesquisa (4:45, 25s)

**Visual:** só a pergunta, grande, com "visualmente semelhantes" e "movimento rápido e não linear" em verde, ecoando C1+C2 do slide anterior.

**Frase-síntese:** a pergunta junta o par em aberto com a exigência de rodar em hardware comum.

> Tudo isso se condensa na seguinte pergunta de pesquisa: quais combinações de detector e rastreador melhor equilibram acurácia e viabilidade prática pra rastrear alvos quase idênticos, em movimento rápido e não linear, a partir de vídeo não calibrado e sem marcador? E viabilidade prática tem definição concreta: rodar no computador que as equipes têm, com velocidade e memória medidas. O resto da apresentação é a resposta em andamento.

## Slide 12: Metodologia, visão geral (5:10, 25s)

**Visual:** diagrama da pipeline (8 estágios + ramo offline do anotador).

**Frase-síntese:** pesquisa experimental, oito estágios, comparação contra gold.

> E como a gente responde isso? É uma pesquisa experimental: a gente compara as alternativas medindo cada uma contra um conjunto de referência revisado por humano, que a gente chama de gold. Quem sustenta esses experimentos é essa pipeline de oito estágios. A fileira de cima prepara o quadro, a de baixo extrai a informação, e esse ramo tracejado embaixo é a anotação dos dados de treino, que acontece offline, fora da hora de rodar. Deixa eu passar por cada parte.

## Slide 13: Arena e calibração (5:35, 30s)

**Visual:** frame com a elipse detectada; bullets à direita.

**Frase-síntese:** a arena é fixa por regulamento, então visão clássica resolve e ainda dá a escala.

> A primeira tarefa é achar a arena no vídeo. E pra isso a gente não treina modelo nenhum, porque o dohyo é padronizado por regulamento: sempre um disco escuro com uma borda branca. Então a gente simplesmente procura essa borda clara no quadro e ajusta uma elipse nela, visão computacional clássica. Essa elipse vale muito, porque o regulamento também fixa o diâmetro: 154 centímetros. Se eu sei quantos pixels esse diâmetro ocupa na imagem, eu ganho a conversão de pixel pra centímetro de graça, sem precisar calibrar câmera. E como a câmera de mão se mexe o tempo todo, a gente refaz essa detecção a cada quadro.

## Slide 14: Recorte e detecção (6:05, 25s)

**Visual:** quadro inteiro → recorte do dohyo com caixas; chips de precisão e recall.

**Frase-síntese:** o recorte amplia o robô 3x e tira a plateia; o YOLO roda em cima.

> Com a arena na mão, vem um truque que mudou o projeto. O robô é minúsculo no quadro inteiro, e o detector sofria: perdia robô em movimento e confundia gente da plateia com robô. Recortando o quadro na elipse da arena, o robô fica três vezes maior e a plateia simplesmente some. Só esse passo levou a precisão de 0.71 pra 0.99, medido com o YOLOv8s no nosso gold brasileiro. Em cima desse recorte a gente treinou dois detectores, um padrão e um com um quinto do tamanho, pra saber se o domínio exige modelo grande. E a resposta foi que não: o pequeno empata com o grande, o que é ótimo pra rodar em qualquer máquina.

## Slide 15: Rastreamento (6:30, 20s)

**Visual:** dois cartões com um número grande cada: só movimento (0.94 IDF1, 3448 fps) vs movimento + aparência (0.94 IDF1, 94 fps).

**Frase-síntese:** pra manter a identidade, movimento basta; aparência empata e custa 35-40x mais.

> Depois de detectar, a pipeline precisa manter a identidade: saber qual robô é o A e qual é o B em todos os quadros. Como os dois são quase idênticos, a dúvida era se valia a pena usar a aparência pra distinguir, ou se o movimento bastava. A gente testou os dois caminhos, quatro rastreadores no total, todos sobre as mesmas detecções. Olha o resultado: a acurácia empata, 0.94 de IDF1 dos dois lados. Mas o lado que usa aparência roda 35 a 40 vezes mais lento. Movimento basta.

## Slide 16: Dataset e anotador (6:50, 25s)

**Visual:** SAM 3 anotando a fonte japonesa; tabela do dataset; F1 do anotador.

**Frase-síntese:** sem dataset público, construímos um; o SAM 3 anota e foi validado contra humano.

> Pra treinar esses detectores, não existia dataset público da modalidade. A gente construiu um, com duas fontes escolhidas por serem opostas: torneio brasileiro com câmera na mão, e torneio japonês com câmera fixa vista de cima. Anotar isso quadro a quadro na mão custaria horas por round. Quem anota pra gente é o SAM 3, um modelo de segmentação da Meta: você descreve o que quer em texto, "robô de sumô", e ele devolve as máscaras disso ao longo do vídeo. Ele é pesado demais pra rodar ao vivo, mas como anotador, fora da hora, serve bem. E a gente conferiu o anotador contra revisão humana: 96 por cento de F1. O gold, que é o conjunto de teste, é revisado na mão e nunca entra no treino.

## Slide 17: Números (7:15, 30s)

**Visual:** três cartões: mAP, IDF1, FPS.

**Frase-síntese:** detecção acima de 0.96, IDF1 0.94, mais de 100 fps; tudo medido.

> Juntando os resultados que já temos, todos medidos em vídeo real. Na detecção, mAP acima de 0.96 nas duas fontes, e o modelo com um quinto dos parâmetros empata com o grande. No rastreamento, IDF1 de 0.94, e a aparência não ajudou: zero ganho de acurácia custando 35 a 40 vezes mais tempo. Isso confirma na prática a condição de aparência uniforme, a C2: robô igual não dá sinal visual pra explorar. E de viabilidade, a pipeline completa roda acima de 100 quadros por segundo, com 100 mega de memória de vídeo, num notebook.

## Slide 18: Em vídeo real (7:45, 30s)

**Visual:** quatro vídeos rodando: BR câmera de mão, JP cenital, outra arena BR, Sumô RC.

**Frase-síntese:** a mesma pipeline em quatro contextos, sem retreinar.

> Isso é a pipeline rodando. Quatro contextos, o mesmo modelo, nenhum retreino. Brasil com câmera de mão. Japão visto de cima. Outra arena brasileira. E o quarto é Sumô de rádio controle: uma categoria que nem estava no treino. Funciona de primeira.

## Slide 19: O artigo (8:15, 25s)

**Visual:** páginas do artigo em leque; lista das quatro frentes de escrita.

**Frase-síntese:** o artigo existe inteiro, e não está sozinho.

> Sobre o texto: o artigo já existe. Dezesseis páginas no formato da SBC, com introdução, estudos relacionados, metodologia e os resultados parciais. É a versão que vocês receberam. Desde o início a gente mantém em paralelo a versão em inglês no formato IEEE, o projeto de pesquisa em ABNT, e um diário de bordo com 19 entradas que registra o processo: o que falhou, por que falhou e o que a gente fez.

## Slide 20: Publicação (8:40, 20s)

**Visual:** WVC em destaque; arXiv e versão IEEE ao lado.

**Frase-síntese:** WVC porque o escopo casa; arXiv pra ter link citável; inglês de reserva.

> O alvo de submissão é o WVC, o Workshop de Visão Computacional da SBC. A justificativa é de escopo: é o evento da comunidade de visão aplicada no Brasil, no idioma do artigo, e o trabalho é exatamente isso. Em paralelo, um preprint no arXiv pra ter link citável desde já. E a versão em inglês fica pronta caso a gente mire um evento internacional depois.

## Slide 21: Limitações e o teste mais duro (9:00, 40s)

**Visual:** vídeo da pipeline na final do mundial; três pares limitação → ação.

**Frase-síntese:** as limitações estão assumidas no texto, e cada uma tem um próximo passo.

> As limitações estão escritas no artigo, não escondidas. O gold ainda é pequeno, então os números são ordem de grandeza: a resposta é anotar mais rounds. As métricas em centímetros e os eventos ainda não têm validação quantitativa: ficam como trabalho futuro, com homografia da arena e um conjunto de calibração. E esse vídeo é o teste mais duro: a pipeline na final do mundial. Transmissão de TV, colisão violenta, borrão extremo. No momento mais rápido o detector perde os dois robôs. Eu mostro de propósito: a gente sabe exatamente onde quebra e o que falta, que é treinar com vídeo dessa distribuição.

## Slide 22: Conclusão (9:40, 20s)

**Visual:** três contribuições; link do repositório.

**Frase-síntese:** três contribuições medidas e públicas; os detalhes estão no artigo.

> Três contribuições até aqui. A caracterização do domínio em seis condições, com o par em aberto. A primeira pipeline aberta pra modalidade, medida. E o dataset, validado. Tudo isso está no artigo com bem mais detalhe do que coube aqui, e tudo é público, no repositório. Obrigado.

---

## Perguntas prováveis (preparação, não apresentar)

| Pergunta | Resposta curta |
|---|---|
| Por que isso é ciência e não uma ferramenta de nicho? | O par alvos idênticos + movimento não linear é um problema aberto de rastreamento. O DanceTrack mostrou a queda em estúdio; a gente ataca em vídeo real, e a resposta transfere pra outras disputas rápidas entre alvos iguais. |
| Que tipo de pesquisa? | Experimental, quantitativa, comparação controlada contra gold revisado. |
| Onde vão submeter? | WVC (SBC). Preprint no arXiv em paralelo. Versão IEEE em inglês de reserva. |
| O artigo existe? | Sim, completo, 16 páginas SBC, todas as seções. A banca recebeu. |
| Por que SAM 3 como anotador? | Anotar na mão custa horas por round. O anotador foi validado contra humano: F1 0.96. Inviável como pipeline final: 2 fps e 7 GB de VRAM. |
| Um round de gold basta? | Não, e o artigo assume isso. Detecção está respondida; rastreamento é preliminar. Mais gold está no plano. |
| E as métricas em centímetros? | Sem validação quantitativa ainda. Perspectiva da câmera distorce; homografia é o próximo passo. Por isso não apresentei número de velocidade. |
| Dá pra reproduzir? | Semente fixa, versões travadas, dados versionados, instruções no README. |

## Checklist antes do dia

- [ ] Testar no projetor: vídeos tocam sozinhos (mudos), fontes caem pro fallback se não tiver internet.
- [ ] Lembrar do build no slide 10: uma seta a mais pra acender C1+C2.
- [ ] Cronometrar uma passada. Estourou 10 min: encurtar slides 17 e 21 falando só a frase-síntese.
- [ ] Levar o PDF do artigo aberto pras tabelas na hora das perguntas.
