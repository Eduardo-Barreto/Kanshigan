# Roteiro da apresentação: pré-banca (10 minutos)

Apresentação: `index-banca.html` (abrir no navegador, navegar com setas, espaço ou Page Down). 15 slides, mais ou menos 40 segundos cada. Os tempos aparecem no canto de cada slide.

A fala está escrita pra ser dita em voz alta, não lida. Frases curtas, sem rodeio. Cada slide tem uma **frase-síntese** em negrito. Se o tempo apertar, fale só ela e o número principal.

Uma coisa pra ter clara o tempo todo: o valor deste trabalho não é "ninguém estudou Sumô". O valor é que o Sumô é o caso mais limpo de um problema de rastreamento que a área ainda não resolveu. Rastrear alvos idênticos, em movimento imprevisível, em vídeo real e sem marcador. Resolver isso no Sumô vale pra muito mais coisa.

---

## Slide 1: Título (0:00 – 0:35)

**Frase-síntese:** o Kanshigan mede partidas de Sumô de Robôs a partir do vídeo, e tudo que eu mostro já foi medido.

> Bom dia. Eu sou o Eduardo, e vou apresentar o Kanshigan. É uma pipeline de visão computacional que mede partidas de Sumô de Robôs a partir do vídeo da luta. Essa é uma entrega de meio de percurso. Então tudo que eu mostrar aqui já foi medido de verdade. Nada é projeção.

## Slide 2: O percurso (0:35 – 1:05)

**Frase-síntese:** vou em quatro partes, do problema até onde a pesquisa está hoje.

> Vou dividir em quatro partes. Primeiro o problema. Depois a metodologia e os resultados que a gente já tem. Em seguida o artigo. E no fim, pra onde isso vai e onde a gente quer publicar.

## Slide 3: O domínio (1:05 – 1:50)

**Frase-síntese:** dois robôs de 3 kg, uma arena de 154 cm, e o round acaba em menos de um segundo.

> Primeiro, o domínio. No Sumô de Robôs, dois robôs autônomos de até 3 quilos tentam empurrar um ao outro pra fora de uma arena redonda de 154 centímetros. O que importa pra gente é a velocidade. Os robôs aceleram muito, e a maioria dos rounds acaba em menos de um segundo. Vou deixar um round rodar pra vocês verem.

**[VÍDEO]** Deixe o round correr uns 5 a 10 segundos. Quando acabar, diga: "isso aí foi uma luta inteira".

## Slide 4: O problema (1:50 – 2:30)

**Frase-síntese:** acontece rápido demais pro olho, ninguém consegue medir, e o domínio concentra um problema difícil de rastreamento.

> Acontece tudo rápido demais pro olho. Quem ganhou, quem rampou quem, as trajetórias, tudo em poucos quadros. E hoje não existe ferramenta pra medir isso. A análise é no olho e no replay manual. Mas o ponto interessante não é só que falta uma ferramenta de Sumô. É que esse domínio junta, num caso só, um problema que a área de rastreamento ainda não resolveu. É disso que eu quero falar.

## Slide 5: As seis condições (2:30 – 3:20)

**Frase-síntese:** seis condições definem o domínio, e duas delas formam um problema de rastreamento que está em aberto.

> O que torna isso difícil são seis condições. As duas em verde são o coração do trabalho. A primeira é o movimento não linear. Os robôs batem e giram, então não dá pra prever onde eles vão estar no próximo quadro. A segunda é a aparência uniforme. Os dois robôs são quase iguais, duas caixas pretas. Você não distingue um do outro pelo visual, só pelo jeito que cada um se move. As outras quatro são o cenário. Eventos de menos de um segundo, vídeo de qualquer câmera, análise depois da luta, e nenhum marcador no robô. Esse par, alvos iguais em movimento imprevisível, é um problema em aberto no rastreamento. E ele não é exclusivo do Sumô.

Fale devagar aqui. Esse vocabulário, C1 e C2, volta no resto da apresentação.

## Slide 6: Trabalhos relacionados (3:20 – 4:10)

**Frase-síntese:** cada trabalho da literatura resolve uma parte, ninguém resolve o conjunto, e o Sumô é o caso mais limpo pra atacar a parte em aberto.

> Comparando com a literatura, cada trabalho vizinho resolve um pedaço, mas nenhum resolve tudo junto. O caso mais claro é o DanceTrack. Ele isolou exatamente esse par, alvos parecidos com movimento brusco, e mostrou os melhores rastreadores despencando, de 63 pra 48. Só que ele fez isso em vídeo de estúdio, controlado. Ninguém testou esse problema em vídeo real, de câmera na mão, sem marcador, ainda por cima cobrando que rode rápido num computador comum. É esse o espaço em aberto. E o Sumô é o caso perfeito pra atacar ele, porque junta todas essas condições no estado mais puro. Resolver no Sumô responde uma pergunta que vale pra muito mais, drone racing, outras lutas de robô, qualquer disputa rápida entre alvos iguais.

## Slide 7: Metodologia (4:10 – 4:55)

**Frase-síntese:** pesquisa experimental, comparação controlada contra um gold, numa pipeline de oito estágios.

> Sobre o método. É uma pesquisa experimental. A gente compara de forma controlada contra um conjunto de referência, que a gente chama de gold. A pipeline tem oito estágios. Vou destacar três decisões. A arena a gente acha com visão clássica, sem treinar modelo, porque o formato dela é fixo por regulamento. O diâmetro de 154 centímetros é conhecido, então ele dá a escala em centímetros de graça. E todas as medidas vivem no referencial da arena, o que cancela o tremor da câmera na mão. O SAM 3 entra aqui, mas como anotador.

## Slide 8: Dataset e SAM 3 (4:55 – 5:35)

**Frase-síntese:** sem dataset público, a gente construiu um, e o SAM 3 deixou de ser a pipeline pra virar o anotador, validado contra humano.

> Como não existe dataset público, a gente construiu um. E aqui tem a virada mais importante do projeto. A gente testou o SAM 3 pra rodar a pipeline inteira. Ele segmenta muito bem, mas roda a 2 quadros por segundo e come 7 gigas de memória. Inviável. Então a gente virou a chave. Em vez de rodar ao vivo, ele anota o treino, fora da hora. E um modelo pequeno faz a parte rápida. O dataset tem duas fontes bem diferentes de propósito, câmera na mão do Brasil e câmera fixa de cima do Japão. E a gente conferiu o anotador contra revisão humana. Deu 96 de F1.

Se perguntarem por que SAM 3, a resposta é prática, não científica. Anotar à mão levaria horas por round, e o anotador foi validado.

## Slide 9: Resultados parciais (5:35 – 6:20)

**Frase-síntese:** mAP acima de 0.96, IDF1 de 0.94, mais de 100 quadros por segundo, tudo medido em vídeo real.

> Os números que a gente já tem, todos medidos. Na detecção, mais de 0.96 de mAP nas duas fontes. E o modelo pequeno, com um quinto do tamanho, empata com o grande. No rastreamento, 0.94 de IDF1. A gente testou se usar aparência ajuda, e não ajuda. Ainda por cima deixa tudo de 35 a 40 vezes mais lento. Isso confirma a condição dois na prática. Como os robôs são iguais, a aparência não tem o que explorar, sobra o movimento. E de viabilidade, mais de 100 quadros por segundo, com 100 mega de memória, num notebook comum.

## Slide 10: A pipeline em vários contextos (6:20 – 7:10)

**Frase-síntese:** a mesma pipeline, sem retreinar, funciona da câmera na mão à câmera fixa e até numa categoria que não estava no treino.

> Esse é o resultado rodando em vídeo real. São quatro contextos, a mesma pipeline, sem retreinar nada. Um round do Brasil com câmera na mão. A fonte japonesa, de cima. Outra arena brasileira. E, de bônus, Sumô de rádio controle, que é uma categoria que nem estava no treino. Funciona de primeira.

**[VÍDEO]** Deixe os quatro rodando enquanto fala.

## Slide 11: O artigo (7:10 – 7:50)

**Frase-síntese:** o artigo já existe inteiro, com introdução, relacionados e metodologia, mais versões em paralelo e o diário.

> Sobre o artigo. Ele já existe, inteiro. São 16 páginas no formato da SBC, com introdução, trabalhos relacionados, metodologia e resultados. As primeiras páginas estão aí na direita. E não é só ele. A gente mantém em paralelo uma versão em inglês, o projeto de pesquisa em ABNT, e um diário de bordo com 19 entradas. Tudo público e dá pra reproduzir.

## Slide 12: Publicação (7:50 – 8:25)

**Frase-síntese:** o alvo é o WVC, com um preprint no arXiv como garantia e a versão em inglês de reserva.

> A gente pretende submeter no WVC, o Workshop de Visão Computacional da SBC. É o lugar certo, porque o trabalho é de visão computacional aplicada, que é o tema deles. Junto com isso, a gente põe um preprint no arXiv, que já dá um link pra citar agora. E tem a versão em inglês guardada, caso apareça um evento internacional.

## Slide 13: Limitações e próximos passos (8:25 – 8:55)

**Frase-síntese:** o gold ainda é pequeno, a gente assume isso no texto, e cada limitação tem um próximo passo.

> As limitações estão todas assumidas no texto. O gold ainda é pequeno, então os números são uma ordem de grandeza, não a palavra final. A resposta é anotar mais rounds. A medida em centímetros é aproximada por causa do ângulo da câmera, e a solução é corrigir a perspectiva. E dos eventos, hoje só o início do round a gente pega com confiança.

## Slide 14: O alvo: o mundial (8:55 – 9:25)

**Frase-síntese:** o teste mais duro é o nível mundial, e a gente sabe exatamente onde a pipeline ainda quebra.

> E pra onde isso vai. O teste mais difícil é o nível mundial. Esse vídeo é a final do mundial japonês. É transmissão de TV, bem diferente do que a gente treinou. Tem corte de câmera, colisão violenta, borrão de movimento. E dá pra ver aqui: no momento mais rápido, o detector ainda perde os dois robôs. Eu mostro isso de propósito. A gente sabe exatamente onde está o limite, e sabe o que falta pra passar dele, que é treinar com vídeo desse tipo. Esse é o alvo da versão final.

**[VÍDEO]** Deixe o round do mundial rodar. É o ponto alto da fala.

## Slide 15: Conclusão (9:25 – 10:00)

**Frase-síntese:** três contribuições medidas e públicas, e o recado de que o Sumô é a porta de entrada pra um problema bem maior.

> Pra fechar, três contribuições. A primeira é caracterizar esse problema, as seis condições, e separar o par que está em aberto. A segunda é a pipeline, a primeira aberta pra modalidade, com mais de 0.96 de mAP e mais de 100 quadros por segundo num notebook. A terceira é o dataset, anotado e validado. E o recado principal é esse. O Sumô não é só um caso curioso. Ele é a forma mais limpa de atacar um problema de rastreamento que ainda está aberto, e que vale pra muito mais coisa. Está tudo público no repositório, o link está aí na tela. Obrigado.

---

## Perguntas prováveis da banca (preparação, não apresentar)

| Pergunta | Resposta curta |
|---|---|
| Por que isso é científico, e não só uma ferramenta de Sumô? | O par alvos iguais mais movimento não linear é um problema aberto de rastreamento. O DanceTrack mostrou a queda, mas só em vídeo controlado. A gente ataca ele em vídeo real, e o resultado vale pra outras disputas rápidas entre alvos iguais. |
| Onde pretendem submeter? | WVC, da SBC. Preprint no arXiv pra ter link citável. Versão em inglês de reserva. |
| Que tipo de pesquisa é? | Experimental e quantitativa, com comparação controlada contra um gold revisado. |
| O artigo já existe? | Sim, completo, 16 páginas no formato SBC, com todas as seções. |
| Por que SAM 3 e não anotar à mão? | Tempo. Anotar à mão leva horas por round. E o anotador foi validado, deu 96 de F1. |
| Um round de gold basta? | Não, e o paper diz isso. A detecção está respondida, o rastreamento é preliminar. Mais gold está no plano. |
| A velocidade em centímetros é confiável? | Ainda não. O ângulo da câmera distorce. A correção de perspectiva é o próximo passo. |
| Dá pra reproduzir? | Dá. Semente fixa, versões travadas, dados versionados, e uma seção no README. |

## Checklist antes do dia

- [ ] Testar no projetor. O `index-banca.html` abre offline, os vídeos estão na pasta `videos/`. As fontes do Google precisam de internet, mas o fallback do sistema serve.
- [ ] Conferir se os vídeos tocam sozinhos. O slide 10 tem quatro ao mesmo tempo.
- [ ] Cronometrar uma passada inteira. Se passar de 10 minutos, encurtar o slide 9 e o 13.
- [ ] Levar o PDF do artigo aberto pra consultar as tabelas na hora das perguntas.
