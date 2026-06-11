# Roteiro: pré-banca, versão curta

Apresentação: `index-curta.html`. 9 slides, ~7 a 8 minutos de fala, deixando folga para perguntas. É a versão enxuta e direta da entrega de meio de percurso. A versão narrativa longa, com os vídeos e o aprofundamento técnico, fica em `index.html` / `roteiro.md`.

Regra geral: cada slide tem uma **frase-síntese** (em negrito). Se o tempo apertar, fale só ela e a evidência principal.

---

## Slide 1 — Título (0:00 – 0:30)

**Frase-síntese:** apresento o Kanshigan, uma pipeline de visão computacional que mede o que hoje ninguém mede em Sumô de Robôs.

> Bom dia. Sou o Eduardo, e esta é a entrega de meio de percurso do Kanshigan, uma pipeline de visão computacional de código aberto para extrair métricas de partidas de Sumô de Robôs autônomos a partir do vídeo. Todos os números que vou mostrar são medidos.

## Slide 2 — O percurso (0:30 – 1:10)

**Frase-síntese:** vou passar por quatro paradas, do problema até onde a pesquisa está hoje.

> Em quatro paradas: primeiro o problema e a lacuna na literatura; depois a metodologia e o que já roda sobre footage real; o estado do artigo; e para onde o trabalho vai ser publicado.

## Slide 3 — Problema (1:10 – 2:10)

**Frase-síntese:** tudo acontece rápido demais para o olho, e mesmo assim a análise da modalidade é toda observação humana.

> O que medimos é a posição, a trajetória e os eventos dos dois robôs ao longo da partida, a partir do vídeo, sem instrumentar a arena. Isso importa porque hoje a análise é só observação: equipes ajustam estratégia e hardware por memória e replay manual, sem ferramenta, dataset ou métrica, e o round acaba em menos de um segundo. Os esportes humanos já têm visão computacional, mas sob premissas que aqui não valem: atletas distinguíveis, broadcast multicâmera, eventos em segundos. Para Sumô de Robôs não há pipeline, dataset ou benchmark publicado, e essa busca está registrada no repositório.

## Slide 4 — Trabalhos relacionados (2:10 – 3:10)

**Frase-síntese:** cada vizinho da literatura cobre parte das condições; nenhum reúne o par C1+C2 ao regime não controlado.

> Para situar o trabalho, derivamos seis condições do domínio e mapeamos os vizinhos. O DanceTrack é o mais revelador: isola o par C1 mais C2, movimento não linear e aparência uniforme, e os melhores rastreadores degradam, ByteTrack cai de HOTA 63 para 48, mas em captura controlada. O SSL-Vision da RoboCup resolve robôs rápidos instrumentando, com marcadores que o Sumô não tem. A célula em aberto é o par C1+C2 sob vídeo heterogêneo e sem fiduciais. É exatamente o que a pergunta de pesquisa ataca.

## Slide 5 — Metodologia (3:10 – 4:10)

**Frase-síntese:** é uma pesquisa experimental e comparativa, numa pipeline de oito estágios avaliáveis isoladamente.

> A abordagem é experimental e quantitativa, com avaliação comparativa controlada contra um conjunto gold. A pipeline tem oito estágios, quatro que preparam o quadro e quatro que extraem a informação. Três decisões de projeto: a arena é detectada por visão clássica, sem modelo treinado, porque a geometria é fixada por regulamento; o diâmetro de 154 cm ancora a calibração centímetro-por-pixel; e as métricas vivem no referencial do dohyo, o que cancela o movimento da câmera de mão. Comparamos duas arquiteturas de detector e quatro rastreadores, com o SAM 3 como anotador validado.

## Slide 6 — Resultados parciais (4:10 – 5:20)

**Frase-síntese:** a pipeline roda ponta a ponta sobre footage real, com detecção acima de 0.96, IDF1 0.94 e mais de 100 FPS.

> O que já está rodando, com números medidos. Detecção acima de 0.96 de mAP nas duas fontes, e o detector compacto, com um quinto dos parâmetros, empata. Rastreamento com IDF1 de 0.94; testamos aparência contra movimento e a aparência não traz ganho, e ainda custa de 35 a 40 vezes em throughput. E viabilidade: mais de 100 FPS em cerca de 100 MB de VRAM, num notebook. Três coisas concretas sustentam isso: um dataset multi-fonte de torneios reais, anotado e revisado; os detectores treinados e a pipeline rodando ponta a ponta; e a avaliação concluída contra o gold. À esquerda, a saída real nas duas fontes.

## Slide 7 — O artigo (5:20 – 6:10)

**Frase-síntese:** o artigo está completo, com introdução, relacionados e metodologia, mais versões em paralelo e o diário.

> O artigo está completo: 16 páginas no formato SBC, com introdução que define problema, lacuna e pergunta; trabalhos relacionados com a matriz; metodologia com a pipeline; e os resultados, mais discussão e conclusão. À direita, as primeiras páginas. Além desse texto, mantemos uma versão em inglês no formato IEEE, o projeto de pesquisa em ABNT, e um diário de bordo com 19 entradas que registra a jornada, inclusive o que falhou. Tudo público e reprodutível.

## Slide 8 — Publicação (6:10 – 6:50)

**Frase-síntese:** alvo WVC, arXiv como garantia citável, versão internacional em reserva.

> Sobre a publicação: o alvo primário é o WVC, o Workshop de Visão Computacional da SBC, porque é o escopo direto deste trabalho, detecção e rastreamento aplicados, em português. Em paralelo, um preprint no arXiv dá uma URL citável independente da janela do evento. E mantemos a versão em inglês em reserva, para um venue internacional. A escolha segue o escopo: um paper de visão computacional aplicada, com dataset e avaliação quantitativa.

## Slide 9 — Onde a pesquisa está (6:50 – 7:20)

**Frase-síntese:** problema e lacuna claros, metodologia com resultados medidos, artigo escrito e publicação definida; tudo público.

> Fechando: o problema e a lacuna estão explícitos; a metodologia é experimental, com resultados medidos sobre footage real; o artigo está completo, com versões em paralelo; e a publicação tem alvo definido. Tudo público e reprodutível no repositório. Fico à disposição.

---

## Observações

- Para uma arguição mais técnica em algum ponto (ex.: por que a aparência não ajuda no rastreamento), use os slides equivalentes da versão narrativa (`index.html`).
- Esta versão usa só imagens, para ser mais curta. Se quiser rodar o overlay ao vivo, abra o `index.html` no slide 11.

## Perguntas prováveis (preparação)

| Pergunta | Resposta curta |
|---|---|
| Onde pretendem submeter? | WVC (Workshop de Visão Computacional, SBC); arXiv como preprint citável; versão IEEE em inglês em reserva. |
| Que tipo de pesquisa é? | Experimental e quantitativa, com avaliação comparativa controlada (detector e rastreador fixos, gold revisado). |
| O artigo já existe? | Sim, completo, 16 páginas SBC: introdução, relacionados, metodologia, resultados, discussão e conclusão. |
| Por que SAM 3 e não anotação manual? | Horas por round vs revisão; anotador validado contra gold (F1 0.96, IoU 0.91). |
| Um round gold basta? | Não, e o paper diz isso: detecção respondida, rastreamento preliminar; mais golds no plano. |
| Velocidade em cm é confiável? | Ainda não validada: foreshortening da câmera oblíqua; homografia é o trabalho futuro nomeado. |
| Reprodutibilidade? | Semente fixa, versões pinadas (torch 2.12, Ultralytics 8.4.55, boxmot 19), DVC, seção no README. |
