# Roteiro: pitch de 5 minutos

Apresentação: `index.html` (abrir no navegador, navegar com setas ou espaço). Capa e quatro slides, um por bloco da defesa: problema, método, resultados, limitações. Os resultados ficam mais tempo na tela.

Cada slide tem uma **frase-síntese**. Se o tempo apertar, fale só ela. Os tempos somam 4:25, deixando margem para o alarme dos 5 minutos.

---

## Capa (0:00, 10s)

**Visual:** o round da final do mundial rodando ao lado do título, pra banca já ver o que é uma luta antes do primeiro slide.

> Eu sou o Eduardo, e esse é o Kanshigan: uma pipeline de visão computacional que mede partidas de Sumô de Robôs a partir do vídeo da luta.

## Slide 1: Problema e lacuna (0:10, 55s) [TEM BUILD: apertar seta uma vez a mais]

**Visual:** matriz trabalhos × C1-C6. Ao avançar, C1 e C2 acendem, o resto esmaece, aparece a coluna "regime de captura" e a faixa com a lacuna e a pergunta.

**Frase-síntese:** quem resolve alvos idênticos em movimento não linear só resolve em estúdio, broadcast ou com marcador; em vídeo real, está em aberto.

> Um round de Sumô de Robôs dura menos de um segundo, e hoje tudo é analisado no olho: quem ganhou, quem rampou quem, com que velocidade, decidido de memória no replay de celular.
>
> A gente caracterizou o domínio em seis condições, as colunas, e cruzou com os trabalhos mais próximos. Nenhum cobre mais de três.
>
> [avançar o build] O recorte está nessas duas: alvos visualmente idênticos, em movimento não linear. Alguns trabalhos marcam essas colunas, mas olhem o regime: estúdio, broadcast, marcador colado no robô. Em vídeo real, sem marcador e em hardware comum, ninguém respondeu. Daí a pergunta: que combinações de detector e rastreador equilibram acurácia e custo nesse cenário?

## Slide 2: Método (1:05, 55s)

**Visual:** a pipeline em oito estágios; embaixo, o desenho do experimento em três colunas: o que comparamos, o que fica fixo, contra o quê.

**Frase-síntese:** experimento controlado: troca-se uma peça por vez, com o resto fixo, e mede-se contra um gold que nunca entra no treino.

> A pesquisa é experimental. Essa é a pipeline: a fileira de cima acha a arena e recorta o quadro, a de baixo detecta, rastreia e mede. A arena sai por visão clássica, e os 154 centímetros do regulamento convertem pixel em centímetro.
>
> O experimento troca uma peça por vez. Nos detectores, comparamos um YOLO padrão contra um cinco vezes menor, e o YOLO sem treino no domínio como linha de base. Nos rastreadores, dois que usam só o movimento contra dois que também usam aparência. O que fica fixo: os detectores treinam no mesmo dataset, e os quatro rastreadores recebem exatamente as mesmas detecções, então qualquer diferença é do rastreador. E tudo é medido contra um gold: rounds que nunca entram no treino, anotados pelo SAM 3 e revisados à mão, com acurácia e custo medidos no mesmo notebook.

## Slide 3: Resultados (2:00, 95s)

**Visual:** a resposta à pergunta numa faixa verde; três números com a explicação de cada métrica embaixo; quatro vídeos da pipeline rodando, o último (final do mundial) com borda laranja.

**Frase-síntese:** a resposta é detector compacto + rastreador só de movimento: mesma acurácia, a mais de 100 fps num notebook; a final do mundial ainda falha.

> A resposta à pergunta: detector compacto com rastreador só de movimento. Os números que sustentam isso, todos contra o gold:
>
> Detecção: mAP de 0.96 pra cima. O mAP vai de 0 a 1 e mede se o detector acha os robôs onde eles estão sem inventar robô onde não tem. O modelo cinco vezes menor empata com o grande. E o YOLO sem treino no domínio fica em 0.03: o treino específico é o que faz funcionar.
>
> Rastreamento: IDF1 de 0.94. O IDF1 mede quanto do tempo cada robô mantém o rótulo certo, o A continua A. A aparência não ajudou: mesmo IDF1, 35 a 40 vezes mais lento. Faz sentido: com robôs idênticos, a aparência não tem o que diferenciar.
>
> E a pipeline completa passa de 100 quadros por segundo num notebook, com uns 100 mega de memória de vídeo. O vídeo tem 30 a 60, então é mais rápido que o tempo real.
>
> Embaixo, a pipeline rodando, o mesmo modelo, sem retreino: Brasil com câmera de mão, Japão com câmera fixa, e Sumô rádio controlado, uma categoria que nem estava no treino. [apontar o último] E aqui é onde ainda não funciona: a final do mundial, em transmissão de TV. No momento mais rápido, o borrão faz o detector perder os dois robôs.

## Slide 4: Limitações e próximos passos (3:35, 50s)

**Visual:** três pares limitação → ação; faixa verde com a contribuição e o repositório.

**Frase-síntese:** cada limitação é medida e tem um próximo passo; a entrega é a primeira pipeline aberta da modalidade.

> Três limitações, cada uma com o próximo passo. A conversão pra centímetro oscila perto de 18% dentro de uma mesma luta nos vídeos de celular, contra 0,1% na câmera fixa, então as métricas de velocidade ainda não estão validadas: o próximo passo é retificar a arena por homografia. O gold é pequeno, dois rounds, então os números são ordem de grandeza: a resposta é anotar mais, começando pelo RC. E o blur de transmissão, que vocês acabaram de ver: falta treinar com vídeo dessa distribuição.
>
> O que fica é a primeira pipeline aberta pra modalidade, com dataset validado, rodando num notebook. Tudo público no repositório. Obrigado.

---

## Perguntas prováveis (preparação, não apresentar)

| Pergunta | Resposta curta |
|---|---|
| Por que isso é ciência e não ferramenta de nicho? | O par alvos idênticos + movimento não linear está em aberto fora do estúdio. O mesmo par aparece em drone racing e em outras disputas rápidas entre alvos iguais. |
| Como avaliar os vídeos que não têm gold? | Pela regra do jogo: são sempre dois robôs no dohyo, então dá pra contar em quantos quadros do round a pipeline vê os dois, sem rótulo. Comparado com o round gold (94%): JP 100%, RC 99%, BR baixa resolução 56%, transmissão do mundial 46%, final 0%. Bate com a ordem que a inspeção visual já indicava. |
| Então o RC funciona melhor que o gold? | Não dá pra dizer: é um único round medido; os 23 rounds do acervo RC ainda vão passar pela pipeline completa. O número diz que o RC opera no regime do gold, não que é melhor. |
| O detector da arena é mesmo mais robusto que pegar a maior mancha branca? | Medido em 30 clips: empata em 17, ganha em 9 e perde em 4. Decide quando o fundo tem outro candidato brilhante, como na câmera de mão brasileira (0,002 contra 0,438 de variação da escala); empata quando a arena é o único brilho da cena. |
| Os 18% de variação da escala são erro? | Parte é legítima: na câmera de mão a perspectiva muda de verdade. É um limite superior da instabilidade, e o contraste com 0,1% na câmera fixa aponta a homografia como correção. |
| Por que SAM 3 como anotador e não como pipeline? | Validado contra humano com F1 0.96, mas roda a 2 fps com 7 GB de VRAM. Serve offline, não ao vivo. |
| Dá pra reproduzir? | Semente fixa, versões travadas, dados versionados, instruções no README. |

## Colinha das métricas

| Métrica | O que mede | Como ler |
|---|---|---|
| **mAP@0.5** (mean Average Precision) | Qualidade da detecção. Uma caixa prevista conta como acerto se se sobrepõe a pelo menos 50% da caixa real (IoU ≥ 0.5). A precisão (quantas caixas previstas são robôs de verdade) e o recall (quantos robôs de verdade foram achados) são combinados ao longo dos limiares de confiança; o mAP é a área sob essa curva, com média entre as classes. | 0 a 1. 0.96 = quase todos os robôs achados, quase nenhum falso positivo. |
| **IDF1** (Identity F1) | Qualidade do rastreamento: consistência de identidade. Casa cada trajetória prevista com uma real e conta os quadros em que o rótulo está certo. É a média harmônica entre precisão e recall de identidade. | 0 a 1. Cai quando o A vira B, quando um robô some ou quando nasce uma trajetória falsa. 0.94 = em 94% dos quadros o robô certo tem o rótulo certo. |
| **fps** | Quadros processados por segundo pela pipeline inteira (decodificação, arena, detecção, rastreamento). | Acima da taxa do vídeo (30 a 60 fps) = mais rápido que o tempo real. |
| **VRAM** | Memória da placa de vídeo usada durante a inferência. | ~100 MB cabe em qualquer GPU de notebook. |
| **F1 do anotador** | Concordância do SAM 3 com a revisão humana, quadro a quadro: média harmônica entre precisão e recall das caixas. | 0.96 = o anotador automático erra pouco o suficiente pra gerar treino. |

## Checklist antes do dia

- [ ] Testar no projetor da sala (o deck está em tema claro por causa da iluminação): os vídeos tocam sozinhos (mudos) e as fontes caem pro fallback sem internet.
- [ ] Lembrar do build no slide 1: uma seta a mais pra acender C1+C2.
- [ ] Cronometrar uma passada. Passou de 4:45: no Método, pular a frase da arena; nas Limitações, só a primeira por extenso.
- [ ] Levar o PDF do artigo aberto pras tabelas na hora das perguntas.
