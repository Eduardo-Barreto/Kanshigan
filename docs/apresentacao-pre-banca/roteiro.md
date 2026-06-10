# Roteiro da apresentação — pré-banca (10 minutos)

Apresentação: `index.html` (abrir no navegador; navegar com setas, espaço ou Page Down).
13 slides, ~45 s por slide. Os tempos-alvo aparecem discretamente no canto superior
direito de cada slide.

Regra geral: cada slide tem **uma frase-síntese** (em negrito abaixo). Se o tempo
apertar, fale só ela e a evidência principal.

---

## Slide 1 — Título (0:00 – 0:40)

**Frase-síntese:** apresento o Kanshigan, uma pipeline de visão computacional que mede o que hoje ninguém mede em Sumô de Robôs.

> Bom dia. Eu sou o Eduardo e vou apresentar o Kanshigan, uma pipeline de visão
> computacional de código aberto para extrair métricas de desempenho de partidas de
> Sumô de Robôs autônomos a partir do vídeo da partida. O que vou mostrar é uma
> entrega de meio de percurso: todos os números são medidos, e vou ser explícito
> sobre o que já está respondido e o que ainda não está.

Dica: o badge "resultados parciais" já ancora a expectativa da banca — use isso a seu favor.

## Slide 2 — O domínio (0:40 – 1:30)

**Frase-síntese:** dois robôs autônomos de 3 kg, arena de 154 cm, e o round acaba em menos de um segundo.

> Sumô de Robôs: dois robôs autônomos de até 3 kg disputam quem empurra o outro para
> fora de uma arena circular de 154 cm, o dohyo. A modalidade nasceu no Japão em 1989
> e no Brasil tem circuito consolidado — RoboCore Experience, IRONCup. O ponto que
> importa para este trabalho: os robôs partem com aceleração altíssima e, no footage
> que compõe nosso dataset, a maioria dos rounds termina em **menos de um segundo**.

**[VÍDEO]** Rodar o round real aqui (5–10 s). Deixar o vídeo falar: "isso que vocês
acabaram de ver foi um round inteiro".

## Slide 3 — O problema (1:30 – 2:20)

**Frase-síntese:** tudo acontece rápido demais para o olho humano, e mesmo assim toda a análise da modalidade é feita por observação humana.

> Decisão de vitória, primeiro contato, trajetórias — tudo acontece em poucas dezenas
> de quadros. E ainda assim toda a análise de desempenho é feita por observação:
> equipes ajustam estratégia e hardware com base em memória e replay manual. Não
> existe ferramenta, dataset nem métrica padronizada. Nos esportes humanos a visão
> computacional já transformou a análise — futebol, basquete, tênis — mas sobre
> premissas que não valem aqui: atletas distinguíveis pela aparência, broadcast
> multicâmera, eventos em escala de segundos. Fizemos busca sistemática, registrada
> no repositório: não há pipeline, dataset ou benchmark publicado para a modalidade.

Pergunta provável da banca: "como vocês garantem o ineditismo?" → busca documentada
em `docs/busca-literatura-ineditismo.md`; o que existe é percepção embarcada (o robô
vendo o oponente), não análise de partida em terceira pessoa.

## Slide 4 — Pergunta de pesquisa (2:20 – 3:10)

**Frase-síntese:** o Sumô concentra um problema geral de rastreamento; a pergunta é qual combinação de detector e rastreador equilibra acurácia e viabilidade nesse regime.

> A pergunta não é só "como analisar Sumô". O domínio instancia, em estado
> concentrado, um problema mais geral: **quais combinações de detector e rastreador
> melhor equilibram acurácia e viabilidade prática para rastrear alvos quase
> idênticos, em movimento rápido e não linear, sobre vídeo não calibrado e sem
> marcadores?** Caracterizamos o domínio por seis condições. Duas descrevem a
> instância, duas o regime de uso — e duas, destacadas em verde, formam o par
> científico em aberto: C1, movimento não linear, e C2, aparência uniforme.
> Viabilidade prática aqui é literal: rodar no computador que as equipes têm.

## Slide 5 — Posicionamento (3:10 – 3:50)

**Frase-síntese:** cada vizinho da literatura cobre parte das condições; nenhum reúne o par C1+C2 ao regime não controlado.

> Organizamos a literatura em quatro corpos. O DanceTrack é o mais revelador: isola
> exatamente o par C1+C2 — dançarinos uniformizados, movimento brusco — e os melhores
> rastreadores degradam: ByteTrack cai de HOTA 63 para 48. Mas isso em captura
> controlada. O SSL-Vision da RoboCup resolve robôs rápidos **instrumentando**: 
> marcadores coloridos e câmera calibrada, que o regulamento do Sumô não prevê. A
> célula em aberto é o par C1+C2 sob vídeo heterogêneo e sem fiduciais, julgado por
> acurácia E custo — é onde este trabalho se posiciona.

## Slide 6 — Pipeline (3:50 – 4:40)

**Frase-síntese:** oito estágios com saídas inspecionáveis: quatro preparam o quadro, quatro extraem a informação.

> A pipeline segue tracking-by-detection em oito estágios, cada um avaliável
> isoladamente contra o gold. Três decisões de projeto merecem destaque. Primeira: a
> arena é detectada por visão clássica, sem modelo treinado — a geometria do dohyo é
> fixada por regulamento, então um procedimento determinístico resolve e suas falhas
> são diagnosticáveis. Segunda: o diâmetro de 154 cm, conhecido por regulamento,
> ancora a calibração centímetro-por-pixel — sem calibração de câmera nem marcador.
> Terceira: as métricas vivem no referencial do dohyo, o que cancela o movimento da
> câmera de mão. E os eventos saem de regras determinísticas auditáveis, não de um
> classificador que aprenderia os poucos exemplos disponíveis.

## Slide 7 — Dataset e SAM 3 (4:40 – 5:40)

**Frase-síntese:** o SAM 3 era candidato a pipeline, mostrou-se inviável na inferência, e virou o anotador — validado contra gold humano com F1 0.96.

> Não existe dataset público, então construímos um, e aqui está a jornada mais
> importante do projeto. Testamos o SAM 3 como pipeline: segmenta bem, mas roda a 2
> quadros por segundo com 7 GB de VRAM. Em vez de descartá-lo, viramos a peça: ele
> **anota** o conjunto de treino, offline, e um modelo compacto roda a inferência.
> O conjunto é multi-fonte de propósito: câmera de mão brasileira e cenital fixa
> japonesa, os dois extremos de captura. No Japão os limiares padrão do SAM perdiam
> os robôs pretos pequenos — instrumentamos, baixamos o limiar para 0.15, filtramos
> pela geometria da arena, e a fonte foi de 16 para 202 quadros úteis. A divisão é
> por round, nunca por quadro, para não vazar informação. E o anotador é validado:
> contra o gold revisado por humano, F1 de 0.96 com IoU médio de 0.91.

Pergunta provável: "por que SAM 3?" → resposta metodológica, não científica: anotação
eficiente dada a ausência de dataset; não estamos comparando SAM com YOLO.

## Slide 8 — Detecção (5:40 – 6:30)

**Frase-síntese:** os dois detectores passam de mAP 0.96 nas duas fontes; o compacto empata com 1/5 dos parâmetros; e o recorte no dohyo foi a correção decisiva.

> Primeira metade da pergunta: detecção. Comparamos YOLOv8s, de 11 milhões de
> parâmetros, com o YOLO26n compacto, de 2,4 milhões. Ambos passam de mAP 0.96 nas
> duas fontes — câmeras opostas, mesmo detector. Com um round held-out por fonte,
> diferenças nessa ordem não se distinguem de ruído, então o que o conjunto sustenta
> é equivalência prática: o compacto basta e melhora a viabilidade. Dois contrastes
> dão escala: COCO sem fine-tuning fica em 0.026 — o domínio exige treino específico.
> E sem o recorte no dohyo a precisão era 0.71; recortar ampliou os robôs três vezes,
> removeu o fundo e levou a precisão a 0.99.

## Slide 9 — Rastreamento (6:30 – 7:20)

**Frase-síntese:** quatro rastreadores sobre as mesmas detecções: aparência não ganha nada em acurácia e custa 35 a 40 vezes em throughput.

> Segunda metade: rastreamento. Desenho controlado — os quatro rastreadores recebem
> exatamente as mesmas detecções, então qualquer diferença é do rastreador. A
> condição C2 sugeria que aparência teria pouco a oferecer; tratamos como hipótese e
> testamos. Resultado: BoT-SORT empata com ByteTrack, DeepOCSORT fica atrás dos
> motion-only, todos seguram identidade com no máximo uma troca. Mas o passo de ReID
> derruba o throughput de mais de 3000 FPS para menos de 100. O porquê é estrutural:
> dois robôs pretos quase idênticos não dão sinal discriminativo para o ReID; sobra o
> movimento, que os rastreadores baratos já modelam. Sendo honesto: é um round gold
> com identidades; o veredito fino exige mais rounds, e isso está no plano.

## Slide 10 — Viabilidade (7:20 – 7:50)

**Frase-síntese:** a pipeline completa roda acima de 100 FPS em ~100 MB de VRAM num notebook — a divisão anotador pesado / inferência leve é o que os números obrigam.

> Viabilidade medida, não projetada: pipeline completa acima de 100 FPS, pico de
> processo de cerca de 100 MB de VRAM, numa RTX 4070 de notebook. O contraste fecha o
> argumento: o SAM 3 anotador ocupa 7 GB a 2 FPS. Setenta vezes mais memória,
> cinquenta vezes menos velocidade — a arquitetura "foundation model anota, modelo
> compacto infere" não é preferência, é o que os números obrigam.

## Slide 11 — O que a pipeline entrega (7:50 – 8:40)

**Frase-síntese:** da caixa delimitadora à métrica de combate: trajetórias em centímetros, velocidade de pico de 2,9 m/s, e generalização qualitativa fora do treino.

> O que sai no final: detecção e rastreamento nas duas fontes, e as trajetórias
> projetadas no referencial do dohyo, em centímetros — velocidade de pico medida de
> 2,9 metros por segundo no round gold. Dois bônus qualitativos, reportados como
> qualitativos: a pipeline transfere zero-shot para Sumô de rádio-controle, categoria
> e arena que não existem no treino; e no caso extremo, a final do mundial japonês, o
> blur do combate de elite derruba o detector — sabemos exatamente onde está o limite
> e o que falta para fechá-lo: dados de treino daquela distribuição.

**[VÍDEO]** Saída da pipeline com overlay (10–15 s) — detecção + rastreamento + 
trajetória sobre um round real.

## Slide 12 — Limitações e próximos passos (8:40 – 9:20)

**Frase-síntese:** o gold é pequeno e nós dizemos isso no paper; cada limitação tem um próximo passo correspondente até a versão final.

> As limitações estão assumidas no texto, e cada uma tem contraparte no plano. O gold
> é pequeno — um round por fonte — então os números indicam ordem de grandeza, sem
> intervalo de confiança; a resposta é mais rounds gold anotados. A calibração por
> escala isotrópica é aproximação sob perspectiva; a resposta é a retificação por
> homografia, que destrava a validação da cinemática. Dos eventos, só o início de
> round dispara confiavelmente; contato e ring-out precisam de um conjunto de
> calibração separado do gold. E o caso do mundial exige treino com broadcast.

## Slide 13 — Conclusão (9:20 – 10:00)

**Frase-síntese:** três contribuições medidas e públicas; detecção respondida, rastreamento preliminar — e fecha na versão final.

> Três contribuições. Primeira: a caracterização do domínio por seis condições, com o
> recorte científico no par em aberto. Segunda: a primeira pipeline aberta da
> modalidade — mAP acima de 0.96, IDF1 até 0.94, mais de 100 FPS em 100 MB de VRAM.
> Terceira: o dataset anotado, com anotador validado e protocolo sem vazamento.
> Reporto como entrega de meio de percurso: a detecção está respondida nas duas
> fontes; o rastreamento é preliminar e fecha na versão final com mais rounds gold.
> Código, dados e protocolo estão públicos. Obrigado — fico à disposição.

---

## Perguntas prováveis da banca (preparação, não apresentar)

| Pergunta | Resposta curta |
|---|---|
| Por que SAM 3 e não anotação manual? | Horas por round vs revisão; anotador validado contra gold (F1 0.96, IoU 0.91). |
| Por que não comparar SAM 3 com YOLO? | Pergunta errada: viabilidade já conhecida (7 GB / 2 FPS); SAM é meio metodológico. |
| Um round gold basta? | Não, e o paper diz isso: detecção respondida, rastreamento preliminar; issue aberta para mais golds. |
| E se os robôs fossem distinguíveis? | O ReID voltaria a ter sinal; nosso recorte é exatamente o regime C2 (DanceTrack mostra a queda). |
| Velocidade em cm é confiável? | Ainda não validada: foreshortening da câmera oblíqua; homografia é o trabalho futuro nomeado. |
| Por que regras e não classificador de eventos? | Conjunto pequeno: classificador decoraria; regras são auditáveis e os limiares são versionados. |
| Reprodutibilidade? | Semente fixa, versões pinadas (torch 2.12, Ultralytics 8.4.55, boxmot 19), DVC, seção no README. |

## Checklist antes do dia

- [ ] Inserir os dois vídeos nos espaços reservados (slides 2 e 11).
- [ ] Testar no projetor/HDMI: `index.html` abre offline; fontes Google precisam de internet (fallback: sans-serif do sistema, aceitável).
- [ ] Cronometrar uma passada completa; se passar de 10 min, cortar as falas de "Leitura 2" do slide 8 e o bônus RC do slide 11.
- [ ] Levar o PDF do paper impresso ou aberto para consultar tabelas na arguição.
