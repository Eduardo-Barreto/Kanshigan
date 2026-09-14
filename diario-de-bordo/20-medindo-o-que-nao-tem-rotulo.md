# Medindo o que não tem rótulo: auditoria estrutural e ablação do estágio 2

## Contexto

Primeira sprint do pós-banca. As cinco issues que sobraram do feedback (#8, #10, #11,
#14, #15) travam todas no mesmo portão: dependem de gold anotado, e anotar um round
quadro a quadro custa horas que esta janela não tem. A pergunta desta entrada é o que
ainda dá para medir sem rótulo novo — e, como se descobriu no caminho, sem nem sequer
carregar os pesos do detector.

O material é o acervo de Sumô RC (rádio-controlado) da ThundeRatz, 25 arquivos de
câmera de celular. Ele aparece hoje no artigo apenas como figura qualitativa, com a
ressalva explícita de que "nenhuma métrica é reportada sobre ele".

## O corpus RC

Três arquivos são duplicatas exatas (md5 idêntico), então o acervo real é de 22 vídeos.
O `segment_rounds.py` encontrou 23 rounds, sem nenhuma falha e sem nenhum vídeo vazio.

| | |
|---|---|
| Vídeos únicos | 22 (de 25 arquivos) |
| Rounds segmentados | 23 |
| Quadros | 9.026 |
| Duração total | 160 s (round mediano de 6,6 s) |
| Resolução | 14 em 4K, 7 em 1080p, 1 em retrato (1080x1920) |
| Taxa | 19 a 60 fps, 3 a 30 fps |

Diferença estrutural em relação ao acervo BR: lá cada vídeo concatena vários rounds mais
cartões de patrocínio, e a segmentação é trabalho de verdade. Aqui cada arquivo já é uma
gravação por partida, então a segmentação devolve o vídeo inteiro como round único em 21
dos 22 casos. A heterogeneidade que interessa é outra: 4K misturado com 1080p, e um
vídeo em retrato, formato que nenhuma fonte do treino tem.

## Parte 1: auditoria estrutural sem rótulos

Sem gold não há MOTA nem IDF1. Mas o domínio restringe o que uma saída correta precisa
ser, e essas restrições são observáveis no próprio JSON da pipeline: são exatamente dois
robôs no dohyo durante o round, o movimento de um robô de 3 kg é limitado pela física, e
a trajetória de cada robô é contínua. O `experiments/pos-banca/audit.py` conta as
violações.

Proxy só vale reportado com âncora, e a âncora é o `gold_zb01`, onde MOTA 0,881 e IDF1
0,933 foram medidos contra anotação humana. Ler a taxa de violação de um clip ao lado da
do gold diz se ele opera num regime cuja qualidade real é conhecida.

A primeira medição saiu errada e o conserto é a parte que importa. Contando o clip
inteiro, o gold acusava 18,4% de violação, número alto demais para um round com IDF1
0,933. A causa: quadros antes do hajime e depois do ring-out mostram legitimamente menos
de dois robôs no dohyo, e contá-los como falha infla toda taxa. Restringindo à janela do
round (`round_start` a `round_end`, eventos que a pipeline já emite), o gold cai para
6,0%. A janela do round passou a ser parte da definição do indicador.

| Clip | Fonte | Quadros com os 2 robôs | Violação |
|---|---|---|---|
| `demo_jp` | JP, cenital fixa | 1,000 | 0,000 |
| `rc1591` | RC | 0,992 | 0,008 |
| `gold_zb01` | BR, gold | 0,940 | 0,060 |
| `atena` | BR, 848x478 | 0,556 | 0,444 |
| `w_replay` | mundial | 0,459 | 0,541 |
| `worlds_...12401` | mundial | 0,000 | 1,000 |

O indicador se validou sozinho, por um caminho que não estava planejado: ele reproduz a
ordenação qualitativa que já existia escrita nas notas de seleção de vídeo do deck da
pré-banca. Lá está registrado que o `atena` tem resolução baixa demais e que o clip de
mundial é "caso de falha (blur derruba o detector)". O indicador põe os dois exatamente
nessas posições sem que nada tenha sido ajustado para isso.

## Parte 2: ablação do estágio 2, sem pesos

A detecção da arena não usa aprendizado: limiariza luminância, ajusta elipses aos
contornos brilhantes e pontua cada candidata por tamanho, centralidade e preferência por
mais larga que alta. Como não há modelo, ela roda sem checkpoint nenhum, o que a torna a
única parte da pipeline mensurável enquanto os pesos treinados estão em outra máquina.

O módulo afirma, em prosa, que a elipse pontuada é "far more robust on handheld amateur
footage than taking the largest white blob, where background highlights win". O diário
14 registra que a heurística ingênua de fato falhou durante o desenvolvimento, mas o
número nunca foi produzido. Esta é a medição.

O `experiments/pos-banca/dohyo_ablation.py` roda os dois seletores sobre **os mesmos
quadros amostrados**, com pré-processamento idêntico: a única variável é a escolha da
candidata. O baseline ingênuo vive no script de ablação, não na pipeline, porque é a
opção refutada, não um modo suportado.

Sem verdade de campo, a qualidade do ajuste é lida por auto-consistência. O dohyo é fixo
no mundo e tem diâmetro conhecido, então um ajuste correto é estável: o centro e a escala
mal se movem entre quadros, enquanto um ajuste que se agarra a um brilho de fundo pula.
O jitter do centro e o coeficiente de variação do `cm_per_px` ordenam qualidade de
ajuste sem nenhuma anotação.

**Validação interna do indicador.** O `demo_jp` é câmera cenital fixa: a escala estimada
ali tem que ser constante, e o `cv` medido é 0,001 nos dois métodos. O indicador mede o
que se propõe a medir.

### O resultado

Trinta clips, 24 RC e 6 das outras fontes.

| Categoria | n | `cv` da escala (pontuado) | `cv` da escala (ingênuo) |
|---|---|---|---|
| JP (cenital fixa) | 1 | 0,001 | 0,001 |
| BR (mão) | 3 | 0,005 | 0,064 |
| Mundial | 2 | 0,031 | 0,031 |
| RC (celular) | 24 | 0,181 | 0,203 |

Valores medianos. Por clip, o quadro é mais interessante que a mediana sugere:

- **17 dos 30 clips empatam** (diferença de `cv` abaixo de 0,01).
- O pontuado ganha em 9, e dois desses ganhos são grandes: `round_raw_br` (0,002 contra
  0,438) e `IMG_1585` (0,004 contra 0,263).
- O pontuado **perde em 4**, o pior deles por 0,107 (`IMG_1600`).

A alegação do módulo, portanto, não se sustenta como escrita. "Far more robust" descreve
bem o caso do `round_raw_br`, footage brasileiro de câmera de mão, que é exatamente o
caso que a frase cita, e ali a diferença é de duas ordens de grandeza. Mas não é o regime
típico: na maioria dos clips os dois seletores encontram a mesma elipse, e em quatro o
pontuado fica atrás. A formulação que os dados sustentam é mais estreita: a pontuação é
indiferente quando há um único candidato brilhante dominante, e decisiva quando o fundo
oferece competidor, o que a footage de mão produz e a câmera cenital fixa não.

### O achado que vale mais que a ablação

O `cv` da escala no RC tem mediana de 0,18 e chega a 0,38. A estimativa de
centímetros por pixel oscila perto de vinte por cento dentro de uma mesma partida, e toda
métrica de velocidade e aceleração sai dessa escala. Isso é a issue #15 (validação
metrológica da escala e homografia) deixando de ser suspeita e virando medida.

Parte dessa variação é legítima: sob câmera de mão a perspectiva muda de verdade, e com
ela o eixo maior aparente da elipse. O `cv` é um limite superior da instabilidade, não a
sua decomposição. Mas o contraste com o 0,001 da câmera fixa localiza a origem do erro
com precisão suficiente para justificar a retificação por homografia, hoje listada como
trabalho futuro.

### Três ressalvas de validade

A taxa de detecção do seletor ingênuo é vácua: ele nunca rejeita candidata, então marca
1,00 por construção, enquanto o pontuado rejeita elipses implausíveis de propósito e por
isso marca abaixo de 1,00 em sete clips. As duas colunas não são comparáveis, e a
comparação honesta entre os métodos é estabilidade e concordância.

O jitter do centro confunde câmera em movimento com ajuste instável. Ele só é comparável
entre os dois métodos no mesmo clip, nunca entre clips, e é assim que está reportado.

Os clips RC são arquivos de câmera crus, enquanto os de referência BR, JP e mundial vêm
do deck da pré-banca e já passaram por corte e escala. A comparação **entre categorias**
mistura, portanto, proveniências diferentes. A comparação **entre métodos**, que é a
alegação desta parte, não é afetada: os dois seletores veem sempre os mesmos quadros.

## Status

- `experiments/pos-banca/` criado. A pipeline da pré-banca fica congelada como o artefato
  entregue; os estudos que a leem moram no diretório novo.
- `audit.py` com 7 testes: auditoria estrutural sem rótulos, ancorada no gold.
- `dohyo_ablation.py`: ablação do estágio 2 sobre 30 clips, sem pesos e sem rótulos.
- Corpus RC caracterizado: 22 vídeos, 23 rounds, 9.026 quadros.
- Alegação de robustez do estágio 2 medida e reescopada.
- Instabilidade da escala quantificada, dando base empírica à issue #15.

## Pendência

Os 23 rounds RC ainda não passaram pela pipeline completa. O detector treinado não está
na máquina desta sprint (`results/training/` não é versionado e não há CUDA aqui), então
a auditoria estrutural roda hoje sobre um único round RC. A tabela por round, a figura de
distribuição por categoria e o parágrafo que substitui a figura qualitativa do artigo
dependem de executar `infer.py` nos 23 rounds na máquina com a 4070; o resto do caminho
já está pronto e é mecânico a partir dos JSONs.
