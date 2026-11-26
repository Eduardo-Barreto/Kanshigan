# Coleta de vídeos públicos: de 12 vídeos de um canal para centenas de torneios

## Contexto

A banca apontou o mesmo buraco por dois lados. O avaliador 1 pediu teste "com mais material, de outros torneios e com robôs que o sistema não viu no treino, incluindo o caso difícil prometido na pergunta, que são robôs parecidos se chocando". O avaliador 2 pediu "a ampliação da amostra de testes".

Conferindo a origem do dado BR, o problema fica concreto: os seis vídeos BR saíram do canal da ThundeRatz, todos da IRONCup 2025, e todos têm o Galena num dos lados. O JP vem de dois vídeos. Qualquer rumo que o trabalho tome agora (pipeline melhor, reprodução virtual da partida) precisa do mesmo insumo: muito mais partidas anotadas, de muitos torneios e muitos robôs.

Esta entrada registra a coleta: de onde veio o vídeo, como virou candidato no studio e o que quebrou no caminho.

---

## de onde veio o vídeo

Três fontes, buscadas com o `yt-dlp` (versão 2026.11.23; a do sistema, 2026.06.09, estava com mais de 90 dias):

| Fonte | Como | Resultado |
|---|---|---|
| Busca no YouTube | 42 consultas em 8 línguas, mais 36 focadas no BR (nomes de robôs, torneios, equipes) | 1.191 + 899 vídeos únicos, filtrados por título |
| Acervo de canais | listagem completa de 12 canais (All Japan, campeonato colegial japonês, RoboCore, Robotex, equipes BR) e de 5 canais BR achados na segunda busca | transmissões de 3 a 8 h e séries de partidas |
| Fórum da ThundeRatz | todo post com link do YouTube, só leitura | 868 posts, 1.034 vídeos, 39 de sumô 3 kg aproveitáveis |

O filtro por título tirou montagem, tutorial, mini sumô, LEGO, combate e notícia de TV. Ficaram 250 vídeos curtos (60 BR, 77 JP, 113 de outros países), 63 BR da segunda busca, 39 do fórum e 42 transmissões longas, depois reduzidas a 33.

O fórum rendeu o que a busca não acha: a campanha do Rubi RC na RCX 2026, round a round, e as partidas RC do Galena na IRONCup 2025.

---

## o que a pipeline faz com cada vídeo

O orquestrador (`experiments/harvest/harvest.py`) chama os mesmos scripts que o studio chama, em etapas que podem ser retomadas:

1. `download`: o vídeo inteiro, até 1080p.
2. `scout`, só nas transmissões longas: baixa uma cópia em 360p, acha os rounds nela e guarda só esses trechos. Baixar 163 h em 1080p não cabia no disco nem fazia sentido.
3. `segment`: `segment_rounds.py` propõe as janelas de round.
4. `cut`: `prep_clips.py` corta cada round e recorta no dohyo.
5. `annotate`: SAM 3 em lote, com o modelo carregado uma vez.
6. `register`: sidecar vazio para o studio e uma tabela de procedência (vídeo, instante, canal, título) por clip.

Vídeos de fora do BR e do JP ganharam a origem `intl` no studio.

---

## o que quebrou, e por quê

**Segmentação lenta.** O piloto (11 vídeos, 37 min de vídeo) gastou 7.468 s de CPU só na segmentação. O `segment_rounds.py` fazia um seek por amostra, e cada seek decodifica desde o keyframe anterior. Trocamos por leitura sequencial com `grab()`, que pula quadro sem decodificar. As janelas saíram idênticas nos 11 vídeos.

**Corte que não cabia na memória.** O `prep_clips.py` carregava o round inteiro na RAM: 30 s de 1080p60 são uns 11 GB. Virou leitura em fluxo. A primeira versão errou o recorte, porque contava os quadros pelo metadado do contêiner, que depois do corte com `-c copy` não bate com o que decodifica. Contando pela decodificação, a saída ficou idêntica byte a byte à antiga.

**OOM em cascata.** No piloto, 52 de 61 clips falharam. O primeiro estourou os 8 GB, e a sessão do SAM que estourou nunca era fechada: ela segurava a VRAM e derrubava todos os clips seguintes. A sessão agora fecha num `finally`, e um bloco que estoura é dividido ao meio e refeito.

Mesmo assim continuou estourando, agora em blocos de 15 quadros. A causa era o limiar 0,15 em câmera de mão: o SAM passava a rastrear mãos, pés e textura do chão, e a memória cresce com o número de objetos, não com o tamanho do bloco.

**Que limiar usar.** O 0,15 foi validado no diário 14 só para o JP, de câmera alta com robôs pequenos. Comparamos as duas opções nos mesmos clips do piloto:

| | Quadros rotulados | Quadros com os dois robôs | Clips perdidos por OOM |
|---|---|---|---|
| 0,15 em tudo | 893 | 503 | 13 |
| padrão no BR e intl | 1.657 | 1.132 | 0 |

O padrão ganhou com folga, mas não em tudo: no Robotex 2013, filmado de cima com robôs pretos e pequenos, ele não achou robô nenhum, e o 0,15 achou. A regra final é adaptativa: roda no padrão e, se menos de 30% dos quadros rotulados tiverem os dois robôs, refaz em 0,15 e fica com a rodada que tiver mais pares. Num teste com três clips, o fallback acionou nos dois esperados (Paralela e Robotex) e não acionou na Tailândia. O JP continua em 0,15 direto, como o studio já fazia.

**Contar caixas engana.** "Quadro com duas caixas" mede pouco. Nas pranchas aparecem caixa numa mão fora do ringue, caixa cobrindo os dois robôs, caixa na tela de transição "2° Round" e recorte fixo mostrando a plateia quando a câmera de mão vira. Por isso tudo entra como candidato, para revisão no studio, e nada vira gold sem aprovação.

**Bloqueio do YouTube.** Depois de umas mil consultas de metadados seguidas, o YouTube passou a pedir "confirme que não é um robô" para o IP, e 25 transmissões falharam no batedor. O `yt-dlp` agora espera entre requisições, e a segunda leva roda devagar.

**Transmissão produzida não serve.** A final do All Japan é transmitida com quatro dohyos na tela, replays e placares: o recorte pega o mosaico inteiro. As regionais do mesmo canal são o oposto, câmera fixa num dohyo só e o nome dos robôs na tela. As finais saíram da fila; as regionais ficaram, limitadas a 40 trechos por transmissão para espalhar a amostra por mais robôs.

---

## números da primeira leva

Os 250 vídeos curtos baixaram sem falha (15 GB) e viraram 1.565 rounds: 156 BR (de 41 vídeos), 653 de outros países (70 vídeos) e 756 JP (57 vídeos). São 14.347 s de round, uns 143 mil quadros para o SAM 3. Nos primeiros clips o lote levou 50 s por clip na RTX 4070; pela média de 9 s por round, a estimativa é de um dia de GPU, com o BR na frente da fila.

O resto (BR extra, fórum e trechos das transmissões) entra numa segunda leva, encadeada para rodar quando a GPU liberar.
