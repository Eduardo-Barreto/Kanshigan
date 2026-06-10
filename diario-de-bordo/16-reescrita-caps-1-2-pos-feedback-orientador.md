# Reescrita dos capítulos 1 e 2 após feedback do orientador

## Contexto

O orientador mandou sete áudios revisando o artigo SBC antes da banca de meio de
percurso. Os pontos: a introdução não tinha nenhuma fonte e funcionava como um
resumo longo do trabalho, em vez de contextualizar e justificar; os trabalhos
relacionados citavam conceitos nominalmente em sequência, sem explicar técnica e
resultado de cada um; a tabela de características C1-C6 aparecia sem contexto,
sem emanar da análise dos trabalhos; a metodologia estava condensada, com
estrutura de enumeração com cara de IA; e 90% do esforço deveria ir para os
capítulos 1 e 2. Esta entrada registra a reescrita.

## Pesquisa de fontes (nada citado sem verificação)

Toda citação nova passou por verificação em fonte primária (arXiv, páginas
oficiais, PDFs de regulamento). O material verificado, com números e BibTeX,
está em `.omc/research/paper-citations.md`. Destaques:

- Regulamento oficial FSI v4.2 (PDF): dohyo 154 cm, limite 3 kg, formato de
  batalha. Guinness para "longest-running robot sumo tournament" (desde 1989).
- Cena brasileira: RoboCore Experience (maior evento de combate da América
  Latina), IRONCup/Inatel. Sumô como estratégia STEM: Carbone et al. 2022.
- Surveys de CV em esportes: Thomas et al. (CVIU 2017) e Naik et al. (2022).
- Números exatos de benchmark: SORT (MOTA 34.0 MOT15, 260 Hz), ByteTrack (MOTA
  80.3 / IDF1 77.3 / HOTA 63.1 MOT17; 47.7 DanceTrack), OC-SORT (HOTA 63.2
  MOT17; 55.1 DanceTrack), SportsMOT (240 seqs, MixSort 66.2 no basquete),
  SoccerNet-Tracking (200 seqs + 45 min), Grounding DINO (52.5 AP zero-shot).
- Citação do SoccerNet trocada do tech report MOT4MOT 2023 para o paper canônico
  do dataset (Cioppa et al., CVPRW 2022). Jiu-jitsu agora com autores corretos
  (Hudovernik e Skočaj); a acurácia exata não foi confirmada em fonte aberta,
  então o texto descreve o resultado qualitativamente.

## Introdução nova (funil)

Contexto da modalidade com 7 citações (Fujisoft/Guinness/regras/RoboCore/
IRONCup/Carbone) → problema da análise sub-segundo (ancorado no nosso corpus) →
lacuna frente à análise esportiva por CV (Thomas/Naik) → justificativa →
pergunta de pesquisa → contribuições → organização. Os parágrafos que descreviam
a pipeline e o SAM 3 saíram da introdução; o "o que fizemos" virou um parágrafo.

## Trabalhos relacionados novos (técnica + resultado + correlação)

Reorganizado em quatro corpos de literatura (MOT genérico, MOT em esportes,
combate/robôs de competição, foundation models como anotadores) + subseção da
interseção. Cada trabalho-chave agora tem: o que a técnica faz, número que
obteve, e o que resolve ou não no nosso problema. Cada característica C1-C6 é
derivada no corpo do texto a partir do trabalho que a motiva, e foi RENUMERADA
para seguir a ordem de derivação (C1 movimento não linear, C2 aparência
uniforme, C3 eventos sub-segundo, C4 vídeo heterogêneo, C5 análise post-match,
C6 ausência de fiduciais). A renumeração cascateou para metodologia, resultados
e discussão. A tabela virou consolidação, com nota honesta sobre C4 (restrição
de regime de dados) e C6 (critério de exclusão, não eixo de dificuldade).

## Metodologia arejada + fluxograma

A enumeração de oito itens virou prosa em subseções (arena/calibração,
recorte/detecção, rastreamento/identidade, métricas/eventos, dataset, treino),
cada uma com o porquê da decisão. Fluxograma da pipeline em Typst puro (boxes e
setas, sem pacote externo), com o ramo offline do SAM 3 tracejado.

## Validação em quatro passes

1. Reviewer (citações/anti-slop): pegou a frase "no máximo quatro das seis" que
   contradizia a tabela (máximo real: 3 "S"), autor faltando no BibTeX do
   Grounding DINO e DOIs ausentes. Corrigidos.
2. Critic (compliance com o feedback): aprovou itens 1-6, pediu âncora para as
   duas afirmações fortes da introdução (sub-segundo; "não há ferramentas") e a
   renumeração C1-C6 pela ordem de derivação. Feitos.
3. Deslop: removeu meta-frase na abertura do cap. 2, o tique "não exceção" e o
   autoelogio "sustentando o diferencial científico".
4. Banca simulada (avaliador independente, nota 7,5, COM RESSALVAS): críticas
   incorporadas no texto: abstract não "vende" mais ordenação de detectores
   indistinguível de ruído com n=2 (agora "equivalência prática"); pergunta de
   pesquisa qualificada (componente de tracker é preliminar por desenho); 82 MB
   qualificado como alocação do detector; vazamento de calibração de limiares no
   gold explicitado; conclusão cita os dois pares comparados; labels
   sec-metodologia/sec-resultados criados.

## Pendências para a arguição (não resolvíveis por texto)

- Validar amostra das anotações de treino JP (predictor de vídeo, limiar 0.15)
  contra revisão humana: hoje só o gold tem validação humana e o recall por
  quadro do SAM 3 no JP é 0.22.
- Medir pico real de VRAM do processo (`torch.cuda.max_memory_allocated`), não
  só a alocação do detector.
- Protocolo de busca para a alegação de ineditismo (bases e termos), se a banca
  cobrar "primeira pipeline".
- Conjunto de calibração de limiares separado do gold antes de reintroduzir a
  cinemática quantitativa (junto com o bug do round_start por borda do SG, da
  entrada 15).

## Status

- Paper recompila sem erros, 16 páginas (eram 10; o crescimento está nos caps.
  1-2 e na metodologia, como pedido). Conferir limite de páginas da entrega.
- Citações: 100% verificadas; chaves e números auditados contra a ficha de
  pesquisa.

## Adendo: rodada 2 da banca simulada e diagrama definitivo

O fluxograma SVG colorido da primeira versão tinha estética de infográfico
(faixas coloridas, badges numerados); foi refeito em estilo de figura de paper:
monocromático, caixas brancas de traço fino, setas pretas, rótulos de fileira em
itálico e ramo de anotação tracejado.

Uma segunda banca simulada (avaliador independente, sem acesso à primeira) deu
nota 8,0, COM RESSALVAS. Correções de enquadramento aplicadas no texto:

- Pergunta de pesquisa reformulada de "qual melhor equilibra" para suficiência
  ("que combinação é suficiente"), fechando o flanco de n=1-2 não suportar
  eleição de vencedor.
- "F1 0.96" do SAM 3 qualificado como gold BR no abstract/resumo (o predictor de
  vídeo no JP não tem validação quantificada; ver issue #14).
- Tabela de interseção reposicionada como mapa de cobertura, não placar: "S" do
  Kanshigan significa "opera sob a restrição"; C1-C3 são dificuldades técnicas,
  C4-C6 regime de dados/uso.
- Velocidade de pico (2,9 m/s) enquadrada como verificação de plausibilidade,
  sem ground-truth físico.
- Fronteira explícita entre fontes de treino/gold (BR, JP) e footage qualitativo
  OOD (RC, mundial).
- YOLO26n agora com a referência oficial (Jocher et al., arXiv 2606.03748);
  Ultralytics 8.3 e implementação oficial do SAM 3 citadas na metodologia.

O que exige rodar experimento virou issue: #8 (validar anotações de treino JP),
#9 (pico real de VRAM), #10 (conjunto de calibração de limiares), #11 (mais
rounds gold p/ trackers + poder estatístico), #12 (protocolo de busca de
ineditismo), #13 (fixar versões/checkpoints), #14 (predictor de vídeo vs gold
JP), #15 (validação metrológica da escala/homografia).
