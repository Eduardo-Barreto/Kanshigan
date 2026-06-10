# Integração do feedback da banca: VRAM, versões e busca de ineditismo

## Contexto

A `main` avançou com a reescrita dos capítulos 1 e 2 (entrada 16) e o feedback da
banca virou oito issues (#8 a #15). Esta sessão integrou o nosso branch
`autonomous-run` (faxina + comparação de quatro rastreadores) à `main` reescrita e
atacou as três issues que não dependem de anotar gold novo: #9 (VRAM), #13 (versões) e
#12 (busca de ineditismo). As issues que exigem gold (#8, #10, #11, #14, #15) ficam
pendentes de aprovação.

## Merge e a contradição do rastreador de aparência

O merge teve dois conflitos mecânicos triviais (refs.bib e methodology.typ, este
porque a `main` trocou a lista enumerada por prosa). O conflito que importava era
semântico e o git não detecta: a `main` propunha, em trabalhos futuros, "o passo
seguinte é um rastreador com aparência", enquanto o nosso branch **já tinha testado**
DeepOCSORT e BoT-SORT e mostrado que a aparência não ajuda e custa 35 a 40 vezes. Os
dois textos empilhados se contradiziam. Reconciliei: o experimento fecha metade do que
era "futuro", e o que sobra como futuro é só "mais rounds gold" (issue #11). Também
restaurei a chave `aharon2022botsort` na bib, que a `main` tinha removido na limpeza
mas que a nossa tabela de quatro rastreadores ainda cita, e corrigi o abstract e a
conclusão, que ainda diziam "dois rastreadores".

## #9: VRAM real do processo

O `infer.py` só reportava `max_memory_allocated`. A banca apontou que isso é a
alocação de tensores, não o pico do processo. Passei a reportar três números: tamanho
dos pesos, pico alocado e pico reservado (`max_memory_reserved`, a estimativa mais
fiel do que o processo ocupa).

O que falhou no caminho: rodei primeiro em `data/processed/sam_input/gold_zb01.mp4`,
que é a entrada **decimada** do SAM (10 fps), não o round nativo. O FPS saiu
incoerente (50 fps para 385 quadros não bate). O clip nativo certo é
`data/processed/clips/br/gold_zb01.mp4` (385 quadros, 60 fps). Refeito ali:

| Detector | Pesos | Pico alocado | Pico reservado |
| --- | --- | --- | --- |
| YOLOv8s | 44,5 MB | 82,2 MB | 100,7 MB |
| YOLO26n | 10,0 MB | 66,5 MB | 90,2 MB |

Descoberta: os "82 MB" do paper são o **pico alocado do YOLOv8s**, reproduzido exato. O
pico de processo (reservado) é cerca de 101 MB. Tudo minúsculo frente aos 8 GB da GPU.

Sobre o FPS: o "133" não é robusto. Em partida fria (carregamento do modelo e init do
contexto CUDA dentro do timer) mediu 108 a 118; o "134" original era execução
aquecida. Troquei o número fixo por "acima de 100 FPS (até ~130 aquecida)" no resumo,
no texto e na tabela. É a afirmação defensável; a viabilidade em tempo real continua
sustentada.

## #13: versões, sem re-pinar o SAM 3

Decisão: não adicionar `rev` no `pyproject.toml` do SAM 3. Ele já está travado no
`uv.lock` (commit `f6e51f5`), roda só como anotador offline (E1) e sua saída está
congelada no DVC, então os resultados medidos não dependem da versão dele. Documentei
as versões que importam (estão no caminho do resultado) numa seção de reprodutibilidade
do README: torch 2.12.0+cu126, Ultralytics 8.4.55, boxmot 19.0.0, pesos-base e
checkpoints treinados, ReID `osnet_x0_25_msmt17`.

Furo encontrado e corrigido: a metodologia dizia "Ultralytics versão 8.3"; o lock
resolve **8.4.55**. Alinhado ao lock, que é o que foi usado.

## #12: busca de ineditismo

Busca web (indexada) com quatro strings. Resultado: nenhum trabalho cobre análise de
partida de Sumô de Robôs em terceira pessoa. O que existe é o oposto, percepção
embarcada (o robô vendo o oponente por IR/ultrassom/visão onboard), além de datasets de
detecção de hobby e análise de Sumô humano. Registrei strings, achados e categorização
em `docs/busca-literatura-ineditismo.md`, com a pendência honesta: as contagens exatas
em IEEE Xplore, ACM DL e Scopus autenticados ainda precisam ser rodadas à mão antes da
arguição. É material de defesa, não muda o paper.

## Páginas

O artigo SBC ficou em 17 páginas (15 de corpo, 2 de referências). Removi a figura
`fig-tracker-comparison`, que duplicava exatamente a tabela `tab-tracking`, e apertei a
prosa repetida entre resultados e discussão. As 2 páginas de referências vêm das ~25
citações verificadas que o orientador pediu; cortar mais exigiria remover citação ou
conteúdo dos capítulos 1 e 2, o que contraria o feedback.

## Pendências (dependem de gold, barram no portão de aprovação)

#8 (validar anotações de treino JP), #10 (conjunto de calibração de limiares), #11
(mais rounds gold para os trackers), #14 (predictor de vídeo vs gold JP), #15
(validação metrológica da escala/homografia).
