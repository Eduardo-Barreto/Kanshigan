# Execução da pipeline da pré-banca

## Contexto

Esta entrada documenta a execução do plano definido nos diários [12](12-design-da-pipeline-pre-banca.md)
e [13](13-protocolo-experimental-e-cronograma.md): aquisição de dados,
implementação da pipeline completa, anotação semiautomática e os ajustes que a
realidade dos dados impôs sobre o design.

## Aquisição de dados

Sem dataset público do domínio, os clips vêm de torneios reais via `yt-dlp`. O
subconjunto amador brasileiro usa seis partidas autônomas (categoria Auto, 3 kg) do
canal da ThunderRatz, cada vídeo já recortado em uma partida (IRONCup 2025 e
outras). Cada vídeo tem cartões de abertura, patrocínio e b-roll; os trechos de
partida foram identificados por inspeção de contact-sheets e recortados em
`configs/clips.yaml` (fora do git, dado versionado por DVC).

Para o broadcast japonês, a live do All Japan Robot Sumo 2025 (4h16) tem metadados
de capítulos inconsistentes (a "premiação" alegava durar quatro horas), então não
serve para corte cego. Clips japoneses individuais avaliados eram recapitulações
mistas, sem rounds contínuos limpos. O broadcast japonês fica como trabalho futuro
para estressar a heterogeneidade de qualidade (restrição C3), conforme já previsto
na discussão do artigo.

## Ajustes de design forçados pelos dados reais

Três premissas do design não sobreviveram ao contato com a footage real:

**Câmera não é cenital fixa.** A footage amadora é de mão, com movimento e ângulo
oblíquo. Consequências: a detecção do dohyo passa a ser por quadro (com reuso da
última detecção válida em falhas), e as métricas viram referencial centrado no
dohyo, o que cancela o movimento de câmera (o dohyo é fixo no mundo). A calibração
centímetro-por-pixel por escala isotrópica vira aproximação, com o erro de
perspectiva documentado como limitação; retificação por homografia fica para o
trabalho final.

**Detecção do dohyo por "maior região branca" falha.** O dohyo real é uma
plataforma preta com anel branco (tawara), sobre tapete colorido, com fundo ruidoso
(texto de overlay, cercas, pessoas). A heurística do PoC pegava objetos brancos do
fundo. A solução foi pontuar cada elipse candidata por tamanho, centralidade e
razão de aspecto (um círculo visto de lado é mais largo que alto), escolhendo a de
maior escore. Validado visualmente em múltiplos frames reais.

**SAM 3 estoura a VRAM de 8 GB em clips longos.** O preditor de vídeo do SAM 3
mantém feature maps de todos os quadros da sessão; clips de centenas de quadros
esgotam a GPU. A correção foi processar em janelas de 60 quadros, encerrando a
sessão entre janelas (`close_session`) para liberar memória, e ler os quadros
nativos sob demanda em vez de carregar o clip inteiro em RAM (um clip de 84 s a
1080p60 não cabe nos 30 GB de RAM). Rounds de Sumô são curtos, então uma janela
cobre um round inteiro.

## Pipeline implementada

A pipeline de inferência está implementada em `experiments/pre-banca/`, modular e
testada na lógica pura (sem GPU):

- `dohyo.py`: detecção da arena e calibração.
- `tracking.py`: OC-SORT (boxmot) com convenção de identidade A/B.
- `metrics.py`: cinemática por Savitzky-Golay no referencial do dohyo.
- `events.py`: eventos determinísticos (início, contato, ring-out, vencedor).
- `infer.py`: orquestração ponta a ponta, saída JSON + vídeo + MOT.
- `train.py`, `evaluate.py`: treino do YOLOv8s e avaliação contra o gold.

A suíte de testes (`pytest`) cobre geometria do dohyo, cinemática, regras de evento
e atribuição de identidade. A anotação semiautomática (SAM 3 → bbox YOLO) gera o
conjunto de treino e validação, com divisão por clip para evitar vazamento entre
quadros vizinhos.

## Iteração de qualidade dos dados

A primeira passada produziu números enganosos porque os dados estavam ruins, não a
pipeline. Duas causas, ambas apontadas na revisão:

**Clips multi-round + ending.** Cada vídeo BR concatena vários rounds mais um outro
de patrocínio. Em clip multi-round, o SAM 3 perdia um robô em ~35% dos quadros
(durante resets e na ending). Em round único limpo, o SAM 3 pega os dois robôs em
93% dos quadros. A correção foi segmentar em rounds únicos (motion + presença do
dohyo, em `segment_rounds.py`) e descartar as endings.

**Rótulos incompletos no treino.** Quadros com só um robô são rótulos incompletos
que ensinam o detector a ignorar um robô. Passamos a treinar apenas com quadros de
rótulo completo (os dois robôs), descartando os incompletos.

**Gold revisado por humano.** O gold (um round único) foi pré-anotado pelo SAM 3 e
revisado/aprovado manualmente quadro a quadro (100% completo). Nenhum número de
avaliação é tratado como válido sem essa aprovação.

## Recorte no dohyo: a correção decisiva

Mesmo com dados limpos, o detector ainda perdia robôs em movimento e gerava falso
positivo no fundo (precisão 0.71). Causa: os robôs ocupam fração pequena do quadro;
alimentar o quadro inteiro a um detector de 640 px os encolhe abaixo do que sobrevive
ao borrão de movimento. A correção foi **recortar no dohyo** antes de detectar (o
dohyo já é detectado pela visão clássica): o robô fica ~3x maior e o fundo some. As
caixas voltam a coordenadas nativas por um deslocamento. O gold revisado é
transformado para o espaço recortado deterministicamente, preservando a revisão
manual.

## Resultados medidos (gold aprovado, pipeline com recorte)

Conjunto após recorte e filtro de quadros completos: treino 423 (6 clips),
validação 59, gold 59 (1 round revisado manualmente).

**Detector.** YOLOv8s fine-tuned (E2) sobre o recorte: mAP@0.5 de 0.994 na validação
e **0.984 no gold held-out, com recall e precisão de 0.98**. Baseline COCO (E3):
mAP@0.5 de 0.026. O recorte levou recall 0.91→0.98 e precisão 0.71→0.98.

**Viabilidade.** Pipeline completa a 133 FPS, pico de 82 MB de VRAM (RTX 4070 Laptop).
SAM 3 anotador a ~2 FPS, ~7 GB.

**Rastreamento e eventos.** OC-SORT mantém A/B consistentes no round held-out,
inclusive com os robôs em movimento (validado no overlay nativo raw→recorte→volta).
Início de round dispara confiável; ring-out e contato precisam de calibração.
IDF1/HOTA dependem de um gold com identidades anotadas (próximo passo).

## Rastreamento medido (gold com identidades aprovado)

Geramos um gold com identidades A/B ligando as duas caixas por continuidade de
centroide; o usuário revisou e aprovou o overlay. Contra ele, OC-SORT: **IDF1 0.94,
MOTA 0.90, 1 ID switch**. O único switch ocorre na aproximação dos robôs idênticos:
a limitação esperada do tracker motion-only.

## Tentativa de fonte japonesa (heterogeneidade C3)

Baixamos um torneio regional japonês (câmera fixa cenital, oposto da BR de mão) para
estressar a heterogeneidade de qualidade. O segmentador de rounds funcionou muito
melhor aqui (26 rounds únicos limpos), e o detector clássico de dohyo generalizou de
cara para o footage JP.

A anotação SAM 3 no JP foi uma investigação em camadas:
- **Decimação a 480px**: SAM achava 0-1 robô. Diagnóstico instrumentado: o SAM
  rodava limpo (sem OOM) mas retornava 0 máscaras, pois o robô em vista cenital
  ficava com ~15px, abaixo do que o detector de conceito do SAM dispara.
- **Prompts**: testados toy, robot, sumo robot, black robot, machine. Nenhum mudou
  o resultado: não era problema de prompt.
- **Resolução 960px**: o robô dobra de tamanho e o SAM passa a pegar os dois robôs.
  Mas só em rounds que começam com os robôs nítidos e separados.
- **Gargalo real**: o SAM 3 vídeo semeia no frame 0; rounds JP costumam começar com
  os robôs colados/ocluídos/sendo posicionados, então o SAM não engata no round
  inteiro. Resultado: dos 8 rounds JP, só 1 anotou limpo (16 quadros com os dois).

Primeira conclusão (errada): "anotação semiautomática do JP não é viável". O usuário
questionou, e ao instrumentar o SAM descobri o verdadeiro motivo: o SAM 3 tem limiares
de detecção hardcoded (`new_det_thresh=0.7`, `score_threshold_detection=0.5`) altos
demais para os robôs japoneses, caixas pretas pequenas que pontuam baixo para o
conceito "toy". Varredura no round que dava 0 quadros: limiar 0.5 → 2 quadros com os
dois robôs; limiar 0.15 → 45/57. Adicionei limiar configurável (`--score-thresh`) e um
filtro geométrico que descarta caixas fora do dohyo (falsos positivos que surgem no
limiar baixo).

Resultado: o JP passou de 16 para **202 quadros com os dois robôs**. O conjunto virou
multi-fonte de verdade (treino 423 BR + 202 JP; gold com um round por fonte, ambos
revisados manualmente). Um único detector atinge mAP@0.5 de 0.985 no gold BR e 0.976
no gold JP, apesar das câmeras opostas (mão oblíqua vs cenital fixa): a restrição C3
deixou de ser projeção e virou resultado medido. Lição: antes de declarar uma
ferramenta "incapaz", instrumentar e checar os parâmetros padrão dela.

## Status

- Conjunto multi-fonte (423 BR + 202 JP no treino), com recorte no dohyo.
- Detector multi-fonte: mAP@0.5 0.985 no gold BR e 0.976 no gold JP (ambos aprovados
  manualmente); recall/precisão ~0.98; C3 medido, não projetado.
- Tracking: IDF1 0.93, MOTA 0.88, 1 ID switch (gold de identidades aprovado).
- Viabilidade: 133 FPS, 82 MB VRAM.
- Pendente: calibração dos limiares de evento (ring-out/contato) com timestamps do gold.
