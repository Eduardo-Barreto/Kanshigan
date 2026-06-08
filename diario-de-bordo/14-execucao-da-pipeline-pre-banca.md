# Execução da pipeline da pré-banca

## Contexto

Esta entrada conta a execução do plano dos diários [12](12-design-da-pipeline-pre-banca.md)
e [13](13-protocolo-experimental-e-cronograma.md): da aquisição de dados ao detector
multi-fonte avaliado. É um relato da jornada, então registra também o que falhou, por
quê, e o que corrigiu cada ponto. Os números finais aparecem consolidados na seção de
resultados; o texto antes dela mostra como chegamos neles.

## Aquisição de dados

Sem dataset público do domínio, os clips vêm de torneios reais baixados com `yt-dlp`,
de duas fontes propositalmente diferentes (para a restrição de qualidade heterogênea,
C3):

- **Brasil (IRONCup 2025, canal ThunderRatz):** seis partidas autônomas de 3 kg,
  câmera de mão em ângulo oblíquo. Cada vídeo tem cartões de abertura, patrocínio e
  b-roll; os trechos de partida foram identificados por contact-sheets.
- **Japão (torneio regional):** câmera fixa cenital, oposto da captura brasileira.
  Vídeo contínuo de várias partidas.

O `configs/clips.yaml` (fora do git; dado versionado por DVC) lista os recortes. A live
de 4h do All Japan 2025 foi descartada como fonte primária: metadados de capítulos
inconsistentes e muito corte; ela reaparece adiante só como teste extremo.

## Premissas de design que a footage real quebrou

Três suposições do plano não sobreviveram ao contato com os dados, e cada uma virou uma
correção na pipeline:

**A câmera não é cenital fixa (no BR).** A captura brasileira é de mão, com movimento e
ângulo oblíquo. Por isso a detecção do dohyo passou a ser por quadro (reusando a última
detecção válida nas falhas) e as métricas passaram a viver no referencial centrado no
dohyo, o que cancela o movimento da câmera, já que o dohyo é fixo no mundo. A calibração
centímetro-por-pixel por escala isotrópica vira aproximação sob perspectiva; a
retificação por homografia fica para o trabalho final.

**Detectar o dohyo pela "maior região branca" falha.** O dohyo é uma plataforma escura
com anel branco (tawara) sobre tapete colorido, com fundo ruidoso (overlays, cercas,
plateia). A heurística do PoC pegava objetos brancos do fundo. A correção foi pontuar
cada elipse candidata por tamanho, centralidade e razão de aspecto (um círculo visto de
lado é mais largo que alto) e escolher a de maior escore. Validado em vários frames
reais, incluindo a arena japonesa de cor diferente.

**SAM 3 estoura a VRAM de 8 GB em clips longos.** O preditor de vídeo guarda feature
maps de todos os quadros da sessão. A correção: processar em janelas curtas, encerrar a
sessão entre janelas (`close_session`) para liberar memória, e ler os quadros nativos
sob demanda em vez de carregar o clip inteiro em RAM (84 s a 1080p60 não cabe nos 30 GB).

## Qualidade dos dados: três iterações

A primeira passada deu números bonitos mas enganosos, porque os dados estavam ruins, não
a pipeline. Cada problema foi diagnosticado e corrigido:

**Clips multi-round e ending.** Cada vídeo concatena vários rounds mais um encerramento
de patrocínio. Em clip multi-round o SAM perdia um robô em boa parte dos quadros; em
round único limpo ele pega os dois. Solução: segmentar em rounds únicos por motion mais
presença do dohyo (`segment_rounds.py`) e descartar as endings.

**Rótulos incompletos.** Quadro com só um robô é rótulo incompleto que ensina o detector
a ignorar um robô. Passamos a treinar apenas com quadros de rótulo completo (os dois
robôs).

**Robôs pequenos somem no quadro inteiro (a correção decisiva).** Mesmo com dados
limpos, o detector perdia robôs em movimento e gerava falso positivo no fundo (precisão
0.71). Causa: o robô ocupa fração pequena do quadro, e reduzir o quadro inteiro para 640
px o encolhe abaixo do que sobrevive ao borrão de movimento. Solução: **recortar no
dohyo** antes de detectar (o dohyo já é detectado). O robô fica cerca de três vezes
maior e o fundo some; as caixas voltam às coordenadas nativas por um deslocamento. Isso
levou recall e precisão de ~0.9/0.71 para ~0.98/0.98.

**Gold revisado por humano (gate de aprovação).** Os golds (um round por fonte) são
pré-anotados pelo SAM e revisados manualmente quadro a quadro pelo autor. Nenhum número
de avaliação é tratado como válido sem essa aprovação.

## A fonte japonesa: de "inviável" a multi-fonte

A anotação SAM no JP parecia falhar por completo, e quase a descartamos como inviável.
Instrumentando o SAM, o motivo real apareceu: ele tem limiares de detecção fixos no
código (`new_det_thresh=0.7`, `score_threshold_detection=0.5`) altos demais para os
robôs japoneses, caixas pretas pequenas que pontuam baixo para o conceito textual. Não
era prompt nem resolução nem ponto de corte: era o limiar. Uma varredura num round que
dava zero quadro: limiar 0.5 dava 2 quadros com os dois robôs; limiar 0.15 dava 45 de
57. Adicionamos um limiar configurável (`--score-thresh`) e um filtro geométrico que
descarta caixas fora do dohyo (falsos positivos que surgem no limiar baixo).

Com isso o JP foi de 16 para **202 quadros com os dois robôs**, e o conjunto virou
multi-fonte de verdade. A restrição C3 deixou de ser projeção e virou resultado medido.
Lição: antes de declarar uma ferramenta incapaz, instrumentar e checar os parâmetros
padrão dela.

## Resultados medidos

Conjunto, após segmentar em rounds únicos, recortar no dohyo e filtrar quadros de rótulo
completo (treino e validação) com gold revisado manualmente:

| Subconjunto | BR | JP |
|---|---|---|
| Treino | 423 | 202 |
| Validação | 59 | 15 |
| Gold (teste) | 59 | 57 |

**Detector multi-fonte (YOLOv8s fine-tuned, E2)**, no gold held-out de cada fonte:

| Fonte | mAP@0.5 | mAP@0.5:0.95 | precisão | recall |
|---|---|---|---|---|
| Gold BR (câmera de mão) | 0.985 | 0.781 | 0.99 | 0.98 |
| Gold JP (cenital fixa) | 0.976 | 0.695 | 0.99 | 0.91 |
| E3 baseline COCO (gold BR) | 0.026 | 0.017 | 0.03 | 0.75 |

Um único detector acima de 0.97 nas duas fontes, apesar das câmeras opostas, é a entrega
concreta de C3. O baseline COCO sem fine-tuning, duas ordens de grandeza abaixo, mostra
que o domínio exige treino específico.

**Rastreamento (OC-SORT, gold de identidades aprovado):** IDF1 0.93, MOTA 0.88, 1 troca
de identidade no round. O único switch ocorre na aproximação dos dois robôs idênticos: a
limitação esperada de um tracker motion-only.

**Viabilidade (RTX 4070 Laptop 8 GB):** pipeline completa a 133 FPS com pico de 82 MB de
VRAM. O SAM como anotador roda a ~2 FPS com ~7 GB, o que justifica usá-lo só como
anotador, nunca na inferência.

**Eventos:** o início de round dispara de forma confiável; ring-out e primeiro contato
ainda dependem de calibração de limiar com timestamps marcados no gold.

## Caso extremo: final de mundial (o alvo máximo)

Testamos pipeline e SAM na final do 84º All Japan Robot Sumo (3 kg autônomo): broadcast
muito fora da distribuição de treino, com arena azul-escura, overlay de placar, cortes
frequentes e colisões de blur extremo. A detecção do dohyo generaliza para a arena nova.
No Round 1, nosso modelo e o SAM acham os dois robôs na maior parte dos quadros; no auge
da colisão o blur derruba ambos. O replay em câmera lenta, contra a intuição, é mais
difícil para o SAM (o replay desfoca cada quadro). O gargalo não é a semelhança nem as
bandeiras dos robôs: é o blur do combate de elite. Esse vídeo fica registrado como o
alvo máximo do projeto, e fechá-lo exige dados de treino dessa distribuição.

Detalhe de método: ao comparar modelo e SAM, o vídeo do SAM parecia pior por um recorte
ruim no script de teste (amostrava os primeiros quadros para achar a ROI, em vez de
espalhados como a pipeline faz). Corrigida a amostragem, o SAM no Round 1 saltou de 18
para 45 de 61 quadros com os dois robôs: a qualidade do recorte domina a anotação.

## Status

- Pipeline ponta a ponta com recorte no dohyo, testada na lógica pura (sem GPU).
- Conjunto multi-fonte (BR + JP), golds aprovados manualmente.
- Detector mAP@0.5 0.985 (BR) e 0.976 (JP); tracking IDF1 0.93 / MOTA 0.88; 133 FPS.
- Artigo SBC, README e este diário consolidados; resultados reprodutíveis.
- Pendente: calibração dos eventos (ring-out e contato) com timestamps do gold; treino
  com a distribuição de broadcast de mundial.
