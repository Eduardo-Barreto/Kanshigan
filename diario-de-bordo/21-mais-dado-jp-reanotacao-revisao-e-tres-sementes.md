# Mais dado JP: reanotação, revisão em grupo e o que três sementes mostram

## Contexto

A pergunta de partida era simples: sobrou vídeo sem anotar que o SAM 3 consiga anotar? Sobrou, e bastante. O caminho até um resultado passou por um bug no filtro do anotador, um editor web para revisar em grupo e uma conclusão que mudou duas vezes antes de três sementes a fixarem. Esta entrada registra o caminho inteiro, incluindo as leituras erradas.

## O que estava sem anotar

Cruzando `data/raw/` com `configs/clips.yaml` e com `data/annotations/`:

| Fonte | Duração | Situação |
|---|---|---|
| `raw/jp/Z_MC3r_c2P0.mp4` | 412 s | 10 dos 26 trechos segmentados em uso (8 treino, 1 validação, 1 gold) |
| `raw/jp/s0Y09ZC9kHI.mp4` | 162 s | nunca cortado |
| `raw/jp/probe_3kg_680_900.mp4` | 220 s | nunca cortado |
| `raw/jp/worlds_jE4E-XhiyjU_12401.mp4` | 90 s | só inferência; caso de falha conhecido (diário 20) |

Escolhemos os 16 trechos restantes do `Z_MC3r`: mesma câmera, mesmo formato, e o `segment_rounds.py` reproduz exatamente as janelas dos clips já em uso (r03 em 36,0 a 40,2 s, e assim por diante), então os limites dos novos são confiáveis.

O `prep_clips.py` reprocessava o `clips.yaml` inteiro a cada execução e sobrescreveria os clips de treino e gold. Ganhou a opção `--config`; os 16 cortes ficam em `configs/clips_jp_candidates.yaml`, com prefixo `cand_jp_`.

## Primeira anotação e o tripé

Rodamos o SAM 3 com os parâmetros dos JP atuais (prompt "toy", limiar 0,15) numa pasta separada, `candidates_jp`, sem tocar em treino, validação ou gold.

O r26 estourou a VRAM de 8 GB duas vezes, no primeiro bloco de 60 quadros. Olhando o clip, ele nem é da câmera fixa: é um trecho de câmera de mão, em ângulo baixo. Ficou de fora.

Os outros 15 deram 458 quadros rotulados, 334 com duas caixas. O número era enganoso. Uma prancha com um quadro por round mostrou caixas no tripé de uma câmera e em riscos do dohyo, e robôs sem caixa. Contar linhas por arquivo não mede acerto.

O caso do tripé tinha causa no nosso código, não no SAM. No `cand_jp_r01`, o SAM devolvia três detecções: robô da direita, tripé, robô da esquerda. O `annotate_to_yolo.py` descartava justamente o robô, por três motivos somados:

1. O recorte é o retângulo do dohyo com 20% de margem (`crop.py`), então os cantos ainda mostram o que está fora da arena.
2. O filtro "dentro do dohyo" usa a elipse inscrita nesse recorte, 20% maior que a arena. O centro do tripé dava 0,71 nessa conta, abaixo do limite de 1,0. O filtro de tamanho só olhava área (15%, abaixo do teto de 20%), embora a caixa ocupasse 89% da altura.
3. O corte em duas caixas (`[:MAX_OBJECTS]`) seguia a ordem do SAM, não a plausibilidade. O tripé vinha em segundo.

Correção: ordenar as caixas pela distância ao centro do dohyo antes de cortar, e rejeitar caixas mais altas que metade do recorte. Reanotando, 101 quadros mudaram de conteúdo. O r01 passou a ter os dois robôs. O r02 zerou: as 10 caixas dele eram todas o tripé, e o SAM nunca encontrou os robôs naquele round.

A contagem de quadros com duas caixas continuou em 334. O ganho foi de qualidade (tripé trocado por robô), não de quantidade.

A mudança vale para anotações futuras. Os rótulos de treino, validação e gold já existentes não foram regerados.

## Editor web para revisar em grupo

A revisão precisava acontecer de outra máquina, e depois por mais de uma pessoa. O `editor.py` é uma janela OpenCV local. Escrevemos o `web_editor.py`: mesmo trabalho, servido por HTTP, só com a biblioteca padrão.

O que entrou, na ordem em que a revisão pediu:

- Edição de caixas com o mouse e status por quadro (aprovado ou rejeitado) em `review.json`. Rejeitar não apaga arquivo.
- Play a 10 quadros por segundo (a taxa da anotação), com velocidade de 0,5x a 4x, pausa na troca de clip e a opção de aprovar cada quadro ao avançar.
- Uma caixa já selecionada ao abrir cada quadro, Tab para trocar, WASD para mover, Shift+WASD para redimensionar, Delete para apagar. As correções mais comuns deixaram de precisar do mouse.

Por padrão o servidor escuta só no IP do Tailscale. Para o grupo revisar de fora da tailnet, publicamos por Tailscale Funnel, sem autenticação, por decisão consciente e por tempo limitado. Um tropeço: o primeiro comando do Funnel tomou a porta 443, que já servia outra aplicação restrita à tailnet. Foi revertido em seguida e o editor foi para a 8443. Antes de publicar qualquer coisa por Funnel, vale olhar `tailscale serve status`.

Resultado da revisão: os 448 quadros dos 14 rounds restantes foram aprovados, 430 com duas caixas e 18 com uma.

## Quanto a revisão mudou a proposta do SAM

A primeira conta que fiz estava inflada: 390 quadros "alterados", comparando com os rótulos de antes da reanotação, onde muita coisa só tinha mudado a ordem das caixas. Para medir direito, rodamos o SAM de novo com o filtro corrigido numa pasta à parte (`candidates_jp/labels_sam_rerun`) e comparamos com os rótulos revisados pelo `evaluate.py`, por IoU.

| Quadros | Caixas revisadas | Precisão | Recall | F1 | IoU médio (casadas) | Quadros alterados |
|---|---|---|---|---|---|---|
| 448 | 878 | 0,80 | 0,71 | 0,75 | 0,94 | 318 (71%) |

Quando o SAM propõe uma caixa certa, ela quase coincide com a humana (IoU 0,94). O problema é cobertura: ele deixa cerca de três em cada dez robôs sem caixa, e uma em cada cinco caixas propostas não casa com nenhuma revisada. A revisão manual não foi acabamento, foi o que tornou os rounds utilizáveis.

Ressalva: a revisão partiu da proposta do SAM, então essa concordância é um limite superior. O resultado fica em `results/E5_jp_data_scaling/annotator_agreement.json` e entrou como linha nova na tabela de validação do anotador do artigo.

## Retreino com uma semente: duas leituras erradas

Decisão de dado: os 18 quadros de uma caixa entraram no treino. Isso relaxa a regra do diário 14 (só quadros com os dois robôs), com a justificativa de que agora há revisão humana quadro a quadro, e uma caixa só pode ser o rótulo correto.

O treino JP foi de 202 para 650 quadros. Para a comparação ficar limpa, o `train/` congelado não mudou: o `configs/kanshigan_jpplus.yaml` soma `train/` e `candidates_jp/`, e o `train.py` ganhou `--data`. A linha de base, reavaliada nesta máquina, reproduziu o diário 16 dígito a dígito.

Com a semente 42:

| Gold | mAP@0.5 antes | depois | mAP@0.5:0.95 antes | depois |
|---|---|---|---|---|
| BR | 0,965 | 0,975 | 0,767 | 0,752 |
| JP | 0,976 | 0,977 | 0,695 | 0,709 |

Primeira leitura: detector saturado, diferença dentro do ruído, a anotação não é o gargalo. Meio certa, mas tirada só do mAP.

Segunda leitura, depois de comparar as predições quadro a quadro contra o gold: no JP, o detector novo deixava de perder robôs (5 para 0) e os falsos positivos dentro do dohyo iam de 13 para 0. Parecia um ganho limpo escondido pelo mAP. As outras sementes desmentiram metade disso.

## Três sementes

Repetimos as duas configurações com as sementes 0 e 1 (o `train.py` ganhou `--seed`). Um dos quatro treinos foi morto na época 19 pelo limite de tempo da tarefa em segundo plano e foi refeito do zero. O `experiments/pos-banca/data_scaling.py` avalia tudo e separa os erros por lugar.

Média ± desvio padrão de três sementes (42, 0, 1); contagens por semente, com confiança mínima 0,25 e casamento por IoU ≥ 0,5:

| Gold / treino | mAP@0.5 | mAP@0.5:0.95 | Robôs perdidos | FP dentro do dohyo | FP fora |
|---|---|---|---|---|---|
| BR / original | 0,968 ± 0,006 | 0,786 ± 0,018 | 4, 4, 4 | 1, 28, 1 | 0, 0, 0 |
| BR / + 14 rounds JP | 0,957 ± 0,032 | 0,728 ± 0,047 | 3, 4, 3 | 1, 70, 1 | 0, 1, 1 |
| JP / original | 0,963 ± 0,019 | 0,669 ± 0,023 | 5, 5, 10 | 13, 1, 0 | 17, 20, 9 |
| JP / + 14 rounds JP | 0,978 ± 0,009 | 0,717 ± 0,020 | 0, 2, 1 | 0, 21, 1 | 54, 37, 47 |

O gold BR tem 112 robôs e o JP tem 82.

O que se sustenta nas três sementes:

- **JP melhora.** Menos robôs perdidos (5 a 10 para 0 a 2) e caixas mais justas: o mAP@0.5:0.95 das três sementes novas (0,703 a 0,740) fica acima das três originais (0,652 a 0,695). Os casos recuperados incluem robô na borda do quadro e robô borrado pelo movimento.
- **BR piora.** O mAP@0.5:0.95 cai nas três sementes (0,674 a 0,757 contra 0,767 a 0,804). Os quadros novos são todos JP e invertem a proporção do treino (423 BR para 650 JP). O desbalanceamento é a explicação provável, mas não foi testado isoladamente.
- **Detecções fora da arena triplicam no JP.** Nos quadros que olhamos, são o robô já ejetado. A primeira explicação foi que os rounds novos rotulam esse robô e o gold não. Conferindo, o treino JP original já rotulava na mesma proporção: 71 de 404 caixas (17,6%) ficam fora da elipse da arena, contra 153 de 878 (17,4%) nos rounds novos e 1 de 82 no gold JP. A divergência de convenção é entre treino e gold e já existia. O aumento é consistente com um detector que aprende melhor o que o treino já pedia, mas não conferimos as detecções uma a uma.

O que não se sustenta:

- **mAP@0.5.** Os intervalos se sobrepõem nos dois golds.
- **Falsos positivos dentro do dohyo.** A história "13 para 0" era da semente 42. Com a semente 0, o detector novo tem 21 no JP e 70 no BR. Olhando os quadros do BR, é um robô só partido em duas ou três caixas parciais, e acontece nas duas configurações com essa semente (28 e 70). É efeito de semente, não de dado.

Um achado lateral: o desvio entre sementes no mAP@0.5 do gold JP (0,019) é da mesma ordem da diferença entre YOLOv8s e YOLO26n na tabela do artigo (0,017). Isso é compatível com a leitura de equivalência prática que o artigo já fazia, sem prová-la: cada arquitetura tem uma semente só.

Três sementes indicam direção, não significância.

## Decisões

- O detector reportado continua sendo o original (semente 42, treino sem os rounds novos). Os pesos e o `train/` não mudaram.
- O resultado entra no artigo SBC como uma seção própria de escala de dados (4.3), com tabela e figura de antes e depois, sem trocar os números principais.
- Trabalhos futuros no artigo passam a citar o reequilíbrio das fontes e a convenção de rótulo do robô ejetado, no lugar de "ampliar a base japonesa".

## Lições

- Contar rótulos não é medir rótulos. Os 334 quadros "com duas caixas" incluíam tripé.
- Antes de culpar o anotador, olhar a saída bruta dele. O SAM tinha achado o robô; o nosso filtro jogou fora.
- Uma semente conta uma história convincente e errada. A leitura mudou duas vezes entre o primeiro retreino e a terceira semente.
- mAP sozinho esconde a anatomia do erro. Separar perdidos, falsos positivos dentro e fora da arena mostrou o que mudou e também expôs a divergência de convenção.

## Pendências

- Os rótulos revisados (`candidates_jp`) existem só na máquina com a 4070. O DVC não tem nenhum arquivo rastreado nem remoto configurado; falta versionar.
- Unificar a convenção do robô ejetado entre treino e gold, e reavaliar. Ela pode afetar também a precisão já reportada no gold JP, porque o detector original também marca o robô ejetado (17, 20 e 9 detecções fora da arena nas três sementes).
- Testar o desbalanceamento: retreinar com as fontes equilibradas (subamostrar o JP ou anotar mais BR).
- O YOLO26n não foi retreinado com os rounds novos.
- `s0Y09ZC9kHI.mp4` e `probe_3kg_680_900.mp4` seguem sem cortar nem anotar.
