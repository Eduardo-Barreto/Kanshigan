# Shift de escopo para a pré-banca e reformulação da pergunta-problema

## Contexto

Em 2026-05-19 surgiu uma demanda concreta: pré-banca em três semanas. Ela vai avaliar quatro critérios com pesos definidos:

| # | Critério | Peso |
|---|---|---|
| 1 | Contribuição científica (clareza da solução + 1 artigo publicável + repositório) | 35% |
| 2 | Validação e dados (métricas reais, baselines, análise crítica, honestidade dos dados) | 25% |
| 3 | Metodologia e execução (sprints, reprodutibilidade, pipeline experimental) | 25% |
| 4 | Posicionamento científico (mapa de literatura, diferencial original) | 15% |

Isso obriga a entregar até a pré-banca: artigo publicável com resultados parciais, repositório alinhado à metodologia, ao menos um modelo da pipeline planejada implementado e avaliado, e métricas reais (não projeções).

## O que muda em relação ao plano anterior

O [diário 08](08-sam3-nao-e-viavel-como-pipeline-final.md) já tinha estabelecido que o SAM 3 não seria o pipeline final de inferência. Esta entrada vai além: também muda a forma da pergunta-problema e a ordem de execução.

### Pergunta-problema anterior

> Como a escolha de arquitetura de visão computacional influencia a acurácia e viabilidade prática da classificação automática de partidas de Sumô de Robôs?

Boa parte é defensável: arquitetura como variável focal, acurácia e viabilidade como variáveis dependentes, domínio específico. Mas o verbo "influencia" e o objeto "classificação automática" carregam dois problemas para a pré-banca.

Primeiro, "influencia" é descritivo. A pergunta convida a um benchmark passivo: a gente compara A, B e C e relata o que mudou. Isso atende rigor, mas não responde à banca o que de fato a pesquisa entrega como artefato.

Segundo, "classificação automática" é vago. Não diz se o resultado é uma label de evento, uma trajetória, um vencedor ou uma série temporal de métricas. A metodologia já aponta para extração de métricas e detecção de eventos, então o termo precisa convergir com o que será medido.

### Pergunta-problema reformulada

> Quais escolhas de arquitetura de detecção e de algoritmo de tracking compõem uma pipeline de visão computacional que melhor equilibra acurácia e viabilidade prática para extração automatizada de métricas de desempenho em partidas de Sumô de Robôs autônomos?

O que mudou e por quê:

- "Quais escolhas compõem uma pipeline que melhor equilibra" troca o verbo descritivo por um verbo prescritivo. A pesquisa passa a construir a melhor pipeline possível dentro da metodologia, não apenas observar o efeito de cada peça.
- "Acurácia e viabilidade prática" continua como par dependente. Permite análise de Pareto entre os dois eixos, evitando otimização ingênua de um só.
- "Extração automatizada de métricas de desempenho" substitui "classificação automática". Define com precisão o que sai da pipeline: posição, velocidade, aceleração, trajetória e eventos derivados. Casa com o pipeline de cinco estágios já descrito na metodologia.
- O domínio continua estreito: Sumô de Robôs 3kg autônomos.

A variável independente principal continua sendo arquitetural: que combinação de detecção e tracking. A variável dependente é dupla: métricas de acurácia (mAP, MOTA, IDF1, ID switches, erro de métricas extraídas vs. ground truth) e métricas de viabilidade (FPS, VRAM, robustez entre vídeos heterogêneos).

### O papel do SAM 3 deixa de ser objeto da pergunta

O SAM 3 deixa explicitamente de ser variável comparada. Ele se torna meio metodológico para construir o dataset por anotação semi-automática com revisão humana. Quando a banca perguntar "por que SAM 3?", a resposta é metodológica (anotação eficiente em escala dada a ausência de dataset público do domínio), não científica (não estamos comparando SAM 3 com YOLO; estamos comparando arquiteturas leves entre si, treinadas com dados anotados via SAM 3).

Isso resolve uma fricção da versão anterior. Comparar SAM 3 com YOLO seria desonesto em viabilidade (já sabemos a resposta) e pouco interessante em acurácia (não é a pergunta certa para o domínio).

## Ordem de execução até a pré-banca

A nova ordem coloca SAM 3 antes da pipeline final, como insumo, e não como concorrente.

1. Avaliar SAM 3 com rigor no domínio. Esperado: boa acurácia, viabilidade ruim (VRAM, FPS). Essa evidência é necessária no artigo para justificar o uso de SAM 3 apenas como anotador.
2. Usar SAM 3 para gerar anotações sobre um subconjunto representativo de vídeos, com revisão manual.
3. Treinar um dos modelos leves da metodologia sobre o dataset anotado.
4. Avaliar o modelo leve com as mesmas métricas (acurácia e viabilidade).
5. Escrever o artigo no formato curto (modelo SBC IC), reportando resultados parciais e a metodologia consolidada.

## Critérios de "melhor pipeline"

Para "equilibra acurácia e viabilidade prática" não virar texto vazio na banca, ficam por fixar dois pontos. O primeiro: limiar de viabilidade explícito (por exemplo, FPS mínimo em GPU consumer-grade com VRAM máxima fixada) ou análise de Pareto sem limiar fixo. O segundo: ground truth contra o qual a acurácia das métricas extraídas é avaliada (anotação manual de subset, comparação com SAM 3 como referência forte, ou ambos). Essas decisões ficam pendentes para a próxima rodada de planejamento.

## Status

- Pergunta-problema reformulada e justificada nesta entrada.
- Papel do SAM 3 redefinido como ferramenta de anotação.
- Ordem de execução até a pré-banca esboçada.
- Pendente: critérios numéricos de viabilidade, escolha do modelo leve, escopo do dataset.

## Decisões consolidadas após mapeamento de literatura

Após a pesquisa documentada em [diário 11](11-mapa-de-literatura-related-work.md), as escolhas arquiteturais e de escopo ficam:

**Detector:** YOLOv8 small como experimento principal. RT-DETR fica como ablação opcional (apenas se sobrar tempo na janela de 3 semanas). Justificativa: W7 demonstra que YOLOv8l rastreia bola de futebol em broadcast; robô de 20cm em arena de 154cm é alvo muito mais fácil.

**Tracker:** OC-SORT como experimento principal. Sai do plano original (ByteTrack/BoT-SORT). Justificativa: a literatura W3 e W4 mostra que OC-SORT foi projetado pra movimento não-linear e atinge SOTA em DanceTrack, que é o benchmark mais análogo ao Sumô (aparência uniforme + movimento brusco). BoT-SORT é descartado porque sua camera motion compensation não paga em câmera cenital fixa do dohyo.

**Limitação assumida do OC-SORT:** ele é motion-only, sem features de aparência. Em frames de oclusão prolongada entre dois robôs visualmente idênticos, pode trocar identidades. Essa limitação fica documentada no artigo como expectativa, e Deep HM-SORT (W6, com features profundos) fica listado como próxima sprint para endereçar esse ponto. Se sobrar tempo na janela de 3 semanas, ByteTrack (já integrado ao Ultralytics, custo baixo de execução) entra como segundo ponto experimental na tabela, permitindo discussão comparativa empírica.

**Dataset:** 10-20 clips anotados via SAM 3 com revisão humana, mais 2 clips anotados manualmente frame a frame como gold set (1 broadcast JP, 1 amador BR). O gold set fica 100% fora do treino. Permite três medições independentes: SAM 3 vs gold (validação do anotador), modelo treinado vs gold (generalização), modelo vs SAM 3 (deriva).

**Artigo:** Short paper 4-6 páginas no formato SBC (eventos satélite, IC/CTIC/WTI). PT ou EN. Escopo: 1 experimento principal (YOLOv8 + OC-SORT), tabela comparativa de related work, descrição do dataset, discussão curta. Risco baixo para a janela de 3 semanas.

**Hardware alvo:** RTX 4070 Laptop 8GB VRAM (já validado no experimento SAM 3 PoC). Critério de viabilidade detalhado fica para a spec.

Os gaps que sustentam o diferencial científico (critério 4 da pré-banca, 15%): Sumô de Robôs como domínio inédito em CV externa, aparência uniforme extrema sem marcadores fiduciais, rounds curtíssimos com decisão crítica em sub-segundo, vídeo de qualidade heterogênea (broadcast vs amador), métricas específicas de combate (jyusho, ippon), arquitetura validada para análise post-match.
