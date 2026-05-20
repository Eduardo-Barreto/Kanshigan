# Protocolo experimental, artigo e cronograma da pré-banca

## Contexto

Esta entrada documenta o protocolo experimental, a estrutura do artigo curto formato SBC, a organização do repositório e o cronograma de três semanas até a pré-banca. Complementa o [diário 12](12-design-da-pipeline-pre-banca.md), que detalhou a pipeline e o dataset.

## Pontos experimentais

O estudo executa de três a cinco configurações, com as duas últimas condicionais à folga de tempo:

| ID | Configuração | Função no estudo |
|----|--------------|------------------|
| E1 | SAM 3 (oracle) avaliado no gold set | Valida o SAM 3 como anotador. Estabelece o teto de acurácia da pipeline de anotação. |
| E2 | YOLOv8s + OC-SORT, treinado no dataset SAM-anotado revisado | Experimento principal. Resposta à pergunta-problema. |
| E3 | YOLOv8s pré-treinado COCO sem fine-tuning + OC-SORT | Baseline negativo. Mostra o ganho do fine-tuning no domínio. |
| E4 | YOLOv8s + ByteTrack (se sobrar tempo) | Comparação empírica de trackers. ByteTrack já vem integrado ao Ultralytics, custo de execução baixo. |
| E5 | RT-DETR + OC-SORT (se sobrar tempo) | Ablação de detector. |

E3 é importante para a defesa científica: se YOLO COCO já resolve, o fine-tuning é desnecessário; se não resolve, o esforço de dataset e treino fica justificado.

## Métricas

| Categoria | Métrica | Como medir |
|-----------|---------|------------|
| Detecção | mAP@0.5, mAP@0.5:0.95 | Ultralytics val padrão, no gold set |
| Tracking | MOTA, IDF1, HOTA, ID Switches | Biblioteca motmetrics ou TrackEval, alinhado com SportsMOT e SoccerNet |
| Métricas extraídas | Erro médio absoluto de posição (cm), erro médio absoluto de velocidade (cm/s) | Comparação ponto a ponto contra trajetórias do gold |
| Eventos | Precision, recall e erro temporal médio (ms) por tipo | Detector cinemático de contato contra julgamento humano; ring-out via geometria contra ground truth |
| Viabilidade | FPS end-to-end (batch=1, warm start), pico de VRAM (nvidia-smi) | 3 runs separadas, reporta média e desvio |
| Robustez | Variância das métricas entre subset JP e subset BR | Reportada separadamente, não só agregada |

## Protocolo de treino do YOLOv8s

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Pesos iniciais | yolov8s.pt (COCO pré-treinado) | Padrão Ultralytics |
| imgsz | 640 | Padrão Ultralytics; robôs ocupam fração significativa do frame |
| Epochs | 100 com early stopping (val mAP plateau por 20 epochs) | Dataset pequeno, evita overfitting |
| Batch | 16 ou o máximo que couber na 4070 | |
| Optimizer | AdamW (padrão atual do Ultralytics) | |
| Learning rate | Default Ultralytics | Sem hyperparameter sweep nesta sprint |
| Augmentation | Default Ultralytics (mosaic, flip, hsv) | Não inventa, usa o padrão |
| Seed | 42, fixo e documentado | Reprodutibilidade |
| Hardware | RTX 4070 Laptop 8GB | Mesmo do PoC SAM 3 |

## Protocolo de avaliação

1. O modelo treinado fica congelado depois do treino. Nenhum ajuste de hiperparâmetro olhando o gold set.
2. Inferência roda em fps nativo de cada clip do gold (60 fps JP, 30 fps BR ou o que vier no arquivo).
3. Métricas reportadas separadamente por subset (JP, BR) e agregadas.
4. Viabilidade medida com a pipeline completa, do `cv2.VideoCapture` até a escrita do JSON. Não só o forward do YOLO; inclui detecção de dohyo, tracking, métricas e eventos.

## Análise estatística e limitação assumida

Com gold set de 2 clips, intervalos de confiança e testes de significância ficam fracos. A honestidade exige declarar isso no artigo: o gold é amostra pequena, os números reportados têm incerteza que não vamos quantificar com rigor estatístico (n=2 não permite). O foco é demonstrar viabilidade e medir desempenho em ordem de grandeza, não estabelecer modelo definitivo. Esse posicionamento responde diretamente ao critério 2 da pré-banca (validação e dados, 25 por cento), no qual projeções sem base experimental são penalizadas.

## Ground truth do detector cinemático de contato

A calibração de `v_min_inicio`, `Δv_threshold` e `d_threshold` é feita nos 2 clips do gold pelo método mais simples possível: gradiente discreto sobre uma malha pequena de valores, otimizando precision e recall combinados. Os valores escolhidos ficam documentados nos configs.

Como calibrar hiperparâmetros num test set pequeno é fonte de viés, a mitigação é dupla: reporta precision e recall com os valores calibrados e com valores padrão de literatura (por exemplo, `Δv > 50 por cento` da velocidade média do round). Esse duplo reporte mostra que o detector não depende exclusivamente do gold.

## Estrutura do artigo (formato SBC, 4 a 6 páginas)

| Seção | Páginas | Conteúdo |
|-------|---------|----------|
| Resumo + palavras-chave | 0.25 | 150 a 200 palavras: problema, pergunta, abordagem, resultado, contribuição |
| 1. Introdução | 0.75 | Contexto, pergunta-problema, contribuições (pipeline, dataset, matriz combinatória) |
| 2. Trabalhos Relacionados | 1.0 | Tabela enxuta (8 a 10 trabalhos), matriz de combinatorialidade como tabela ou figura, parágrafo do gap combinatório |
| 3. Metodologia | 1.5 | Pipeline (5 estágios, figura), SAM 3 como anotador, dataset (composição, splits, gold), detector cinemático de contato |
| 4. Resultados | 1.5 | Tabela E1/E2/E3 (mAP, IDF1, HOTA, FPS, VRAM), tabela de erros de métricas, 2 a 3 figuras |
| 5. Discussão e Trabalhos Futuros | 0.5 | Falhas observadas, Deep HM-SORT como próximo passo, ablação RT-DETR pendente, escala do dataset, contato via máscara como evolução |
| Referências | 0.5 | 15 a 20 referências do mapa de literatura |

Idioma: português, alinhado com a maioria dos eventos SBC IC, CTIC e WTI. Inglês fica opcional caso surja uma submissão internacional viável.

Submissão alvo: WTI (Workshop de Trabalhos de Iniciação), ENCOMPIF ou similares com janela de submissão aberta nas próximas 6 a 8 semanas. Mesmo sem submissão imediata, o artigo fica pronto como preprint no arXiv para ter URL citável na banca.

## Estrutura do repositório na pré-banca

```
Kanshigan/
├── README.md                     atualizado: visão, como reproduzir, links
├── docs/
│   ├── paper-sbc/                NOVO: artigo SBC (Typst ou LaTeX)
│   ├── paper/                    IEEE existente (artigo final do TCC)
│   ├── projeto-de-pesquisa/      ABNT existente
│   └── project-plan/             Inteli existente
├── experiments/
│   ├── sam3-poc/                 existente
│   └── pre-banca/                NOVO
│       ├── README.md             como rodar cada script, na ordem
│       ├── annotate.py
│       ├── train.py
│       ├── infer.py
│       ├── evaluate.py
│       ├── metrics.py
│       ├── kanshigan.yaml
│       └── configs/              hiperparâmetros versionados
├── data/                         DVC tracked
├── notebooks/                    existente
├── results/                      NOVO: outputs de cada run
│   ├── E1_sam3_vs_gold/
│   ├── E2_yolo_oc_vs_gold/
│   ├── E3_yolo_coco_vs_gold/
│   └── figures/
└── diario-de-bordo/              existente, manter alimentado
```

### Critérios de reprodutibilidade na pré-banca

- README com passo a passo: clonar, instalar via uv, baixar dados via DVC, rodar `experiments/pre-banca/infer.py video.mp4`.
- Hiperparâmetros versionados no git, não escondidos em comentários.
- Seed fixo e documentado.
- Vídeos de exemplo em `results/figures/` para ilustrar.
- Logs de treino salvos, mesmo que como JSON, sem necessidade de W&B.

## Cronograma de 3 semanas

```
Semana 1 (anotação)
├─ D1-2: Setup pre-banca dir, instala boxmot/OC-SORT, sobe CVAT local (docker)
├─ D2-3: Coleta de clips (10 a 20 + 2 gold); decimate ffmpeg + cortes
├─ D3-5: Anotação manual do gold (2 clips JP+BR frame a frame)
└─ D5-7: SAM 3 anota os 10 a 20 clips; importa CVAT; revisão humana

Semana 2 (treino + avaliação)
├─ D8-9: Treina YOLOv8s (E2); val durante o treino
├─ D10: Roda E1 (SAM 3 vs gold) e E3 (YOLO COCO vs gold)
├─ D11: Roda E2 (YOLO fine-tuned vs gold); coleta métricas todas
├─ D12: Detector cinemático de contato + calibração no gold
├─ D13: Geração de figuras (trajetórias, exemplos, evento)
└─ D14: Análise qualitativa de falhas; se sobrar tempo, dispara E4 (ByteTrack)

Semana 3 (artigo + repo + polimento)
├─ D15-17: Escrita do artigo SBC (rascunho completo)
├─ D18: Revisão com orientador; ajustes
├─ D19: README do repositório, instruções de reprodução
├─ D20: Build final do artigo via CI, tag de versão pré-banca
└─ D21: Buffer
```

### Checkpoints com o orientador

- Fim da semana 1: gold set anotado, dataset montado. A coleta funciona?
- Fim da semana 2: tabela E1/E2/E3 com números reais. Os resultados fazem sentido?
- Meio da semana 3: rascunho do artigo. O argumento está claro?

## Mapeamento contra critérios da pré-banca

| Critério | Peso | Como o design atende |
|----------|------|----------------------|
| 1. Contribuição científica | 35 por cento | Artigo SBC submetido ou preprint + repositório reproduzível + dataset + matriz combinatorial inédita |
| 2. Validação e dados | 25 por cento | Gold set manual (n=2 declarado como limitação), E1/E2/E3 com métricas reais medidas, nenhuma projeção |
| 3. Metodologia e execução | 25 por cento | Sprints semanais documentadas no diário, pipeline 5-estágios documentado, hiperparâmetros versionados, repo reproduzível com README |
| 4. Posicionamento científico | 15 por cento | Tabela comparativa de 18 trabalhos + matriz de combinatorialidade do problema |

## Riscos da janela de 3 semanas

| Risco | Probabilidade | Mitigação |
|-------|---------------|-----------|
| Anotação consumir mais tempo (especialmente revisão das pseudo-labels) | Alta | Se na quarta-feira da semana 1 a revisão estiver abaixo de 50 por cento, corta dataset para 10 clips |
| Treino do YOLOv8 não convergir, dataset muito pequeno | Média | Augmentation padrão Ultralytics; se mAP val abaixo de 0.5, escala para YOLOv8m ou amplia dataset com mais clips do mesmo round |
| Detector cinemático de contato gera muito falso positivo ou negativo | Média | Fallback de retirar primeiro contato do escopo já planejado |
| Imprevisto de hardware (GPU, infra) | Baixa | Backup: Colab T4 grátis (16GB) para emergência |
| Estouro de prazo do artigo | Média | Sem capítulo 5 (Discussão) fica em 1 parágrafo se for o caso |

## Status

- Pontos experimentais definidos (E1 a E5, com E4 e E5 condicionais).
- Métricas, protocolo de treino e protocolo de avaliação fechados.
- Estrutura do artigo e do repositório alinhada com os critérios da pré-banca.
- Cronograma de 3 semanas com checkpoints semanais e mitigações documentadas.
- Próximo passo: executar a semana 1 (setup, coleta e início da anotação).
