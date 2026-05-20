# Design da pipeline e do dataset para a pré-banca

## Contexto

Esta entrada documenta o design da pipeline e do dataset que serão executados na janela de três semanas até a pré-banca. Complementa o [diário 10](10-shift-pre-banca-e-reformulacao-da-pergunta.md), que reformulou a pergunta-problema, e o [diário 11](11-mapa-de-literatura-related-work.md), que mapeou a literatura e justificou as escolhas arquiteturais.

## Dataset e protocolo de anotação

### Composição

O dataset divide-se em dois subconjuntos com funções distintas e processos de anotação diferentes.

| Subconjunto | Tamanho | Anotação | Função |
|-------------|---------|----------|--------|
| Train + Val | 10 a 20 clips de rounds individuais | SAM 3 gera pseudo-máscaras, conversão para YOLO bbox, revisão manual no CVAT | Treino do detector |
| Gold set | 2 clips (1 broadcast JP, 1 amador BR) | 100 por cento manual, frame a frame, sem auxílio do SAM 3 | Test set único, fora do treino |

O gold set sustenta três medições independentes na fase de avaliação: SAM 3 contra gold (validação do anotador), modelo treinado contra gold (generalização) e modelo contra SAM 3 (deriva entre o que foi aprendido e o sinal supervisor).

### Anotação por frame

Cada frame anotado contém:

- Bounding box de cada robô em jogo, até dois robôs por round. Rumbles ficam fora do escopo.
- Elipse do dohyo com centro, semi-eixos e rotação.
- Identidade consistente do robô ao longo do round, com convenção Robô A igual ao mais à esquerda no frame zero.

### Anotação temporal por clip

Cada clip recebe timestamps para:

- Início do round, definido como o primeiro deslocamento significativo de qualquer robô.
- Primeiro contato entre robôs, com nota de incerteza do anotador (a definição operacional fica para a pipeline cinemática descrita abaixo).
- Ring-out, definido como o centro do robô saindo da elipse do dohyo.
- Fim do round.
- Vencedor, A ou B.

### Pipeline de anotação semi-automática

```
Vídeo bruto
    │
    ▼
ffmpeg decimate a 10 fps + resize para 480x270 (limitação do SAM 3 PoC)
    │
    ▼
SAM 3 video predictor com prompt "object on metal platform" e ROI dinâmica do dohyo
    │
    ▼
Máscaras por frame, com filtros do PoC (área e proximidade ao centro)
    │
    ▼
Conversão máscara para bounding box via boundingRect do maior contorno conexo
    │
    ▼
Importação no CVAT como pré-anotação
    │
    ▼
Revisão humana: corrige caixas, deleta falsos positivos, garante dois IDs consistentes
    │
    ▼
Export no formato YOLO
```

O CVAT é a ferramenta principal por suportar importação direta de máscaras SAM, exportar formato YOLO nativamente e ser self-hosted via Docker. Label Studio fica como alternativa se o CVAT criar fricção.

### Protocolo do gold set

O gold set tem rigor maior por ser o único ground truth manual disponível:

- O anotador marca cada frame do clip do início ao fim, sem ver as predições do SAM 3 antes de terminar.
- Documenta o tempo gasto por frame, para reportar o custo real de anotação manual no artigo.
- Round completo a 10 fps gera 20 a 30 frames por clip. Dois clips somam 40 a 60 frames manuais.
- A cobertura forçada de uma fonte JP e uma fonte BR estratifica o gold por qualidade de vídeo.

### Armazenamento e versionamento

Os vídeos brutos e as anotações ficam em `data/`, versionados via DVC (configurado no [diário 09](09-sincronizacao-de-dados-dvc.md)) com remote no Google Drive. A estrutura segue:

```
data/raw/{jp,br}/
data/processed/clips/
data/annotations/{train,val,gold}/
```

### Splits

O gold (2 clips JP+BR) é test puro e fica intocado até a avaliação final. Dos 10 a 20 clips restantes, 80 por cento vão para train e 20 por cento para val, estratificados por fonte (JP vs BR) e por ângulo de câmera quando viável. Sem cross-validation nesta fase: dataset pequeno torna o ruído de fold maior que o sinal.

### Riscos do protocolo

| Risco | Mitigação |
|-------|-----------|
| SAM 3 falhar em vídeos amadores BR (ângulo lateral, iluminação ruim) | Pipeline dinâmica do ROI já validada no PoC ([diário 05/05](05-experimento-sam3-poc/05-pipeline-dinamico.md)). Se falhar, troca de prompt ("toy" funcionou antes) ou anotação manual desses clips |
| Revisão consumir mais tempo que o estimado | Corte para 10 a 12 clips em vez de 20, se na quarta-feira da semana 1 a revisão estiver abaixo de 50 por cento |
| Inconsistência de IDs entre Robô A e B | Convenção fixa: A é o robô mais à esquerda no frame zero. Documentada no protocolo de anotação |

## Arquitetura da pipeline

### Pipeline de inferência

A pipeline final, que é o produto do artigo, é independente do SAM 3. Roda em fps nativo do vídeo e usa redução de resolução interna apenas dentro do YOLO (parâmetro `imgsz=640`), sem decimate prévio.

```
Vídeo bruto (qualquer fps, qualquer resolução)
    │
    ▼
[1] Decodificação frame a frame via cv2.VideoCapture, mantém fps nativo
    │
    ▼
[2] Detecção do dohyo (visão clássica): threshold no canal de luminância (tawara branca),
    morfologia close+open, maior contorno externo, bounding rectangle, ellipse fit.
    Fallback: último ROI válido se o frame atual falhar.
    Código reaproveitado de experiments/sam3-poc/annotate_video.py
    │
    ▼
[3] Calibração espacial: elipse detectada mapeia pixel para centímetro,
    usando diâmetro conhecido de 154 cm do dohyo 3 kg
    │
    ▼
[4] YOLOv8s: passa o frame nativo, imgsz=640 interno, recebe coords nativas.
    Filtro geométrico: descarta detecções fora da elipse do dohyo.
    Saída: até 2 bboxes por frame.
    │
    ▼
[5] OC-SORT: recebe bboxes em coords nativas, fps nativo via Δt automático.
    Associa caixas entre frames consecutivos.
    Saída: trajetórias contínuas Robô A, Robô B.
    │
    ▼
[6] Extração de métricas:
    - Posição (x, y) em cm no referencial do dohyo
    - Velocidade (cm/s): diferença finita central com Savitzky-Golay janela 5
    - Aceleração (cm/s²): segunda derivada
    - Trajetória completa do round
    - Heatmap espacial
    │
    ▼
[7] Detecção de eventos (regras determinísticas):
    - Início do round: primeiro frame onde min(|v_A|, |v_B|) > v_min_inicio
    - Primeiro contato (cinemático): primeiro frame t onde
        |Δv_A(t)| > Δv_threshold AND |Δv_B(t)| > Δv_threshold
        AND distância(centro_A, centro_B) < d_threshold
      Thresholds calibrados no gold set, com fallback de valores padrão da literatura.
    - Ring-out: centro de qualquer bbox sai da elipse do dohyo
    - Fim do round: ring-out detectado OU 3 segundos sem movimento
    - Vencedor: robô que NÃO sofreu ring-out (ou determinação manual se timeout)
    │
    ▼
Saída dupla:
    - JSON estruturado com trajetórias, métricas e eventos timestamped em milissegundos
    - Vídeo de saída em resolução e fps nativos, com overlay de bbox e eventos
```

### Pipeline de treino

Separada da inferência. Consome apenas o dataset SAM-anotado e revisado, gera os pesos que alimentam o estágio [4] da inferência.

```
Clips brutos (10 a 20)
    │
    ▼
SAM 3 video predictor (prompt + ROI + filtros do PoC)
    │
    ▼
Máscaras por frame → bbox via boundingRect do maior contorno
    │
    ▼
Conversão para YOLO format (class 0 = robot, normalized xywh)
    │
    ▼
Importação CVAT, revisão manual
    │
    ▼
Dataset YAML + splits 80/20
    │
    ▼
ultralytics: yolo train data=kanshigan.yaml model=yolov8s.pt epochs=100 imgsz=640
    │
    ▼
Pesos treinados, congelados para inferência
```

### Por que não decimate na inferência final

O `ffmpeg decimate` foi necessário no SAM 3 PoC por causa do gargalo de VRAM (feature maps fixos por frame). YOLO e OC-SORT não têm esse limite. Cada frame é independente no YOLO, e OC-SORT adapta o modelo de movimento ao Δt entre frames automaticamente.

Rodar em fps nativo melhora o experimento em múltiplas frentes:

| Por que fps nativo ajuda |
|--------------------------|
| Eventos sub-segundo ganham resolução temporal real. A 10 fps, o primeiro contato tem incerteza de ±100 ms; a 60 fps cai para ±16 ms. |
| Detecção cinemática de contato fica menos ruidosa, picos de desaceleração ficam mais nítidos. |
| Velocidade e aceleração calculadas ficam menos suavizadas artificialmente pela amostragem. |
| Output visual fica fluido, importante para figuras do artigo. |

### Proxy de resolução para output

Adobe Premiere chama de "proxies". O equivalente em CV é trivial: roda inferência em resolução reduzida, escala as bboxes de volta. A Ultralytics já faz isso internamente via `imgsz`. Passa o frame nativo, recebe coords nativas, desenha em cima do vídeo em resolução original.

### Tecnologias

| Componente | Stack |
|------------|-------|
| Linguagem | Python 3.12 |
| Gerenciador | uv (já usado no PoC) |
| ML framework | PyTorch + Ultralytics YOLO |
| Tracking | boxmot (implementação canônica de OC-SORT, Apache 2.0) |
| Vídeo | OpenCV, ffmpeg |
| Anotação | CVAT self-hosted (Docker) |
| Versionamento de dados | DVC com remote Google Drive |
| Tracking de experimentos | Weights & Biases free tier, com JSON local como fallback |

### Estrutura de diretórios

```
Kanshigan/
├── data/                    (DVC, fora do git)
├── docs/                    (existente)
├── experiments/
│   ├── sam3-poc/           (existente, vira referência)
│   └── pre-banca/          (novo)
│       ├── annotate.py     SAM 3 para pré-anotações CVAT
│       ├── train.py        YOLO training
│       ├── infer.py        Pipeline completa de inferência
│       ├── evaluate.py     Métricas contra gold set
│       └── metrics.py      Extração de posição, velocidade, eventos
├── notebooks/              (existente)
└── results/                (figuras, tabelas, logs)
```

### Reuso de código existente

- `detect_dohyo()` do `annotate_video.py` é o estágio [2] da inferência sem alteração
- `masks_to_full_frame()` e `filter_closest_to_center()` são reaproveitados no pipeline de anotação
- A detecção dinâmica de ROI por frame é o estágio [2] do pipeline de anotação

## Status

- Dataset e protocolo de anotação definidos.
- Pipeline de inferência e pipeline de treino arquitetadas.
- Tecnologias e estrutura de diretórios escolhidas.
- O próximo passo é o protocolo experimental e o cronograma, documentados no [diário 13](13-protocolo-experimental-e-cronograma.md).
