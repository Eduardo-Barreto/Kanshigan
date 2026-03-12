# Framework de Evidências

Pra responder a pergunta de pesquisa, preciso produzir evidências concretas e mensuráveis. Não basta "rodar o modelo". Cada evidência precisa dizer o que vai ser medido.

## Mapeamento da pergunta para evidências

| Categoria | O que medir | Métricas |
|-----------|------------|----------|
| Acurácia de detecção | O modelo acha os robôs nos frames? | mAP@0.5, mAP@0.5:0.95 |
| Acurácia de tracking | Ele mantém identidade entre frames? | MOTA, IDF1, ID Switches |
| Acurácia de eventos | Classifica corretamente os eventos da partida? | Precision/Recall por tipo de evento, erro temporal em frames |
| Acurácia de métricas | As métricas extraídas são precisas? | Erro de posição em cm |
| Viabilidade prática | Roda em tempo real? | FPS por abordagem, trade-off acurácia vs velocidade |

## Ideias de abordagens a comparar

- YOLOv8 + ByteTrack
- YOLOv8 + BoT-SORT
- YOLOv11 + ByteTrack
- YOLOv11 + BoT-SORT
- RT-DETR + ByteTrack

Cada combinação será avaliada nas mesmas métricas, no mesmo dataset.
