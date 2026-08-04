# O que é o LocateAnything da NVIDIA

## Por que fui olhar isso

Logo depois de fechar a decisão de que [SAM 3 não é viável como pipeline final](../08-sam3-nao-e-viavel-como-pipeline-final.md), apareceu o **LocateAnything**, do laboratório LPR (Learning and Perception Research) da NVIDIA. A página oficial vende exatamente o que parece resolver o nosso problema: localizar qualquer coisa numa imagem a partir de um prompt de texto, rápido. O paper saiu no arXiv dia **26 de maio de 2026** — três dias atrás. Valia investigar antes de assumir que YOLO + tracker é o único caminho.

A pergunta que importa para o Kanshigan é a mesma que fiz pro SAM 3: **isso serve como pipeline final de detecção e tracking, como ferramenta de anotação, ou cai nos mesmos gargalos?**

A resposta curta, que o resto desta pasta justifica: é mais um detector caro. Cai nas mesmas armadilhas estruturais do SAM 3 para o pipeline final, mas tem um perfil diferente o suficiente pra merecer ser documentado.

## O que ele é, exatamente

LocateAnything é um framework **generativo de visual grounding e object detection open-vocabulary**. A entrada é uma imagem mais um prompt textual; a saída são **bounding boxes** (e opcionalmente **pontos**) com labels.

Três coisas que ele **não** é, e que importam pra nós:

- não é segmentação (não produz máscaras como o SAM);
- não é 3D;
- **não é tracking de vídeo.** É detecção frame a frame, sem memória temporal.

Esse último ponto já muda tudo. O SAM 3 pelo menos tinha um video predictor que propagava identidade entre frames. O LocateAnything não tem nada disso. Para vídeo de sumô, eu precisaria plugar ByteTrack ou BoT-SORT por cima de qualquer jeito. Ou seja: ele não compete com o SAM 3 como "segmentador + tracker", ele compete com YOLO e RT-DETR como **detector** — só que pesando 3 bilhões de parâmetros.

## De onde ele vem

Vale notar a linhagem, porque explica o que esperar. Entre os autores estão **Shilong Liu** e **Lei Zhang**, que são parte do time original do **Grounding DINO** e do **DINO-X**, mais **Jan Kautz** e **Andrew Tao** da NVIDIA. As afiliações são HK PolyU, Princeton, Nanjing, UIUC e NVIDIA LPR.

Em espírito, LocateAnything é a versão "VLM generativo com decoding paralelo" da família Grounding DINO. Isso ajuda a calibrar expectativa: é detecção open-set de gente que entende muito do assunto, mas embalada num modelo de linguagem grande.

## A ideia central: Parallel Box Decoding

A contribuição técnica do paper é o **Parallel Box Decoding (PBD)**, e ela é genuinamente elegante.

O problema que ele ataca: VLMs que fazem grounding (Qwen-VL, Rex-Omni e afins) tratam coordenadas como **tokens de texto gerados em sequência**. Para uma caixa, o modelo emite `x1 → y1 → x2 → y2`, um token de cada vez, autoregressivamente. Isso dá dois problemas:

1. **Lentidão que escala com objetos × tokens.** Cena com 300 objetos é catastroficamente lenta.
2. **Incoerência geométrica.** Serializar um retângulo 2D numa sequência 1D quebra a relação entre os cantos — `x2` é gerado como se fosse a próxima palavra, não o canto do mesmo retângulo.

O PBD decodifica **cada caixa como uma unidade atômica, num passo só**, via Multi-Token Prediction (MTP). Preserva a coerência interna da caixa e destrava paralelismo.

A arquitetura por baixo é um VLM clássico:

- **vision encoder:** MoonViT-SO-400M (~400M params, licença MIT);
- **projector:** MLP;
- **language decoder:** Qwen2.5-3B;
- **coordenadas** normalizadas pro intervalo `[0, 1000]`, no formato `<ref>label</ref><box><x1><y1><x2><y2></box>`.

O modelo é treinado com dois streams ao mesmo tempo: NTP (next-token, causal, robusto e lento) e MTP (blocos com atenção bidirecional interna, paralelo e rápido). Na inferência há três modos:

| Modo | Mecanismo | Velocidade (H100) |
|------|-----------|-------------------|
| `slow` | NTP puro (autoregressivo) | ~4.3 BPS |
| `fast` | MTP puro (paralelo) | ~15.3 BPS |
| `hybrid` (default) | MTP com fallback pra AR em caixas incertas | **12.7 BPS** |

A unidade aqui é **BPS — boxes per second**, não FPS. Volto nisso no [próximo arquivo](02-viabilidade-na-4070-e-no-macos.md), porque é uma distinção que muda completamente a leitura de viabilidade.

## Os números

- **~3.4B de parâmetros no total:** Qwen2.5-3B (decoder, 3B) + MoonViT-SO-400M (encoder, ~400M). A metadata do model card arredonda isso pra "4B params" — é contagem de parâmetros, não tamanho de arquivo. Os pesos BF16 ocupam **7.2GB em disco**, que já não cabem na 4070 Laptop 8GB ([próximo arquivo](02-viabilidade-na-4070-e-no-macos.md): OOM medido).
- Treinado em **138M amostras**, **12M imagens únicas**, **785M+ bounding boxes**. A composição é dominada por detecção geral (66.9% das queries), mas tem bastante GUI grounding (16.5%), referring, OCR e layout de documentos.
- A rotulagem foi híbrida, usando inclusive **SAM 3, Qwen3-VL, Molmo e Rex-Omni** como anotadores automáticos. (Detalhe simpático: o SAM 3 que descartei como pipeline final apareceu como ferramenta de anotação no treino deste modelo. É exatamente o papel que reservei pra ele.)

### Benchmarks reportados

| Dataset | Métrica | LocateAnything-3B | Comparação |
|---------|---------|-------------------|------------|
| COCO | F1@mIoU | 54.7 | Grounding DINO 56.6; DINO-Swin-L (closed-set) 62.1; Rex-Omni 52.9 |
| LVIS | F1@mIoU | 50.7 | Rex-Omni 46.9 |
| VisDrone (denso) | F1@mIoU | 39.9 | Rex-Omni 35.8 |
| DocLayNet | F1@mIoU | 76.8 | Rex-Omni 70.7 |
| ScreenSpot-Pro (GUI) | Avg F1 | 60.3 (SOTA) | Rex-Omni 36.8 |
| RefCOCOg | F1@mIoU | 77.6 | Rex-Omni 74.3 |

A leitura crítica para o nosso domínio: em **detecção genérica zero-shot** (COCO/LVIS), o LocateAnything fica **abaixo do Grounding DINO** e **bem abaixo de um detector closed-set fine-tuned** (DINO-Swin-L 62.1). Onde ele brilha é GUI, documentos e OCR — domínios que não têm nada a ver com vídeo de sumô.

Ou seja: para "2 robôs numa arena metálica", um YOLO ou RT-DETR **fine-tuned no nosso dataset** vai superar qualquer um desses números zero-shot com folga, rodando a uma fração do custo. O "Anything" do nome se sustenta na **amplitude de tarefas**, não em bater especialistas em cada uma delas.
