# LocateAnything da NVIDIA

Investigação do [LocateAnything](https://research.nvidia.com/labs/lpr/locate-anything/), framework de visual grounding open-vocabulary do laboratório LPR da NVIDIA (paper no arXiv [2605.27365](https://arxiv.org/abs/2605.27365), 26/05/2026), avaliado como candidato a detector para o pipeline do Kanshigan.

**Veredito:** detector/grounder generativo de ~3.4B params, sem tracking nativo, sem suporte a macOS/MPS e inviável na RTX 4070 8GB — **OOM medido na própria máquina** (`torch.OutOfMemoryError`, ver [arquivo 02](02-viabilidade-na-4070-e-no-macos.md)). Cai nas mesmas armadilhas estruturais do [SAM 3](../08-sam3-nao-e-viavel-como-pipeline-final.md) e ainda perde o tracking. Fica como opção de anotação (atrás do SAM 3), não como pipeline final.

## Arquivos

1. [O que é o LocateAnything](01-o-que-e-o-locate-anything.md) — task, linhagem (Grounding DINO / NVIDIA LPR), Parallel Box Decoding, arquitetura, números e benchmarks.
2. [Viabilidade na RTX 4070 8GB e no macOS](02-viabilidade-na-4070-e-no-macos.md) — os gargalos: VRAM, BPS ≠ FPS, ausência de tracking, sem MPS, licença e treino só em cluster.
3. [Comparação e decisão](03-comparacao-e-decisao.md) — tabela-síntese vs SAM 3 / Grounding DINO / YOLO-World / YOLOv8-v11 / RT-DETR, decisão e próximos passos.
