# Comparação e decisão

Fechando a pasta sobre o [LocateAnything](01-o-que-e-o-locate-anything.md). Os [gargalos de viabilidade](02-viabilidade-na-4070-e-no-macos.md) já apontam pra onde isso vai, mas vale colocar lado a lado com as outras opções que estão no radar do projeto, no mesmo espírito do [framework de evidências](../04-framework-de-evidencias.md).

## Tabela-síntese

| Critério | LocateAnything-3B | SAM 3 | Grounding DINO | YOLO-World | YOLOv8/v11 (fine-tuned) | RT-DETR |
|----------|-------------------|-------|----------------|------------|--------------------------|---------|
| **Tarefa** | Detecção/grounding open-vocab (boxes/pontos) | Segmentação + detecção + tracking por conceito | Open-set detection | Open-vocab detection real-time | Detecção closed-set | Detecção closed-set end-to-end |
| **Tracking nativo** | Não | Sim (vídeo) | Não | Não | Não (usa ByteTrack/BoT-SORT) | Não (usa tracker) |
| **Prompt textual** | Sim (frágil) | Sim (frágil) | Sim | Sim | Não (classes fixas) | Não |
| **Params** | ~3.4B (7.2GB BF16) | encoder pesado | ~172M (Swin-T) | leve (base YOLOv8) | 3–68M (n→x) | ~32–67M (L→X) |
| **Acurácia (COCO)** | 54.7 F1 zero-shot | n/a (seg) | 56.6 zero-shot | ~35–39 zero-shot | >55 mAP **fine-tuned** | 53–55 AP |
| **Velocidade** | 12.7 BPS @ H100 (≈6 FPS p/ 2 obj) | 8–12 FPS @ H100 1080p | baixo FPS | ~74 FPS | 50–100+ FPS | 74–124 FPS (T4) |
| **Roda em 8GB?** | Não (OOM medido na 4070) | Não (OOM >100 frames, testado) | Apertado | Sim | Sim | Sim |
| **macOS/MPS** | Não documentado | Parcial via HuggingFace | Parcial | Sim | Sim (`device=mps`) | Parcial/Sim |
| **Treina domínio pequeno** | Não (cluster 8-GPU) | Pesado | Médio | Médio | Sim (horas em laptop) | Sim (mais pesado que YOLO) |
| **Licença** | NVIDIA (não-comercial) | Meta (ver termos) | Apache-2.0 | AGPL/Ultralytics | AGPL-3.0 (Ultralytics) | Apache-2.0 |
| **Papel no Kanshigan** | Anotação (opcional) | Anotação (já decidido) | Anotação alternativa | Anotação/baseline | **Pipeline final (líder)** | **Alternativa transformer** |

Notas: a tabela mistura números de **hardwares e unidades diferentes** (BPS em H100, FPS em T4/GPU, AP/F1 em datasets distintos) — leia como **ordem de grandeza**, não comparação direta. **BPS (boxes/s) não é FPS (frames/s).** Os FPS de YOLO/RT-DETR variam bastante na 4070 Laptop (de 1–2 FPS sem TensorRT por subutilização até 100+ FPS com modelos pequenos e engine otimizada). Já o "não roda em 8GB" do LocateAnything **não é estimativa: é OOM medido na própria 4070 Laptop** (`torch.OutOfMemoryError`, ver [arquivo 02](02-viabilidade-na-4070-e-no-macos.md)). Os params são contagem (~3.4B), e o `7.2GB` é o tamanho dos pesos em BF16 medido em disco.

## Decisão

**LocateAnything não entra no caminho crítico da entrega.** Pelos mesmos motivos estruturais do SAM 3, e com um agravante: ele nem tem tracking, então seria sempre "detector caro + tracker externo", que é exatamente a arquitetura pesada que o projeto está tentando evitar.

O papel **possível** dele é o mesmo do SAM 3: **ferramenta de anotação / pseudo-labeling**, com uma vantagem específica — entrega bounding box direto via prompt de texto, sem o passo máscara → box. Mas como já tenho o SAM 3 ocupando esse papel, e como o LocateAnything traz a mesma fragilidade de prompt, mais licença não-comercial e zero suporte a macOS, **ele não substitui nem complementa o SAM 3 de forma que justifique adotá-lo agora.**

Resumindo o veredito em uma frase: é um modelo interessante e tecnicamente elegante (o Parallel Box Decoding é genuinamente bom), mas resolve um problema que não é o nosso. O nosso é 2 robôs, 8GB, prazo curto, e precisa de tracking.

## Onde isso me deixa

A investigação reforça — em vez de mudar — a direção que já estava definida nos arquivos [08](../08-sam3-nao-e-viavel-como-pipeline-final.md) e [04](../04-framework-de-evidencias.md):

1. **Pipeline final:** YOLOv8/v11 + ByteTrack como baseline, YOLOv8/v11 + BoT-SORT como comparação de tracking, RT-DETR + ByteTrack como alternativa transformer.
2. **Anotação semiautomática:** SAM 3 continua como ferramenta principal de pseudo-label. O LocateAnything fica anotado aqui como opção avaliada e descartada por enquanto — se em algum momento o SAM 3 não der conta de gerar boxes limpas, o LocateAnything é o segundo nome da fila pra anotação (não pra inferência).
3. **Visão clássica:** a ROI dinâmica do dohyo segue sendo peça da pipeline, independente do detector.

A conclusão metodológica é a mesma do SAM 3: foundation models grandes e generativos são úteis pra explorar e anotar, mas a pergunta de pesquisa do Kanshigan é sobre **acurácia e viabilidade prática**, e viabilidade prática numa 4070 Laptop 8GB não combina com um VLM de 3B sem tracking.

## Referências

- [Página oficial NVIDIA LPR — LocateAnything](https://research.nvidia.com/labs/lpr/locate-anything/): ground truth do projeto, throughput e benchmarks resumidos.
- [Paper no arXiv (2605.27365)](https://arxiv.org/abs/2605.27365): "LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding", submetido em 26/05/2026. Detalha o PBD/MTP e os ablations.
- [Model card `nvidia/LocateAnything-3B` (HuggingFace)](https://huggingface.co/nvidia/LocateAnything-3B): fonte mais densa pra uso prático — licença, base models, dependências, código do `LocateAnythingWorker`, templates de prompt e hardware suportado.
- [Repo NVlabs/Eagle — `Embodied`](https://github.com/NVlabs/Eagle/tree/main/Embodied): código de treino/inferência/avaliação e o comando de fine-tuning DeepSpeed.
- [HF Space — demo](https://huggingface.co/spaces/nvidia/LocateAnything): testar fragilidade de prompt no domínio sumô sem instalar nada.
- [YOLO-World (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.pdf): o concorrente real de "open-vocab + rápido" — ~74 FPS, ~20× mais rápido que o Grounding DINO.
- [RT-DETR (arXiv 2304.08069)](https://arxiv.org/abs/2304.08069): base da alternativa transformer do projeto.
- [Treinar YOLO em Apple Silicon (Ultralytics)](https://docs.ultralytics.com/modes/train): confirma `device="mps"` oficial, o que o LocateAnything não tem.

> Material de pesquisa completo (raw, 6 seções) em `barreto.sh/research/research_locate-anything-nvidia.md`.
