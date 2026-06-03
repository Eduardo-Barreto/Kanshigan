# locate-anything-test

Teste mínimo do `nvidia/LocateAnything-3B` como detector open-vocab para Sumô de Robôs, pra comparar com o pipeline de prompt textual do SAM 3 (`experiments/sam3-poc`).

Conclusão documentada em [diário 09](../../diario-de-bordo/09-locate-anything-nvidia/02-viabilidade-na-4070-e-no-macos.md): **não roda na RTX 4070 Laptop 8GB**, OOM medido (não estimado).

## Setup

```bash
uv sync
uv run python detect.py frame_000.jpg
```

Requer Linux + GPU NVIDIA (o model card não suporta macOS/MPS). Os pesos BF16 (7.2GB) baixam do HuggingFace na primeira execução.

## Resultado medido

GPU: RTX 4070 Laptop (8188 MiB totais, 7.63 GiB utilizáveis), driver 590.48.01.

O modelo **carregou**, mas estourou no forward do vision encoder com uma imagem 848×478:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 306.00 MiB.
GPU 0 has a total capacity of 7.63 GiB of which 134.62 MiB is free.
Including non-PyTorch memory, this process has 7.48 GiB memory in use.
```

Com pesos de 7.2GB carregados na placa de 7.63 GiB úteis, a alocação de 306 MiB durante o forward já não cabe. Sem quantização oficial (TensorRT/Triton ainda não suportados), não há caminho fácil pra caber em 8GB.

Contraste: o SAM 3 roda nessa mesma máquina, só estoura depois de ~100 frames.
