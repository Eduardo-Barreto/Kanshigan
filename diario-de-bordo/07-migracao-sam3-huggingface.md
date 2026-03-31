# Migrando do SAM 3 oficial pro HuggingFace Transformers

O experimento inicial com o SAM 3 ([diario-de-bordo/05](https://github.com/Eduardo-Barreto/Kanshigan/tree/main/diario-de-bordo/05-experimento-sam3-poc)) validou que o modelo funciona pro nosso domínio: detecta robôs de sumô via prompt de texto, mantém identidade no tracking temporal, e roda na RTX 4070 8GB. O problema veio quando tentamos rodar o mesmo código no macOS pra desenvolvimento local. O repositório oficial (`facebookresearch/sam3`) tem CUDA acoplado em todo o codebase, e o [documento de compatibilidade](https://github.com/Eduardo-Barreto/Kanshigan/blob/main/diario-de-bordo/06-configuracao-multi-sistema-operacional.md) detalha os workarounds que foram necessários: fork com patches no `edt.py`, stub pro decord, device detection manual.

Os workarounds funcionam. Mas pesquisando como a comunidade lida com esse problema, encontrei que o HuggingFace Transformers já tem uma [reimplementação device-agnostic do SAM 3](https://huggingface.co/docs/transformers/model_doc/sam3) que resolve tudo de uma vez. Vale migrar.

## Por que migrar

O `transformers` expõe `Sam3Model` e `Sam3Processor` sem dependência de triton, sem decord, sem `.cuda()` hardcoded. Funciona em CUDA, MPS e CPU. Um engenheiro da Meta [recomendou esse caminho](https://huggingface.co/facebook/sam3/discussions/11) pra quem precisa rodar em Apple Silicon, e outros usuários confirmaram que funciona.

```python
from transformers import Sam3Processor, Sam3Model
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")
```

Isso eliminaria três workarounds de uma vez:

- O fork `pedrocruz2/sam3-macos-patch` (patches manuais em 9 arquivos do SAM 3)
- O `local-packages/decord` stub que redireciona pro `eva-decord`
- O import condicional do triton no `edt.py` com fallback scipy

Tem um caveat pra video inference: o `processing_sam3_video.py` tem um bug onde `.pin_memory()` causa device mismatch no MPS. O fix é remover o `.pin_memory()`:

```python
# antes (quebra no MPS)
keep_idx_gpu = keep_idx.pin_memory().to(device=out_binary_masks.device, non_blocking=True)

# depois
keep_idx_gpu = keep_idx.to(device=out_binary_masks.device, non_blocking=True)
```

## O custo de manter o fork

O fork funciona agora, mas tem um custo de manutenção real. O SAM 3.1 (Object Multiplex) [saiu dia 27 de março](https://ai.meta.com/blog/segment-anything-model-3/), e cada release upstream vai exigir que alguém reaplique os patches. O [Sompote/SAM3_CPU](https://github.com/Sompote/SAM3_CPU/) é outro fork que tentou resolver o mesmo problema e tem 1 star, o destino comum de forks de compatibilidade de device. O HuggingFace acompanha releases upstream automaticamente, e o time de manutenção é dedicado.

A Meta também [sinalizou](https://github.com/facebookresearch/sam3/issues/164) que aceita PRs de device-agnostic inference pro image predictor (o [PR #173](https://github.com/facebookresearch/sam3/pull/173) está aberto). Mas o video predictor eles disseram que é "highly optimized for the multi-gpu setting" e não têm planos de suportar outros devices. O caminho upstream é parcial, o que reforça a vantagem do HuggingFace.

## Device detection

Independente de migrar ou não pro HuggingFace, o device detection nos scripts precisa mudar. O código atual:

```python
import sys
gpus_to_use = [torch.cuda.current_device()] if sys.platform == "linux" and torch.cuda.is_available() else []
```

O `sys.platform == "linux"` é redundante. O `torch.cuda.is_available()` já verifica internamente se o PyTorch foi compilado com CUDA, se os drivers estão presentes, e se existe pelo menos um device visível, em qualquer plataforma. Nenhuma lib do ecossistema usa platform check pra isso: nem o [Ultralytics](https://docs.ultralytics.com/reference/utils/torch_utils/), nem o [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/usage_guides/mps), nem o [Lightning](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html). O check também ignora MPS completamente, que é justamente o device que queremos usar no macOS.

O padrão da comunidade:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

Desde o PyTorch 2.4, o [`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html) faz isso automaticamente:

```python
device = torch.accelerator.current_accelerator(check_available=True)
if device is None:
    device = torch.device("cpu")
```

## Sobre o decord

O `decord` é uma dependência de bagagem do SAM 3 oficial. O pacote lista `decord` nos extras `[notebooks]`, e como instalamos `sam3[notebooks,train]`, ele vem junto. O stub `local-packages/decord` que redireciona pro `eva-decord` é um workaround necessário porque o [decord no PyPI](https://pypi.org/project/decord/) não lança release desde 2022 e não tem wheels pra Apple Silicon.

Se migrarmos pro HuggingFace Transformers, o decord sai da equação (o HuggingFace não depende dele). Se por algum motivo precisarmos de video decoding separado no futuro, o [TorchCodec](https://github.com/meta-pytorch/torchcodec) (v0.11.0, projeto oficial do PyTorch, macOS arm64, GPU decode via NVDEC) é pra onde o ecossistema convergiu. O [torchvision deprecated a API de vídeo](https://docs.pytorch.org/vision/0.22/io.html) em favor dele, e o HuggingFace datasets já migrou.

## O que está certo no documento original

O marker do triton no `pyproject.toml` (`sys_platform == 'linux'`) é exatamente o que o [PyTorch faz](https://pypi.org/project/torch/). É o mecanismo padrão do PEP 508, a solução correta.

O diagnóstico dos problemas no documento 06 também está preciso: o mapeamento das ~58 referências CUDA no SAM 3, a análise de por que MPS não funciona direto, e a tabela comparativa dos dois ambientes são documentação útil que nos ajudou a entender o escopo do problema.

## Próximos passos

1. Migrar os scripts de experimento pra usar `transformers.Sam3Model` e validar que o tracking temporal continua funcionando com a API do HuggingFace.

2. Centralizar device detection num `resolve_device()` usando o padrão `cuda > mps > cpu`, sem platform check.

3. Remover o fork `pedrocruz2/sam3-macos-patch` e o stub `local-packages/decord` das dependências depois de validar a migração.

4. Se precisarmos de video decoding separado do SAM 3 no pipeline final, avaliar TorchCodec.
