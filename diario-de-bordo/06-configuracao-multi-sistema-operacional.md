# Configuração Multi-Sistema-Operacional

## Contexto

O desenvolvimento do projeto acontece em dois ambientes diferentes: a máquina de desenvolvimento local (macOS, Apple Silicon) e o servidor de inferência (Linux, NVIDIA GPU). O código dos experimentos foi originalmente escrito e testado só no Linux. A ideia foi migrar os scripts pra notebooks documentados no repositório, o que exigiu fazer o ambiente funcionar no macOS também.

## Problema 1: triton não existe no macOS

O `triton` é uma dependência transitiva do SAM 3 — usado internamente pra compilar kernels CUDA. No macOS não tem wheel disponível, então `uv add triton` falha:

```
error: Distribution `triton==3.6.0` can't be installed because it doesn't have a source
distribution or wheel for the current platform (macosx_15_0_arm64)
```

**Solução:** adicionar `triton` como dependência condicional no `pyproject.toml`, restrita a Linux:

```toml
"triton; sys_platform == 'linux'",
```

O uv resolve o lock file pra todas as plataformas, mas só instala no Linux. No macOS, ignora.

## Problema 2: sam3/model/edt.py importa triton diretamente

Mesmo com o triton fora do ambiente macOS, o SAM 3 tenta importar triton em tempo de execução no `edt.py`. O módulo implementa um kernel Euclidean Distance Transform (EDT) usando Triton, sem fallback.

**Solução:** modificar o `edt.py` pra envolver o import em `try/except` e adicionar um fallback CPU usando `scipy`:

```python
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    HAS_TRITON = False

def edt_triton(data: torch.Tensor):
    if not HAS_TRITON or not data.is_cuda:
        return edt_cpu_fallback(data)
    # ... implementação original com triton
```

O fallback usa `scipy.ndimage.distance_transform_edt`, que produz o mesmo resultado, só que na CPU.

## Fork do SAM 3

Aplicar esse patch manualmente no arquivo instalado pelo uv não é sustentável — qualquer `uv sync` sobrescreve. A solução correta é manter um fork do repositório com o patch aplicado.

Fork criado em `pedrocruz2/sam3-macos-patch`. As mudanças aplicadas:
- `sam3/model/edt.py`: import condicional do triton + fallback scipy
- `pyproject.toml` do fork: `scipy` adicionado como dependência

O `pyproject.toml` do projeto agora aponta pra esse fork:

```toml
[tool.uv.sources]
sam3 = { git = "https://github.com/pedrocruz2/sam3-macos-patch.git" }
```

Qualquer máquina que rodar `uv sync` pega automaticamente o sam3 com compatibilidade macOS, sem passos manuais.

## Problema 3: sam3.py usa CUDA diretamente

O script de anotação de vídeo passava o device de forma hardcoded:

```python
gpus_to_use = [torch.cuda.current_device()]  # quebra no macOS
```

**Solução:** condicionar ao sistema operacional:

```python
import sys
gpus_to_use = [torch.cuda.current_device()] if sys.platform == "linux" and torch.cuda.is_available() else []
```

No macOS, passa lista vazia e o predictor usa CPU. No Linux com GPU, comportamento original.

## Por que não MPS?

O Apple Silicon tem suporte a GPU via MPS (Metal Performance Shaders) no PyTorch. Seria natural tentar usar MPS no macOS em vez de CPU. O problema é que o SAM 3 tem acoplamento profundo com CUDA em todo o código base:

- ~58 referências a `torch.cuda.*` espalhadas em 10+ arquivos
- `.cuda()` hardcoded em tensores em `io_utils.py`, `sam3_tracker_base.py`, e outros
- `@torch.autocast(device_type="cuda")` em decoradores de métodos
- `sam3_multiplex_base.py` chama `torch.cuda.get_device_properties(0)` em nível de módulo — crasha no import antes de qualquer código rodar
- Infraestrutura de multi-GPU com NCCL, que é exclusiva de CUDApra te

Adicionar suporte a MPS eventualmente será necessário, porém devido a complexidade ainda não foi feito.

## Resultado

O ambiente agora funciona nos dois sistemas:

| | macOS (Apple Silicon) | Linux (NVIDIA GPU) |
|---|---|---|
| triton | não instalado (marker) | instalado normalmente |
| edt.py | fallback scipy (CPU) | kernel triton (GPU) |
| inferência SAM 3 | CPU (lenta, pra desenvolvimento) | CUDA (produção) |
| uv sync | funciona | funciona |

O fluxo de trabalho é: desenvolvimento e edição de notebooks no macOS, inferência real rodando no Linux. (Até Adicionar o MPS ao SAM3)
