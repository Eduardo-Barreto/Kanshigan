# Experimento SAM 3 — HuggingFace Transformers

A entrada anterior ([07-migracao-sam3-huggingface](07-migracao-sam3-huggingface.md)) documentou por que migrar do repositório oficial do SAM 3 pro HuggingFace Transformers. Esta entrada documenta a execução: o notebook que reimplementa os quatro experimentos da PoC original ([05-experimento-sam3-poc](05-experimento-sam3-poc/)) usando `transformers.Sam3Model` e `transformers.Sam3VideoModel`.

Notebook: `notebooks/sam3_agnostic_experiments.py` (marimo, não Jupyter).

## Por que marimo

O notebook original era um script Python que rodava linearmente. A versão migrada usa [marimo](https://marimo.io/) porque cada seção do experimento é independente — não faz sentido ter que rodar o teste de prompts inteiro pra chegar no pipeline avançado. O marimo trata cada célula como um DAG reativo: só roda o que tem dependência direta. Além disso, os controles de UI (campos de texto, botões, sliders) permitem iterar nos parâmetros sem editar código.

O arquivo `.py` que o marimo gera é um script Python válido. Dá pra abrir no marimo (`marimo edit`) ou rodar direto (`python sam3_agnostic_experiments.py`).

## Estrutura do notebook

O notebook tem quatro seções de experimento, cada uma com botão de execução independente:

| # | Seção | Modelo | O que faz |
|---|-------|--------|-----------|
| 1 | Teste de prompts | `Sam3Model` (imagem) | Compara descrições de texto no frame 0 |
| 2 | Vídeo baseline | `Sam3VideoModel` | Inferência direta no vídeo completo, sem pré-processamento |
| 3 | Vídeo avançado | `Sam3VideoModel` | Detecta dohyô → recorta → infere → remapeia |
| 4 | Por frame | `Sam3Model` (imagem) | Inferência independente em cada frame, sem propagação temporal |

As seções 1 e 4 usam o modelo de imagem. As seções 2 e 3 usam o modelo de vídeo. A seção 4 existe como controle experimental: sem propagação temporal, pra quantificar o ganho real do tracking.

## O que mudou em relação ao experimento original

### Imports

Antes (repositório oficial):

```python
from sam3.build_sam3 import build_sam3_image_model, build_sam3_video_predictor
```

Depois (HuggingFace):

```python
from transformers import Sam3Processor, Sam3Model, Sam3VideoModel
```

Três imports em vez de dois. O `Sam3Processor` faz o tokenize do texto e o pré-processamento das imagens. No repositório oficial isso era tudo interno.

### Device detection

Antes:

```python
import sys
gpus_to_use = [torch.cuda.current_device()] if sys.platform == "linux" and torch.cuda.is_available() else []
```

Depois:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

Sem platform check. O `torch.cuda.is_available()` já verifica tudo internamente. E agora MPS entra como segundo candidato, que era justamente o device ignorado na versão anterior.

### Inferência de imagem

Antes, o repositório oficial tinha uma API interna que recebia imagem + prompt e retornava máscaras direto. No HuggingFace, a inferência segue o padrão processor → model:

```python
processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").to(device)

inputs = processor(images=pil_image, text=["object on metal platform"], return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = outputs.pred_masks[0].cpu().numpy()   # (num_masks, H, W)
scores = outputs.pred_iou_scores[0].cpu().numpy()
```

Mais verboso, mas o padrão é o mesmo de qualquer modelo HuggingFace. Quem já usou BERT, CLIP ou Whisper reconhece o fluxo.

### Inferência de vídeo

A diferença principal. O `Sam3VideoModel` do HuggingFace recebe uma lista de frames PIL + prompt de texto e retorna máscaras pra todos os frames de uma vez:

```python
model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device)
inputs = processor(videos=pil_frames, text=["toy"], return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

all_masks = outputs.pred_masks.cpu().numpy()  # (num_frames, num_objects, H, W)
```

No repositório oficial, o fluxo era: inicializar estado → condicionar no frame 0 → propagar frame a frame num loop. O HuggingFace abstraiu isso pra uma chamada única. A propagação temporal continua acontecendo internamente, mas o usuário não precisa gerenciar o loop.

### Funções auxiliares

As funções de domínio (`detect_dohyo`, `preprocess_video`, `masks_to_full_frame`, `filter_closest_to_center`, `overlay_masks`) continuam idênticas. São OpenCV puro, não dependem do SAM 3. A lógica de detectar o dohyô por threshold no branco, cropar, e remapear as máscaras pro frame original não muda com a troca de backend.

## Autenticação

Os pesos do SAM 3 continuam gated no HuggingFace. O notebook carrega o token de um `.env` local:

```python
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv(Path(__file__).parent / ".env")
login(token=os.environ["HF_KEY"])
```

O `.env` não está no repositório (está no `.gitignore`). Cada desenvolvedor precisa do seu token.

## Sobre o bug do pin_memory no MPS

O notebook documenta em markdown que o `processing_sam3_video.py` do HuggingFace tem um bug onde `.pin_memory()` causa device mismatch no MPS. O `pin_memory()` é uma operação CUDA-only que copia o tensor pra memória paginada do host pra acelerar transferências CPU→GPU. No MPS não existe equivalente, e o PyTorch levanta erro.

O fix é remover o `.pin_memory()`:

```python
# antes (quebra no MPS)
keep_idx_gpu = keep_idx.pin_memory().to(device=out_binary_masks.device, non_blocking=True)

# depois
keep_idx_gpu = keep_idx.to(device=out_binary_masks.device, non_blocking=True)
```

Isso já estava documentado na entrada 07. O notebook só referencia.

## O que o notebook não faz

O notebook não roda benchmark de performance nem compara métricas numéricas entre a versão oficial e a HuggingFace. O objetivo era validar que a migração funciona — que os mesmos experimentos rodam com a mesma API, nos mesmos vídeos, com resultados visuais equivalentes. Benchmark quantitativo fica pro framework de evidências ([04-framework-de-evidencias](04-framework-de-evidencias.md)).

A validação também foi feita apenas em CUDA (Linux). O device detection está pronto pra MPS, mas a validação end-to-end no macOS com o fix do `pin_memory` fica como milestone pra próxima iteração.

## Dependências eliminadas

Com o notebook rodando via HuggingFace Transformers, as seguintes dependências do experimento original não são mais necessárias:

| Dependência | Por que existia | Status |
|-------------|----------------|--------|
| Fork `pedrocruz2/sam3-macos-patch` | Patches manuais em 9 arquivos do SAM 3 | Pode ser removido |
| `local-packages/decord` stub | Redirecionava pro `eva-decord` | Pode ser removido |
| Import condicional do triton no `edt.py` | Fallback scipy pra macOS | Não se aplica mais |
| `sam3[notebooks,train]` como dependência | Trazia dependências não declaradas | Substituído por `transformers` |

O `pyproject.toml` do workspace `notebooks/` agora depende de `transformers`, `torch`, `torchvision`, `opencv-python`, `Pillow`, `matplotlib`, `python-dotenv`, `huggingface-hub` e `marimo`. Sem fork, sem stub, sem patches.

## Próximos passos

1. Validar o notebook no macOS com MPS — confirmar que o fix do `pin_memory` resolve o device mismatch e que o pipeline roda end-to-end em Apple Silicon.
2. Remover o fork e o stub do `pyproject.toml` principal, já que o notebook migrado não depende deles.
3. Testar em mais vídeos pra confirmar que os resultados são equivalentes aos do experimento original.
