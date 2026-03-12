# Setup e Dependências

## Por que SAM 3?

A ideia era usar um foundation model pra segmentar robôs em vídeo de forma semi-automática. Três opções:

- **SAM (original):** só imagem, sem vídeo nativo. Precisaria rodar frame a frame.
- **SAM 2:** tem video predictor nativo (propaga segmentação entre frames), mas precisa de prompt manual (ponto ou bounding box). Você precisa clicar no robô.
- **SAM 3:** aceita prompt por texto ("robot"), detecta automaticamente e faz tracking no vídeo. Melhor dos dois mundos.

SAM 3 foi lançado em novembro de 2025 pela Meta. Tem 848M parâmetros. O código tá no [GitHub](https://github.com/facebookresearch/sam3) e os pesos no [HuggingFace](https://huggingface.co/facebook/sam3) (gated, precisa pedir acesso).

## Hardware

- **GPU:** RTX 4070 Laptop (8GB VRAM)
- **RAM:** 30GB
- **Python:** 3.12.12

A estimativa inicial: modelo ~3.4GB em bfloat16, sobrando ~4GB pra frames e processamento. Apertado mas possível.

## Gerenciador de pacotes: uv

Usado `uv` pra gerenciar o projeto Python. A configuração do `pyproject.toml` precisou de atenção especial por causa do PyTorch com CUDA.

### Problema 1: index do PyTorch vs PyPI

PyTorch com CUDA precisa vir de um index especial (`https://download.pytorch.org/whl/cu126`). O uv tem proteção contra dependency confusion: quando acha um pacote em qualquer index, só busca versões daquele index. Isso quebrou o `iopath` (SAM 3 precisa >=0.1.10, index do PyTorch só tem 0.1.9).

**Solução:** marcar o index do PyTorch como `explicit = true`. Só pacotes explicitamente apontados pra ele (torch, torchvision, torchaudio) buscam lá. O resto cai no PyPI.

```toml
[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cuda" }
torchvision = { index = "pytorch-cuda" }
torchaudio = { index = "pytorch-cuda" }
sam3 = { git = "https://github.com/facebookresearch/sam3.git" }
```

### Problema 2: pkg_resources removido

`setuptools` 82 removeu o `pkg_resources`, que o SAM 3 ainda usa. Fix: `setuptools<82`.

### Problema 3: dependências não declaradas

O pacote sam3 não declara várias dependências obrigatórias (einops, decord, pandas, scipy, etc.). Elas estão nos extras `[notebooks]` e `[train]`, mas o código de inferência importa tudo. Precisei instalar com `sam3[notebooks,train]`.

### Problema 4: ambientes cruzados

O uv resolve dependências pra todas as plataformas por padrão. Limitei pra Linux x86_64:

```toml
[tool.uv]
environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
```

## Autenticação HuggingFace

Os pesos do SAM 3 são gated. Pedi acesso em `facebook/sam3` no HuggingFace e autentiquei via:

```bash
uv run python -c "from huggingface_hub import login; login(token='hf_TOKEN')"
```
