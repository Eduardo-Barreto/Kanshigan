# Harvest

Transforma vídeos públicos de partidas de sumô 3 kg em clips candidatos, anotados pelo SAM 3, no conjunto `web_candidates` do studio. A história das decisões está no [diário 25](../../diario-de-bordo/25-coleta-de-videos-publicos.md).

## Rodar

Tudo sai de um manifesto TSV com cabeçalho:

| Coluna | Conteúdo |
|---|---|
| `video_id` | id do YouTube |
| `source` | `br`, `jp` ou `intl` (pasta em `data/raw/` e origem no studio) |
| `event` | texto livre |
| `sections` | vazio (vídeo inteiro), `início-fim;início-fim` em segundos, ou `scout` |
| `channel`, `title` | procedência |

```bash
H=experiments/harvest/harvest.py
python3 $H scout    --manifest m.tsv --jobs 2   # só linhas `scout`: acha os rounds numa cópia 360p
python3 $H download --manifest m.tsv --jobs 2
python3 $H segment  --manifest m.tsv --jobs 8
python3 $H cut      --manifest m.tsv --jobs 8
python3 $H annotate --manifest m.tsv            # GPU, um processo
python3 $H register --manifest m.tsv            # sidecars + data/harvest/provenance.tsv
python3 $H report   --manifest m.tsv
```

Cada etapa pula o que já existe em disco, então pode ser interrompida e retomada. Etapas longas rodam como unidade do systemd (`systemd-run --user --unit=... --collect`): `nohup` morre junto com a sessão do agente.

Para conferir a anotação de um conjunto, `qa_sheet.py` monta uma prancha por clip com as caixas desenhadas:

```bash
cd experiments/pre-banca
uv run python ../harvest/qa_sheet.py ../../data/annotations/web_candidates /tmp/qa yt_
```

## Onde fica cada coisa

| Caminho | Conteúdo |
|---|---|
| `data/raw/<source>/yt_<id>[_s<início>].mp4` | vídeo baixado, ou um trecho dele |
| `data/harvest/scout/<id>.{mp4,json}` | cópia 360p e trechos achados pelo batedor |
| `data/harvest/windows/<raw>.json` | janelas de round propostas |
| `data/processed/clips/<source>/<raw>_rNN.mp4` | round recortado no dohyo |
| `data/processed/sam_masks/<clip>.npz` | máscaras do SAM 3 dos objetos mantidos, em bits, na resolução do SAM |
| `data/annotations/web_candidates/` | quadros e rótulos YOLO, revisados no studio |
| `data/harvest/annotate-status.tsv` | por clip: concluído ou falhou, quadros, segundos, limiar usado |
| `data/harvest/provenance.tsv` | por clip: vídeo, instante, canal e título |

## Escolhas que importam

- Limiar do SAM 3: JP em 0,15; BR e intl no padrão, com nova rodada em 0,15 quando menos de 30% dos quadros rotulados têm os dois robôs. 0,15 em câmera de mão rastreia mãos e chão e estoura os 8 GB.
- No máximo 8 objetos rastreados por vez no SAM 3, como teto de memória.
- Rounds entre 2 s e 30 s. Janela maior junta rounds com o reset e gente no dohyo.
- No máximo 40 trechos por transmissão longa, espalhados ao longo do dia.
- O `yt-dlp` espera entre requisições: uma rajada de consultas faz o YouTube bloquear o IP com "confirme que não é um robô".

Nada daqui vira treino ou gold sem revisão no studio.
