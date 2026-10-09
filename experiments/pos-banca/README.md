# Pos-banca

Trabalho posterior a pre-banca. A pipeline de inferencia continua em
[`../pre-banca`](../pre-banca), congelada como o artefato entregue; aqui ficam os
estudos que a leem.

## Auditoria estrutural sem rotulos (`audit.py`)

### Por que

O acervo de Sumo RC (radio-controlado) aparece hoje no artigo apenas como evidencia
qualitativa. A frase em `docs/paper-sbc/sections/discussion.typ` e explicita:

> "Nao ha gold rotulado para o RC, entao o resultado e qualitativo: nenhuma metrica e
> reportada sobre ele."

Anotar esse footage quadro a quadro custa horas que a janela nao tem. Mas o dominio
restringe o que uma saida correta precisa ser, sem nenhuma anotacao: sao exatamente
dois robos no dohyo durante o round, o movimento de um robo de 3 kg e limitado pela
fisica, e a trajetoria de cada robo e continua. Toda violacao dessas restricoes e
observavel no proprio JSON da pipeline.

### O que torna isso defensavel

Os indicadores sao um proxy, nao verdade de campo, e proxy so vale reportado com
ancora. A ancora e o round gold (`gold_zb01`), onde MOTA e IDF1 foram medidos contra
anotacao humana: ler a taxa de violacao de um clip ao lado da do gold diz se ele opera
num regime cuja qualidade real de rastreamento e conhecida. Por isso a tabela sempre
traz a linha do gold.

Os indicadores sao calculados dentro da janela do round (`round_start` a `round_end`),
porque quadros antes do hajime e depois do ring-out mostram legitimamente menos de dois
robos; conta-los como falha inflaria todas as taxas.

### Uso

```bash
uv sync
uv run python audit.py ../../results/E2_yolo_oc_vs_gold/gold_zb01.json \
                       ../../results/examples/rc1591/rc1591.json \
                       --csv ../../results/audit/rc.csv
```

Testes da logica pura: `uv run --group dev pytest`.

### Indicadores

| Campo | O que mede |
|---|---|
| `both_present_rate` | fracao dos quadros do round com os dois robos rastreados |
| `cardinality_violation_rate` | complemento do acima: o indicador principal |
| `longest_gap_frames` / `_ms` | maior lacuna continua sem os dois robos |
| `max_step_cm` / `n_teleports` | deslocamento entre quadros vizinhos; salto impossivel indica troca de identidade |
| `implausible_speed_rate` | fracao de quadros acima do teto fisico de velocidade |
| `ring_out_detected` | se o round resolveu por ring-out ou caiu em `timeout_manual_review` |

## Escala do treino JP (`data_scaling.py`)

Compara detectores treinados sem e com os rounds JP adicionais (`candidates_jp`), em
varias sementes, nos dois golds. Alem do mAP, conta os erros a confianca fixa e os separa
por lugar: robos perdidos, falsos positivos dentro do dohyo e falsos positivos fora dele.
A separacao importa porque o gold so rotula robos em jogo: um detector que tambem marca o
robo ja ejetado leva falso positivo por divergencia de convencao, nao por alucinacao.

Precisa da pilha do detector, entao roda com o ambiente da pre-banca:

```bash
T=../../results/training
uv run --project ../pre-banca python data_scaling.py \
    --before $T/yolov8s_kanshigan{,_s0,_s1}/weights/best.pt \
    --after $T/yolov8s_kanshigan_jpplus{,_s0,_s1}/weights/best.pt \
    --cases gold_jp:gold_jp25_0020 gold_jp:gold_jp25_0031
```

Saidas: `results/E5_jp_data_scaling/summary.json` (por semente, media e desvio) e
`results/figures/jp_data_scaling_cases.png` (antes e depois, primeiro par de sementes).
