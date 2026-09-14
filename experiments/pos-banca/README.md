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
