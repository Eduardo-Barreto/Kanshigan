# E6: medições finais sem anotação nova

Medições computacionais pedidas pela banca para o artigo final. Nenhum rótulo foi alterado, nenhum detector foi retreinado. Diário: `diario-de-bordo/23-medicoes-finais-sem-anotacao.md`. Hardware: Intel Core Ultra 9 185H (22 threads lógicas), RTX 4070 Laptop 8 GB, torch 2.12.0+cu126, Ultralytics 8.4.55, boxmot 18.0.0.

Comandos rodam de `experiments/pos-banca/`. `TE` abrevia `--with "git+https://github.com/JonathonLuiten/TrackEval@12c8791b303e0a0b50f753af204249e622d0281a"`.

| Arquivo | O que tem | Comando |
|---|---|---|
| `tracking/tracker_grid.json`, `.csv` | Grade 2 detectores x 4 rastreadores no gold_zb01: MOTA, IDF1, trocas de ID, FP, FN (motmetrics, mesma lógica do `evaluate.py`), HOTA/DetA/AssA/LocA e conferência CLEAR/Identity (TrackEval), FPS do rastreador (3 repetições). Variantes: trilhas `assigned` (após `assign_ab`) e `raw` (ids crus); gold `as_is` (118 caixas) e `visible` (caixas recortadas ao quadro do clip, 112). Estatística do gold em `gold`. | `uv run --project ../pre-banca TE python final_tracking.py` |
| `tracking/<det>_<trk>_<raw\|assigned>.txt` | Trilhas MOT de cada combinação | idem |
| `tracking/duplicate_frames.json` | Pares de quadros consecutivos quase idênticos (diferença média absoluta de cinza < 0,5 e < 1,0) | trecho Python registrado no diário 23 |
| `viability/<device>_<entrada>_<det>_<trk>.json` | Pipeline completa por estágio (decodificação, carga do modelo, processamento), 1 partida fria + 3 aquecidas, FPS ponta a ponta com decodificação, FPS equivalente ao `infer.py`, VRAM por NVML e torch | `./final_viability_batch.sh` (chama `final_viability.py`) |
| `detection/detector_eval.json` | YOLOv8s, YOLO26n e COCO nos dois golds: `model.val` (P/R no limiar de F1 máximo) e contagem a conf 0,25 com e sem pós-filtro da elipse do dohyo; COCO também sem classe (`single_cls`) | `uv run --project ../pre-banca python final_detection.py` |
| `events/*.json`, `*_tracks.txt` | `infer.py` atual (YOLOv8s + OC-SORT) no gold_zb01 recortado e nativo, atena, worlds e rc1591 | `./final_events_batch.sh` |
| `audit/baseline.csv` | Nova linha de base da auditoria (cópia da que foi escrita em `results/audit/baseline.csv`) | `uv run python audit.py ../../results/examples/demo_jp.json E/rc1591.json E/gold_zb01.json E/atena.json ../../results/examples/worlds/w_replay.json E/worlds_jE4E-XhiyjU_12401.json --csv ../../results/audit/baseline.csv` (E = `../../results/E6_final_measurements/events`) |
| `audit/baseline_before_E6.csv` | A linha de base anterior, guardada para comparação | cópia |
| `audit/gold_variants.csv` | Auditoria do gold nativo e do JSON antigo do E2 | `uv run python audit.py E/gold_zb01_native.json ../../results/E2_yolo_oc_vs_gold/gold_zb01.json --csv ...` |
| `rc/*.json`, `rc/manifest.json` | Pipeline (YOLOv8s + OC-SORT) nos 23 rounds RC; manifesto com janelas, duplicatas e eventos | `uv run --project ../pre-banca python rc_rounds.py --videos "/home/barreto/tcc/videos/3kg RC"` |
| `audit/rc_rounds.csv` | Auditoria estrutural por round RC, com a linha do gold | `uv run python audit.py E/gold_zb01.json ../../results/E6_final_measurements/rc/rc_*.json --csv ../../results/E6_final_measurements/audit/rc_rounds.csv` |
| `appearance/appearance_separability.json` | Similaridade cosseno intra vs inter robô no gold_zb01: OSNet (ReID do boxmot), ResNet-50 ImageNet, histograma HSV | `uv run --project ../pre-banca python appearance_separability.py` |
| `rounds/round_durations.json` | Duração dos rounds: janelas, instantes da pipeline (movimento, contato, saída) e dos rótulos revisados, e o subconjunto em que pipeline e rótulos concordam na saída | `uv run --project ../pre-banca python round_durations.py` |
| `sam3/sam3_vs_gold.json`, `sam3/*_labels/` | SAM 3: predictor de vídeo no gold JP (issue #14) e predictor de imagem nos dois golds (E1) | `./final_sam3_batch.sh`, depois `evaluate.evaluate_annotator` (diário 23) |
| `data_provenance.json` | Fonte e título de cada vídeo; respostas sobre sobreposição gold/treino e onde foi medido o 0,71 | `yt-dlp --skip-download --print ...` |
| `logs/` | Saída bruta de cada execução | |

Testes da lógica nova: `uv run --project ../pre-banca --with pytest TE python -m pytest tests -q` (24 testes).
