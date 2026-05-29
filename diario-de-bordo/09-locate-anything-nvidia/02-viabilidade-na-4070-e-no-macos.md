# Viabilidade na RTX 4070 8GB e no macOS

O [arquivo anterior](01-o-que-e-o-locate-anything.md) explicou o que o LocateAnything é. Este aqui responde a parte que decide tudo: **ele roda no nosso hardware e atende o nosso caso de uso?** A resposta é não, e os motivos são quase um espelho dos cinco gargalos que documentei pro SAM 3.

## Gargalo 1: o modelo não cabe na 8GB

O LocateAnything-3B tem 3 bilhões de parâmetros, ~4B em disco em BF16. Só os pesos em BF16 já são da ordem de **~8GB**. Isso praticamente preenche a RTX 4070 Laptop de 8GB **antes** de carregar ativações e o KV cache do decoder.

A NVIDIA só reporta benchmarks em **H100**. Não existe número oficial de VRAM de inferência nem de FPS em GPU de consumidor. O model card lista microarquiteturas suportadas e inclui Lovelace (a família da 4070/4090), mas a única placa de consumidor citada é a **RTX 4090 — que tem 24GB, não 8GB**.

Preciso ser honesto sobre o que é fato e o que é estimativa aqui: o `~8GB de pesos` é **estimativa** baseada no tamanho BF16, não um número medido pela NVIDIA. Mas a ausência de qualquer número oficial pra 8GB já é, por si só, um sinal de alerta. O SAM 3 pelo menos eu consegui rodar e medir o ponto de OOM (~100 frames). Com o LocateAnything, a fonte oficial simplesmente não responde "roda em 8GB?".

E não há quantização suportada oficialmente — sem TensorRT, TensorRT-LLM ou Triton ainda. Então não dá nem pra apostar num INT8/INT4 fácil pra caber.

## Gargalo 2: "12.7 BPS" não é FPS de vídeo

Esse é o ponto mais fácil de ler errado na página oficial. O número de velocidade que eles divulgam, **12.7 BPS**, é **boxes per second** (caixas por segundo) num H100, não frames por segundo de vídeo.

Pro nosso caso 1v1, são 2 robôs por frame. 12.7 boxes/s ÷ 2 boxes/frame ≈ **6 frames/s — e isso no H100**. Numa 4070 Laptop 8GB seria muito pior, se rodasse.

Comparar BPS de um detector generativo com FPS de um YOLO é comparar "peças por hora" com "carros por hora" numa fábrica. O PBD escala bem quando a cena é **densa** (300 objetos), mas o sumô 1v1 nunca é denso. A gente nunca colhe o benefício que justifica a arquitetura.

Pra referência, no [framework de evidências](../04-framework-de-evidencias.md) uma das métricas centrais é FPS por abordagem. YOLO/RT-DETR entregam 50 a 100+ FPS em GPU; YOLO-World faz ~74 FPS open-vocab. O LocateAnything está ordens de magnitude abaixo disso pro nosso caso.

## Gargalo 3: sem tracking temporal

Já citei no arquivo anterior, mas vale repetir aqui porque é estrutural. O LocateAnything é **frame a frame, sem memória temporal**. Diferente do SAM 3, que tinha video predictor com propagação de identidade.

Isso rebaixa o LocateAnything de "concorrente do SAM 3" pra "mais um detector". Pra medir IDF1, MOTA e ID switches — que são parte do [framework de evidências](../04-framework-de-evidencias.md) — eu precisaria de um tracker externo de qualquer forma. E se vou plugar ByteTrack/BoT-SORT por cima, faz muito mais sentido plugar num detector leve e treinável do que num VLM de 3B.

## Gargalo 4: Linux + NVIDIA, nada de macOS/MPS

O runtime oficial é Linux com GPU NVIDIA. **Não há suporte a macOS/MPS documentado.**

As dependências contam a história e são quase idênticas ao trauma de portabilidade do SAM 3:

- `decord==0.6.0` — o mesmo pacote que me forçou o stub `eva-decord` no macOS no experimento SAM 3;
- `deepspeed`, `liger_kernel` — CUDA-cêntricos;
- **MagiAttention** pra contexto longo, que só roda em Hopper/Blackwell. Sem ela, a atenção cai pra SDPA limitada a ~4K tokens.

Eu já passei por essa novela na [configuração multi-sistema](../06-configuracao-multi-sistema-operacional.md) e na [migração pro HuggingFace](../07-migracao-sam3-huggingface.md). A diferença é que pro SAM 3 existia uma reimplementação device-agnostic no HuggingFace que salvou o desenvolvimento no Mac. Pro LocateAnything não existe esse caminho de fuga ainda.

## Gargalo 5: licença, determinismo e treino

Três detalhes menores que somam contra:

- **Licença não-comercial (NVIDIA License).** Código Apache-2.0, mas os pesos travam uso comercial. OK pra pesquisa acadêmica, mas fecha a porta da visão de longo prazo (a plataforma colaborativa). Compare com YOLO (AGPL, com opção comercial) e RT-DETR (Apache-2.0 livre).
- **Não-determinismo por default.** O modo recomendado é `do_sample=True, temperature=0.7`. Saída estocástica é um anti-padrão pra anotação reprodutível e pra ciência — daria pra forçar greedy (`do_sample=False`), mas é uma pegadinha que precisa ser lembrada.
- **Fine-tuning projetado pra cluster.** O comando de treino do repo é **8 GPUs com DeepSpeed Zero-2**. Não é algo que se faz numa 4070 Laptop num fim de semana. O YOLO treina num laptop, até em MPS, em horas, pro nosso dataset pequeno.

## O que sobra de positivo

Pra não ser injusto: como **anotador**, o LocateAnything tem uma vantagem real sobre o SAM 3. Ele entende prompt textual e **devolve bounding boxes direto**, sem o passo máscara → box que o pipeline SAM 3 exigia. A API (`LocateAnythingWorker`) é conveniente: `worker.detect(frame, ["sumo robot"])` e já vem caixa.

Mas isso vem com o mesmo defeito que me frustrou no SAM 3: **fragilidade de prompt.** Open-vocabulary não entende o domínio "sumô de robôs". Igual ao `"object on metal platform"` que funcionou por acaso visual, e não por entender o que é um robô de sumô, o prompt certo aqui também é descoberta empírica, não conhecimento semântico. Dá pra sondar isso de graça no [HF Space](https://huggingface.co/spaces/nvidia/LocateAnything) antes de instalar qualquer coisa.

A [decisão e a tabela comparativa](03-comparacao-e-decisao.md) fecham a pasta.
