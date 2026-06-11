# Outline: nova apresentação pré-banca (10 min, ~23 slides)

Numeração das condições segue o paper que a banca recebeu: C1 movimento não linear, C2 aparência uniforme, C3 eventos sub-segundo, C4 vídeo heterogêneo, C5 análise post-match, C6 sem fiduciais.

## Bloco 1: Problema e lacuna (0:00 a 4:30)

**1. Título (20s)**
Visual: logo/nome Kanshigan, autores, orientador, Inteli. Fala: uma frase de abertura, sem conteúdo ainda.

**2. O que visão computacional já faz (30s)**
Visual: GIF/vídeo famoso de tracking (YOLO rastreando carros/pessoas em rua, tela cheia). Fala: "isso aqui é o estado da arte rastreando o mundo: caixas, identidades, trajetórias, em tempo real". Asset novo: preciso baixar um demo clássico de YOLO/MOT.

**3. Eu quero fazer isso aqui (30s)**
Visual: `worlds_final.mp4` tela cheia, sem overlay. Fala: "esse é o cenário que eu quero medir: final do mundial japonês, dois robôs de 3 kg, o round acaba em menos de um segundo". O contraste com o slide anterior já planta a dificuldade.

**4. Hoje, a análise é no olho (25s)**
Visual: foto de competição (plateia/replay manual), texto mínimo: "sem métrica, sem trajetória, sem replay quantificado". Fala: quem ganhou, quem rampou quem, tudo decidido no olho e no replay de celular. Sem mencionar lacuna de literatura ainda.

**5. Rastreadores genéricos → C1 (30s)**
Visual: primeira página do ByteTrack/OC-SORT + exemplo deles (pedestres MOT17). Fala: a base é tracking-by-detection com Kalman; funciona pra pedestre, mas robôs colidem, giram e ricocheteiam: quebra a hipótese de velocidade constante. Deriva na tela: **C1 movimento não linear**.

**6. DanceTrack → C2 (30s)**
Visual: primeira página + GIF/frames de dançarinos uniformizados. Fala: quando os alvos são visualmente iguais, os melhores trackers despencam (HOTA 63→48). Dois robôs são duas caixas pretas. Deriva: **C2 aparência uniforme**.

**7. SportsMOT + SoccerNet → C3 e C4 (35s)**
Visual: dividido em dois, frame de basquete + frame de broadcast de futebol, com as duas primeiras páginas. Fala: esporte real tem movimento rápido, mas um rally perde 10 quadros e tudo bem; no Sumô o round inteiro cabe num segundo. E o futebol é broadcast profissional; nosso vídeo é câmera de mão de espectador. Deriva: **C3 eventos sub-segundo** e **C4 vídeo heterogêneo**.

**8. Jiu-jitsu scoring → C5 (25s)**
Visual: figura do paper de jiu-jitsu (pose estimation na luta). Fala: análise de combate por vídeo gravado é viável, mas a técnica é pose humana, robô não tem pose. Deriva: **C5 análise post-match**.

**9. SSL-Vision (RoboCup) → C6 (25s)**
Visual: foto da Small Size League com os padrões coloridos no topo dos robôs. Fala: rastrear robô rápido é fácil quando você cola marcador e calibra câmera; o regulamento do Sumô não permite e o acervo de vídeo não tem. Deriva: **C6 sem fiduciais**.

**10. As seis condições + a lacuna (50s, slide com build)**
Visual passo 1: matriz de cobertura (trabalhos × C1-C6, igual à do paper), Kanshigan na última linha completando tudo. Visual passo 2 (avançar): tudo esmaece, **C1+C2 acendem**. Fala: cada vizinho cobre no máximo três; e o par que sobra em aberto é rastrear alvos idênticos sob movimento imprevisível, fora do estúdio, sem marcador, em hardware comum. DanceTrack isolou o par mas em captura controlada. Resolver isso vale pra drone racing, outras lutas de robô, qualquer disputa rápida entre alvos iguais. O Sumô é o caso mais puro pra atacar.

## Bloco 2: Metodologia (4:30 a 6:45)

**11. Visão geral da pipeline (30s)**
Visual: `pipeline.svg` inteiro (8 quadradinhos, 2 fileiras + ramo tracejado do anotador). Fala: pesquisa experimental, comparação controlada contra gold; primeira fileira prepara o quadro, segunda extrai informação. "Agora eu entro em cada um."

**12. Arena + calibração (30s)**
Visual: foto do dohyo com elipse amarela detectada + `dohyo_3kg.png` (geometria de regulamento). Fala: visão clássica, sem treino, porque a arena é fixada por regulamento; os 154 cm conhecidos dão a escala cm/pixel de graça, sem calibrar câmera.

**13. Recorte + detecção (25s)**
Visual: antes/depois do crop (quadro inteiro vs recorte ampliado com caixas). Fala: o recorte amplia o robô 3x e mata falso positivo da plateia (recall 0.91→0.96, precisão 0.71→0.99); em cima roda YOLO fine-tunado, duas arquiteturas comparadas.

**14. Rastreamento (25s)**
Visual: frame com trajetórias coloridas A/B + os 4 nomes (OC-SORT, ByteTrack, DeepOCSORT, BoT-SORT). Fala: C2 sugere que aparência não ajuda, mas tratamos como hipótese: 2 trackers de movimento vs 2 de aparência, mesmas detecções.

**15. Métricas + eventos (25s)**
Visual: `gold_zb01_trajectories.png` (trajetórias no referencial do dohyo). Fala: tudo em cm no referencial da arena, o que cancela o tremor da câmera; eventos por regra determinística e auditável, não classificador.

**16. Dataset + SAM 3 anotador (30s)**
Visual: frame com máscara do SAM 3 + mini-tabela do dataset (14 clips treino, 2 gold). Fala: sem dataset público, construímos um, duas fontes opostas (BR mão oblíqua, JP cenital fixa); SAM 3 anota offline, validado contra revisão humana: F1 0.96.

## Bloco 3: Resultados parciais (6:45 a 7:50)

**17. Números (35s)**
Visual: 3 cartões grandes: mAP@.5 > 0.96 (e 26n com 1/5 dos parâmetros empata), IDF1 0.94 (aparência não ajuda e custa 35-40x), >100 FPS com ~100 MB num notebook. Fala: tudo medido, nada projetado; a derrota do ReID confirma C2 na prática.

**18. Rodando em vídeo real (30s)**
Visual: grade 2×2 de vídeos: `pipeline_overlay_br.mp4` (BR câmera de mão), `demo_jp.mp4` (JP cenital), `atena_br.mp4` (outra arena BR), `rc_sumo.mp4` (categoria RC, zero-shot). Fala: mesma pipeline, sem retreino; cada vídeo evidencia uma coisa (C4, generalização, cross-categoria).

## Bloco 4: Artigo e publicação (7:50 a 8:40)

**19. O artigo existe (30s)**
Visual: `paper_p01.png`/`p02`/`p03` em leque + lista: SBC 16 pág (o que vocês receberam), versão IEEE em inglês, projeto de pesquisa ABNT, diário de bordo com 19 entradas. Fala: introdução, relacionados, metodologia e resultados já escritos; tudo que eu falei aqui está lá com mais detalhe; o diário registra o processo por escopo, incluindo o que falhou.

**20. Pra onde ele vai (25s)**
Visual: WVC (SBC) em destaque + arXiv + IEEE como reserva. Fala: WVC porque é visão computacional aplicada, o escopo exato do trabalho, em português e com a comunidade certa; preprint no arXiv pra ter link citável já; versão em inglês pronta caso surja venue internacional. (Aqui ganhamos na fala: justificar a coerência venue↔escopo.)

## Bloco 5: Limitações e próximos passos (8:40 a 9:35)

**21. Limitações assumidas + o teste mais duro (35s)**
Visual: `worlds_final.mp4` com a pipeline rodando em cima (ou `worlds_model_vs_sam.png`), mostrando onde perde os robôs no blur. Fala: limitações estão escritas no paper: gold pequeno, cm aproximado pela perspectiva, só início de round confiável; e o mundial mostra exatamente onde quebra: blur de elite. Mostrar a falha de propósito é o argumento de maturidade.

**22. Como atacamos cada uma (20s)**
Visual: três pares limitação→ação: mais gold anotado; homografia da arena; dados de broadcast no treino. Fala: cada limitação já tem o próximo passo definido.

## Fechamento

**23. Conclusão (25s, até 10:00)**
Visual: três contribuições (caracterização C1-C6, pipeline aberta, dataset validado) + link do repositório. Fala: o Sumô é a porta de entrada pro problema maior; está tudo público e detalhado no artigo. Obrigado.

---

### Como isso cobre o que importa (sem nunca dizer que cobre)

- **Problema e lacuna**: slides 2 a 10, com a lacuna emergindo dos próprios papers, não declarada de cima.
- **Metodologia em execução**: slides 11 a 18; os resultados medidos são a prova de "sprint com resultado observável".
- **Rascunho do artigo**: slide 19, com as páginas reais na tela e o reforço verbal "está tudo lá".
- **Estratégia de submissão**: slide 20, justificativa venue↔escopo na fala.

### Assets que precisarei produzir/baixar

1. GIF/vídeo de tracking famoso (slide 2): baixar demo YOLO/MOT.
2. Primeiras páginas + exemplos visuais de ByteTrack, DanceTrack, SportsMOT, SoccerNet, jiu-jitsu, SSL-Vision (slides 5 a 9): baixar dos arXiv/sites oficiais.
3. Frame antes/depois do crop (slide 13) e frame com máscara SAM 3 (slide 16): extrair dos resultados que já temos em `results/`.
4. O resto já existe em `figures/` e `videos/`.

Dois pontos pra você bater o martelo antes de eu construir:

1. **Slide 7 agrupa SportsMOT+SoccerNet** (C3+C4 juntos) pra caber nos 10 min. Se preferir um slide por trabalho, eu separo, mas aí encurto o bloco de metodologia em ~20s.
2. **Slide 21**: a evidência do mundial fica melhor com o vídeo rodando com overlay da pipeline (preciso gerar esse render) ou com a figura estática `worlds_model_vs_sam.png` que já existe? O render é mais impactante, mas depende de rodar a pipeline no vídeo do mundial.
