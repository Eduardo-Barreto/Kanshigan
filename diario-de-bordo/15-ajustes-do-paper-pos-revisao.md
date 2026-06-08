# Ajustes do paper após revisão

## Contexto

Revisão do artigo SBC antes da pré-banca. Cinco pontos saíram da leitura do orientando
sobre o rascunho: a figura de trajetória estava estranha, faltavam exemplos em mais
vídeos, o texto citava SAM 2 onde usamos SAM 3, as figuras do SAM precisavam refletir o
recorte padronizado, e o texto tinha marcas de escrita de IA a remover. Esta entrada
registra o que cada ponto era de fato e como foi resolvido.

## A figura de trajetória estava estranha (e por quê)

A figura vinha de `results/E2_yolo_oc_vs_gold/gold_zb01.json`, que é a saída da própria
pipeline (YOLOv8s + OC-SORT) sobre o round gold brasileiro `gold_zb01`. O gold de
detecção desse round é o aprovado manualmente, então o dado de base é confiável. O
problema é o round em si e a leitura cinemática:

- O round é um standoff de ~5 s (os dois robôs quase parados, velocidade < 3 cm/s)
  seguido de uma arrancada explosiva (robô A a 291 cm/s, B a 267 cm/s) que o clip corta
  no meio do impacto. No plano, isso vira dois rabiscos minúsculos: visualmente fraco.
- O evento `round_start` disparava no quadro 1 (t = 16 ms), errado. Era artefato de
  borda do filtro de Savitzky-Golay: a aceleração nos três primeiros quadros é uma
  constante (957 cm/s² para A), o efeito de fronteira clássico do filtro. O detector de
  eventos disparava nesse ruído, não no movimento real.
- Os robôs ficam a raio ~31 e ~39 cm do centro (arena de raio 77 cm), longe da borda:
  não há ring-out para mostrar.

Decisão: tirar a figura de trajetória e o perfil de velocidade do paper por enquanto, e
manter só a tabela de rastreamento (IDF1 0.93, MOTA 0.88), que é sólida e aprovada. A
análise cinemática em centímetros volta na versão final, junto com a base de eventos
anotada e a calibração dos limiares. A projeção em centímetros sob câmera de mão oblíqua
também é aproximada, o que reforça segurar a cinemática quantitativa até a retificação
por homografia. O `round_start` por artefato de borda fica como bug conhecido a corrigir
antes de reintroduzir eventos (guardar contra os primeiros/últimos quadros do SG).

## Rodar a pipeline em mais vídeos

Rodamos o detector treinado (sem retreino) em footage novo para gerar mais exemplos:

- `atenavsbullbassauro.mp4` e `IMG_1591.MOV` (acervo "3kg RC" da ThundeRatz): Sumô de
  rádio-controle. Categoria diferente da treinada (Sumô autônomo), com robôs de chassi
  diferente e arena de superfície metálica que não aparece no treino. O detector achou
  os dois robôs e a arena em ambos, zero-shot, a 170 fps (atena, 848×478) e ~34 fps
  end-to-end (IMG_1591, decodificação 4K dominando o tempo).

Isso virou uma subseção nova na discussão, "Generalização cross-categoria (Sumô RC)",
com figura qualitativa (`cross_category_rc.png`). Sem gold rotulado para o RC, não
reportamos métrica: o resultado é qualitativo e entra como evidência a favor da classe
de problemas mais ampla (o detector aprende robô-de-combate-em-arena-circular, não um
campeonato específico). Nota: `atenavsbullbassauro.mp4` é do mesmo acervo RC, não é
footage autônomo.

## SAM 2 vs SAM 3

O texto citava `SAM 2` num ponto de trabalhos relacionados, mas usamos o SAM 3. Trocado
para `SAM 3` (@carion2025sam3). A entrada `ravi2024sam2` continua na bib como predecessor
de segmentação em vídeo, mas não é mais citada.

## Figuras do SAM e o recorte padronizado

As figuras compostas (`qualitative_br_jp.png`, `worlds_model_vs_sam.png`) tinham sido
geradas por comandos avulsos, sem script versionado, e a do BR ainda trazia "Amador"
no rótulo (já tínhamos combinado não chamar o footage brasileiro de amador). Reescrevi a
geração como `experiments/pre-banca/compose_figures.py`: lê os MP4 de overlay, escolhe um
quadro em que os dois robôs estão rastreados (pela interseção dos frames no JSON), aplica
um rótulo com acento via PIL e monta os painéis. Agora toda figura qualitativa do paper é
reprodutível com um comando. As figuras do SAM no caso de mundial usam os overlays já
regenerados com o recorte do dohyo casado entre SAM e nosso modelo.

## Remoção de marcas de IA (anti-slop)

Passei o paper pelas diretrizes anti-slop mantendo o tom formal:

- Travessões (`---`, que o Typst renderiza como em dash): 12 ocorrências, todas
  removidas (vírgula, parênteses ou reescrita). É também regra fixa do projeto.
- Construção "não X, e sim Y" (a marca estrutural de IA mais frequente): reescrita para
  começar pelo que a coisa é.
- Inflação ("é a evidência central", "a entrega concreta de C3", "contra a intuição"):
  trocada por afirmação direta.
- Palavras-marca em português ("paradigma", "cenário"): removidas onde eram decorativas.
- Clausula meta de preenchimento ("essa honestidade responde ao critério de validação"):
  cortada.

Mantidos termos técnicos onde "robusto" significa robustez estatística, não enfeite.

## Status

- Paper recompila em 10 páginas, sem erros.
- Figura de trajetória/velocidade fora do paper; tabela de rastreamento mantida.
- Exemplos novos rodados; subseção cross-categoria RC com figura.
- SAM 2 corrigido para SAM 3; figuras compostas agora reprodutíveis via script.
- Travessões e marcas de IA removidos.
- Pendente: corrigir o `round_start` por artefato de borda do SG e calibrar ring-out e
  contato antes de reintroduzir a cinemática quantitativa.
