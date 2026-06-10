= Discussão e Trabalhos Futuros <sec-discussao>

== Limitações assumidas

O conjunto gold é pequeno, o que impede intervalos de confiança e testes de
significância confiáveis. Reportamos os números como ordem de grandeza e demonstração
de viabilidade, não como modelo definitivo. A footage brasileira é capturada com câmera de mão e ângulo
oblíquo, então a calibração centímetro-por-pixel por escala isotrópica é uma
aproximação: a perspectiva introduz erro de foreshortening que as métricas de
detecção e rastreamento (mAP, IDF1, FPS) não sofrem, mas que afeta as métricas
cinemáticas em centímetros. A retificação por homografia da arena fica como
trabalho futuro.

Rastreadores motion-only podem trocar identidades sob oclusão prolongada entre dois
robôs idênticos. Testamos diretamente se a aparência corrige isso: além de OC-SORT e
ByteTrack, avaliamos DeepOCSORT e BoT-SORT, ambos com ReID, sobre as mesmas detecções
(@tab-tracking). A aparência não ajudou. Os quatro mantêm no máximo uma troca no round
gold, mas o passo de ReID custa de 35 a 40 vezes em throughput sem ganho de acurácia.
A causa é o próprio domínio: dois robôs pretos quase idênticos oferecem pouco sinal de
aparência para o ReID explorar, então o movimento basta, e o motion-only é a escolha
viável. Um rastreador de aparência especializado em alvos pouco texturizados, como o
Deep HM-SORT @deephmsort2024, e mais rounds gold com identidade ainda podem refinar a
comparação com significância estatística; a base de um round já indica que a aparência
genérica não compensa aqui.

== Anotação heterogênea (C4)

Treinar nas duas fontes exigiu adaptar a anotação semiautomática. Os robôs japoneses,
em vista cenital, são caixas pretas pequenas que pontuam baixo para o conceito textual
do SAM 3, então o limiar de detecção padrão (0.5 a 0.7) os perdia por completo;
baixá-lo para 0.15, somado a uma resolução de entrada maior (960 px) e a um filtro
geométrico que descarta caixas fora do dohyo, recuperou os dois robôs. Com isso o
treino multi-fonte funcionou, e um único detector atinge mAP acima de 0.96 em
ambas as fontes, apesar das câmeras opostas, o que sustenta a condição C4.

== Generalização cross-categoria (Sumô RC)

O detector foi treinado apenas em footage de Sumô autônomo, mas a mesma pipeline,
sem retreino, encontra os dois robôs e a arena em partidas de Sumô de rádio-controle
(RC), uma categoria distinta, com robôs de chassi diferente e uma arena de superfície
metálica que não aparece no treino (@fig-rc). A @fig-rc mostra duas partidas RC do
acervo da ThundeRatz. Não há gold rotulado para o RC, então o resultado é qualitativo:
nenhuma métrica é reportada sobre ele. Ainda assim, a transferência zero-shot para uma
categoria não vista é evidência a favor da classe de problemas mais ampla descrita na
@tab-matrix, em que o detector aprende a aparência de robôs de combate em uma arena
circular, não a aparência de um campeonato específico.

#figure(
  image("/results/figures/cross_category_rc.png", width: 100%),
  caption: [Generalização zero-shot para Sumô de rádio-controle (categoria fora do treino). O detector, treinado só em Sumô autônomo, mantém arena (amarelo) e robôs A (verde) e B (laranja) em duas partidas RC sobre arena metálica. Resultado qualitativo, sem gold rotulado.],
) <fig-rc>

== Caso extremo: final de mundial (out-of-distribution)

Testamos a pipeline na final do 84º All Japan Robot Sumo (3 kg autônomo), footage de
broadcast bem fora da distribuição de treino: arena azul-escura, overlay de placar,
cortes frequentes e colisões muito mais violentas, com blur extremo. A @fig-worlds
compara o nosso detector e o SAM 3 nos mesmos dois instantes do Round 1, um por linha:
à esquerda o nosso detector no quadro nativo, à direita o SAM 3 no recorte do dohyo. No
início, em movimento lento, os dois métodos acham os dois robôs, e a detecção do dohyo
generaliza para a arena de cor nova. No movimento rápido, o blur extremo derruba o nosso
detector, que perde as duas caixas, e a metade final do round fica majoritariamente sem
detecção: o modo de falha que um quadro do início, sozinho, esconderia. No mesmo
instante, o SAM 3, sobre o recorte mais limpo e com propagação temporal, ainda sustenta
um dos robôs. O fator limitante não é a semelhança entre os robôs, que carregam
bandeiras distintas, mas o blur do combate de elite. O SAM 3 é o anotador pesado e
offline, não a pipeline em tempo real; fechar o caso no modelo rápido exige dados de
treino dessa distribuição, com movimento rápido e blur, não apenas mais exemplos da
arena.

#figure(
  image("/results/figures/worlds_model_vs_sam.png", width: 100%),
  caption: [Nosso detector vs SAM 3 nos mesmos instantes do Round 1 da final de mundial (out-of-distribution). Cada linha é um instante; à esquerda, nosso detector no quadro nativo; à direita, SAM 3 no recorte do dohyo. Em cima, o início em movimento lento: ambos acham os dois robôs. Embaixo, um momento de movimento rápido: nosso modelo perde os dois, o SAM mantém um.],
) <fig-worlds>

== Trabalhos futuros

Além da homografia, planejamos: ampliar a base japonesa
e adicionar broadcast profissional; ampliar os rounds gold com identidade para dar
poder estatístico à comparação de rastreadores; ablação do detector com RT-DETR
@zhao2024rtdetr na arena controlada;
detecção de contato por máscara em vez de bounding box; e a evolução para uma
plataforma colaborativa aberta de análise de combate de robôs. A pipeline e o
conjunto de dados são públicos para reprodução e extensão.

= Conclusão

Apresentamos o Kanshigan, a primeira pipeline aberta de detecção, rastreamento e
extração de métricas para Sumô de Robôs autônomos. Caracterizamos o domínio por
seis condições e recortamos a questão científica ao subconjunto em aberto:
rastrear alvos de aparência uniforme sob movimento não linear, no equilíbrio
entre acurácia e viabilidade. Construímos a pipeline a partir de
tracking-by-detection, comparando duas arquiteturas de detector (YOLOv8s e
YOLO26n) e quatro rastreadores (motion-only e com aparência), com o SAM 3 como
anotador. Os resultados parciais sobre footage real demonstram
viabilidade em hardware de consumo e estabelecem a base experimental para o trabalho
final.
