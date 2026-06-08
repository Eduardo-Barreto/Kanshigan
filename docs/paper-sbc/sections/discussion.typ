= Discussão e Trabalhos Futuros <sec-discussao>

== Limitações assumidas

O conjunto gold é pequeno, o que impede intervalos de confiança e testes de
significância robustos. Reportamos os números como ordem de grandeza e demonstração
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
aparência para o ReID explorar, então o movimento basta. Para este caso, o motion-only
é a escolha viável. Um rastreador de aparência especializado em alvos pouco texturizados,
como o Deep HM-SORT @deephmsort2024, e mais rounds gold com identidade ainda podem
refinar essa comparação com significância estatística; a base de um round, porém, já
indica que a aparência genérica não compensa aqui.

== Anotação heterogênea (C3)

Treinar nas duas fontes exigiu adaptar a anotação semiautomática. Os robôs japoneses,
em vista cenital, são caixas pretas pequenas que pontuam baixo para o conceito textual
do SAM 3, então o limiar de detecção padrão (0.5 a 0.7) os perdia por completo;
baixá-lo para 0.15, somado a uma resolução de entrada maior (960 px) e a um filtro
geométrico que descarta caixas fora do dohyo, recuperou os dois robôs. Com isso o
treino multi-fonte foi viabilizado, e um único detector atinge mAP acima de 0.96 em
ambas as fontes, apesar das câmeras opostas, o que sustenta a restrição C3.

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
cortes frequentes entre dohyo e operadores, e colisões muito mais violentas, com blur
extremo. A @fig-worlds segue o nosso detector ao longo da dificuldade do Round 1 (linha
de cima) e o compara com o SAM 3 (linha de baixo). No início, com os robôs ainda em
movimento lento, o detector rastreia os dois de forma estável, e a detecção do dohyo
generaliza para a arena de cor nova. No auge da colisão, porém, o movimento rápido e o
blur extremo derrubam todas as caixas: o modelo perde os dois robôs exatamente no lance
decisivo, e da colisão em diante a metade final do round fica majoritariamente sem
detecção. É o modo de falha que importa registrar, e que um quadro do início, sozinho,
esconderia. O SAM 3 esbarra no mesmo limite: o replay em câmera lenta, que se esperaria
mais fácil, ainda chega desfocado quadro a quadro. O fator limitante não é a semelhança
entre os robôs, que carregam bandeiras distintas, mas o blur do combate de elite.
Fechar esse caso exige dados de treino dessa distribuição, com movimento rápido e blur,
não apenas mais exemplos da arena.

#figure(
  image("/results/figures/worlds_model_vs_sam.png", width: 100%),
  caption: [Pipeline na final de mundial (out-of-distribution). Linha de cima: nosso detector no Round 1, do início em movimento lento (rastreia A e B) ao auge da colisão em movimento rápido (perde os dois sob blur). Linha de baixo: SAM 3 no recorte do dohyo, no Round 1 e no replay em câmera lenta.],
) <fig-worlds>

== Trabalhos futuros

Além da homografia e do rastreador com aparência, planejamos: ampliar a base japonesa
e adicionar broadcast profissional; ablação do detector com RT-DETR @zhao2024rtdetr na
arena controlada;
detecção de contato por máscara em vez de bounding box; e a evolução para uma
plataforma colaborativa aberta de análise de combate de robôs. A pipeline e o
conjunto de dados são públicos para reprodução e extensão.

= Conclusão

Apresentamos o Kanshigan, a primeira pipeline aberta de detecção, rastreamento e
extração de métricas para Sumô de Robôs autônomos. Caracterizamos o problema pela
interseção de seis restrições não cobertas em conjunto pela literatura, e
construímos a pipeline a partir de tracking-by-detection com YOLOv8s e OC-SORT,
usando o SAM 3 como anotador. Os resultados parciais sobre footage real demonstram
viabilidade em hardware de consumo e estabelecem a base experimental para o trabalho
final.
