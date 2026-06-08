= Discussão e Trabalhos Futuros <sec-discussao>

== Limitações assumidas

O conjunto gold é pequeno, o que impede intervalos de confiança e testes de
significância robustos. Reportamos os números como ordem de grandeza e demonstração
de viabilidade, não como modelo definitivo; essa honestidade responde diretamente
ao critério de validação. A footage amadora é capturada com câmera de mão e ângulo
oblíquo, então a calibração centímetro-por-pixel por escala isotrópica é uma
aproximação: a perspectiva introduz erro de foreshortening que as métricas de
detecção e rastreamento (mAP, IDF1, FPS) não sofrem, mas que afeta as métricas
cinemáticas em centímetros. A retificação por homografia da arena fica como
trabalho futuro.

O OC-SORT é motion-only, sem características de aparência. Em oclusão prolongada
entre dois robôs visualmente idênticos, pode trocar identidades. Documentamos isso
como expectativa; o Deep HM-SORT @deephmsort2024, com características profundas, é o
próximo passo para endereçar o ponto, e o ByteTrack entra como segundo ponto
experimental para comparação empírica de rastreadores.

== Anotação heterogênea (C3)

Treinar nas duas fontes exigiu adaptar a anotação semiautomática. Os robôs japoneses,
em vista cenital, são caixas pretas pequenas que pontuam baixo para o conceito textual
do SAM 3, então o limiar de detecção padrão (0.5 a 0.7) os perdia por completo;
baixá-lo para 0.15, somado a uma resolução de entrada maior (960 px) e a um filtro
geométrico que descarta caixas fora do dohyo, recuperou os dois robôs. Com isso o
treino multi-fonte foi viabilizado, e um único detector atinge mAP acima de 0.97 em
ambas as fontes, apesar das câmeras opostas. Essa robustez entre fontes é a entrega
concreta da restrição C3.

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
