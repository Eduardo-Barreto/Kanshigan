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

== Transferência cross-source (C3)

Como evidência inicial da restrição de qualidade heterogênea (C3), avaliamos o
detector treinado apenas com footage amador brasileiro (câmera de mão, ângulo
oblíquo) sobre footage de torneio japonês (câmera fixa cenital). Em zero-shot, sem
nenhum exemplo japonês no treino, o detector localiza ambos os robôs nos quadros
nítidos, demonstrando transferência entre fontes. O recall cai nos quadros de vista
cenital com robôs pequenos e escuros. A anotação semiautomática com SAM 3 na fonte
japonesa esbarra em dois pontos: a entrada decimada precisa de mais resolução (a 480
px o robô cenital fica abaixo do tamanho de detecção do SAM; a 960 px ele volta a
segmentar os dois robôs), e o SAM de vídeo semeia no primeiro quadro, que em muitos
rounds japoneses começa com os robôs ocluídos. Por isso o treino multi-fonte depende
de anotação manual ou de semeadura em um quadro nítido escolhido, deixado como
trabalho futuro.

== Trabalhos futuros

Além da homografia e do rastreador com aparência, planejamos: treino multi-fonte com
footage de broadcast japonês para fechar a heterogeneidade de qualidade (C3); ablação
do detector com RT-DETR @zhao2024rtdetr na arena controlada;
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
