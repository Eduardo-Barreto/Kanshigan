= Cronograma

O cronograma proposto abrange 12 meses:

#figure(
  caption: [Cronograma proposto para o projeto de pesquisa.],
  table(
    columns: (auto, 1fr),
    align: (center, left),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },
    table.header[Meses][Atividade],
    [1--2], [Revisão bibliográfica aprofundada, coleta de vídeos, setup de ambiente],
    [2--3], [Construção do dataset (anotação semi-automática) e detecção do dohyo],
    [3--5], [Implementação da detecção e tracking (abordagens A, B, C)],
    [5--6], [Extração de métricas e detecção de eventos],
    [6--7], [Avaliação experimental completa e análise de resultados],
    [7--8], [Escrita do artigo],
    [8--9], [Revisão com orientador e ajustes],
    [9--10], [Buffer para imprevistos e melhorias],
    [10--12], [Entrega e defesa],
  )
) <tab:timeline>

= Contribuições esperadas

+ *Kanshigan:* primeiro pipeline open-source de análise automatizada de partidas de Sumô de Robôs via visão computacional.
+ *Dataset:* primeiro dataset público anotado representativo da categoria competitiva atual de Sumô de Robôs 3kg para pesquisa em visão computacional.
+ *Benchmark:* primeira avaliação comparativa de técnicas de detecção e tracking aplicadas a esse domínio.
+ *Métricas:* definição de métricas quantitativas padronizadas para desempenho de robôs de sumô (velocidade, trajetória, tempo de reação).
+ *Perspectiva de espectador:* demonstração de como dados estruturados podem enriquecer a experiência de transmissões ao vivo.
