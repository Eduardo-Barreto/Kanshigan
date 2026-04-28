# Por que SAM 3 não é viável como pipeline final

## Contexto

O experimento com SAM 3, documentado no [diário 05](05-experimento-sam3-poc/04-resultados-e-limitacoes.md), foi útil. Ele provou três coisas importantes:

1. Robôs de sumô 3kg podem ser segmentados em vídeo mesmo sem um dataset específico do domínio.
2. O tracking temporal consegue manter identidade entre robôs, pelo menos em vídeos curtos e controlados.
3. A combinação de ROI do dohyo, prompt textual e filtros simples já produz uma pipeline funcional.

Mas isso não significa que SAM 3 seja a arquitetura certa para o pipeline final do Kanshigan. A pergunta da pesquisa não é "qual modelo mais poderoso consegue segmentar um vídeo?", e sim como a escolha da arquitetura afeta acurácia e viabilidade prática na classificação automática de partidas de Sumô de Robôs.

Depois dos testes, a conclusão é: **SAM 3 é promissor como ferramenta auxiliar de anotação, mas não é viável como pipeline principal de detecção e tracking para a entrega desta fase**.

## O que mudou desde a hipótese inicial

A hipótese inicial era razoável: SAM 3 parecia juntar o melhor dos dois mundos.

- SAM original segmenta imagens, mas não tem vídeo nativo.
- SAM 2 tem video predictor, mas depende de prompt manual por ponto, box ou máscara.
- SAM 3 aceita prompt por texto, detecta instâncias automaticamente e propaga tracking no vídeo.

No papel, isso resolveria boa parte do problema: bastaria usar um prompt como `"robot"` ou `"sumo robot"` e deixar o modelo detectar os dois competidores ao longo do round.

Na prática, o experimento mostrou que o problema é mais específico.

O prompt `"robot"` não detectou nada. O prompt que funcionou melhor foi `"object on metal platform"`, porque descreve o que o modelo enxerga visualmente, não o que o objeto é semanticamente. Isso já indica que o SAM 3 não entende o domínio "sumô de robôs" diretamente. Ele consegue resolver uma aproximação visual do problema.

Essa aproximação funciona em alguns vídeos, mas cria incerteza para escala: vídeos com ângulos diferentes, iluminação ruim, plateia visível, robôs esperando na lateral ou outros objetos sobre a arena podem mudar completamente o comportamento do prompt.

## Gargalo 1: VRAM e taxa de quadros

O principal gargalo foi memória de GPU.

No vídeo de teste:

| Configuração | Resultado |
|--------------|-----------|
| 60fps, 848x478 | OOM |
| 24fps, 848x478 | OOM |
| 10fps, 848x478 | OOM |
| 5fps, 848x478 | Funciona |
| 10fps, 480x270 | Funciona |
| 12fps, 480x270 | OOM |
| 15fps, 480x270 | OOM |
| 15fps, 320p | OOM |

Na RTX 4070 Laptop de 8GB, o limite prático ficou em torno de 100 frames por vídeo. Para um round de 10 segundos, isso equivale a 10fps. Para um round mais longo, seria necessário reduzir ainda mais a taxa de quadros ou quebrar o vídeo em janelas.

Isso é ruim para Sumô de Robôs. As lutas são rápidas, colisões acontecem em poucos frames, e eventos como primeiro contato, ring-out e fim do round dependem de precisão temporal. Rodar a 5fps ou 10fps pode ser aceitável para visualizar uma segmentação, mas é fraco para medir eventos rápidos com confiança.

O ponto mais importante: reduzir resolução ou aplicar crop no dohyo não resolveu proporcionalmente. O custo dominante veio dos feature maps mantidos pelo video predictor, não dos pixels brutos do vídeo. O ROI ajuda a remover falsos positivos, mas não muda o limite estrutural de frames que cabem na VRAM.

## Gargalo 2: pipeline pesado para uma entrega curta

SAM 3 exige mais infraestrutura do que parece:

- pesos gated no HuggingFace;
- autenticação manual;
- dependências CUDA específicas;
- PyTorch instalado por índice separado;
- dependências não declaradas corretamente no pacote oficial;
- problemas com `setuptools`;
- `triton` sem suporte no macOS;
- imports CUDA hardcoded;
- fork temporário para compatibilidade;
- workaround de `decord`/`eva-decord`;
- comportamento diferente entre Linux NVIDIA e macOS Apple Silicon.

O [post anterior](07-migracao-sam3-huggingface.md) mostrou que migrar para HuggingFace Transformers resolve parte disso. A API do HuggingFace é mais device-agnostic e remove vários workarounds do repositório oficial.

Mas isso resolve principalmente **portabilidade de desenvolvimento**, não a viabilidade do SAM 3 como pipeline final. Mesmo com uma API melhor, o modelo continua grande, o processamento de vídeo continua caro, e a entrega ainda dependeria de validar uma arquitetura foundation pesada em um domínio pequeno e específico.

Para uma entrega nesta semana, isso é risco demais.

## Gargalo 3: falsos positivos e dependência do prompt

O melhor prompt encontrado foi `"object on metal platform"`. Ele funciona porque os robôs são objetos escuros sobre uma arena metálica. Mas essa frase também pode capturar:

- o próprio dohyo;
- robôs esperando fora da luta;
- ferramentas ou objetos próximos da arena;
- partes da estrutura da competição;
- outros elementos escuros sobre superfícies brilhantes.

Foi necessário adicionar dois filtros:

1. descartar máscaras muito grandes, porque o SAM 3 às vezes segmenta o dohyo inteiro;
2. manter apenas as máscaras mais próximas do centro do ROI, porque robôs fora da luta podem ser detectados.

Esses filtros são úteis, mas também mostram que o SAM 3 sozinho não entrega a semântica necessária. A pipeline passa a depender de heurísticas do domínio de qualquer forma. Se já precisamos de ROI, filtros geométricos e validação manual, faz mais sentido usar essas heurísticas ao redor de um detector leve e treinável.

## Gargalo 4: tracking bom, mas não suficiente

O video predictor manteve identidades no vídeo principal. Isso foi um resultado positivo.

Mesmo assim, nos testes com rumble, o tracking se confundiu em colisões. Essa é justamente uma situação crítica no domínio: os robôs colidem, se sobrepõem, giram e podem desaparecer parcialmente sob o oponente ou sob motion blur.

Não é um erro inesperado. O próprio problema é difícil. Mas a consequência prática é que SAM 3 não elimina a necessidade de avaliação quantitativa com métricas de tracking como IDF1, MOTA e ID Switches. Sem essa avaliação, não dá para afirmar que a identidade dos robôs se mantém nas partes importantes da luta.

Para a entrega atual, o caminho mais defensável é comparar detectores e trackers tradicionais no mesmo dataset, como já está definido no [framework de evidências](04-framework-de-evidencias.md).

## Gargalo 5: SAM 3.1 melhora eficiência, mas não muda a decisão

Em março de 2026, a Meta lançou o SAM 3.1 com Object Multiplex, prometendo tracking multiobjeto mais eficiente. Segundo a documentação oficial do repositório, a atualização introduz uma abordagem de memória compartilhada para tracking conjunto de múltiplos objetos e exige o código mais recente do repositório para usar os novos checkpoints. A postagem da Meta também destaca ganho de throughput em hardware de alto desempenho, com exemplo em H100.

Isso é relevante, mas não resolve o nosso problema imediato.

O nosso cenário é uma RTX 4070 Laptop de 8GB, vídeos curtos de competição, desenvolvimento em macOS e uma entrega com prazo curto. A melhoria do SAM 3.1 pode reduzir custo quando muitos objetos são rastreados, mas a partida 1v1 tem só dois robôs. O gargalo observado não foi apenas "muitos objetos"; foi o custo de manter o estado de vídeo e os feature maps por frame dentro da VRAM.

Portanto, SAM 3.1 merece ser testado no futuro, mas não deve ser colocado no caminho crítico da entrega desta semana.

## Decisão

SAM 3 não deve ser o pipeline final de inferência do Kanshigan agora.

O papel correto dele é:

- gerar pseudo-labels quando funcionar bem;
- acelerar anotação inicial de alguns vídeos;
- servir como baseline foundation model na discussão metodológica;
- ajudar a validar visualmente hipóteses de ROI, máscara e tracking.

O papel incorreto dele é:

- ser o detector/tracker principal da pipeline final;
- ser requisito para toda inferência;
- depender de GPU grande para processar vídeos em taxa útil;
- substituir um dataset anotado e uma avaliação quantitativa.

Essa decisão não invalida o experimento. Pelo contrário: o experimento foi necessário para descobrir o limite prático antes de comprometer a pesquisa com uma arquitetura pesada demais.

## Abordagens alternativas

### 1. YOLOv8 ou YOLOv11 + ByteTrack

Essa deve ser a primeira alternativa implementada.

YOLO é leve, rápido, simples de treinar e já aparece como baseline natural no projeto. ByteTrack é adequado como primeira estratégia de tracking porque associa detecções de alta e baixa confiança, o que ajuda quando o detector oscila por motion blur ou oclusão parcial.

Vantagens:

- roda em tempo real com muito mais facilidade;
- permite treinar especificamente em robôs de sumô 3kg;
- é fácil medir mAP, FPS, MOTA, IDF1 e ID switches;
- reduz dependência de prompts textuais;
- encaixa direto na metodologia já escrita.

Limitação:

- exige dataset anotado.

Mesmo assim, esse custo é mais controlável do que tentar estabilizar SAM 3 em todos os vídeos. O SAM 3 pode ajudar a gerar rótulos iniciais, mas o modelo final deve ser supervisionado e específico do domínio.

### 2. YOLOv8 ou YOLOv11 + BoT-SORT

BoT-SORT deve ser a segunda comparação.

Ele tende a ser mais robusto em cenários com oclusão porque usa features de aparência além de movimento. No sumô de robôs, isso pode ajudar em colisões e cruzamentos, mas também pode ser limitado porque muitos robôs são visualmente parecidos: caixas pretas, baixos, com pouca textura.

Vantagens:

- melhora potencial em troca de ID;
- comparação direta com ByteTrack;
- boa aderência ao framework de evidências.

Limitação:

- pode não ganhar muito quando os robôs têm aparência semelhante;
- pode ser mais pesado que ByteTrack.

Mesmo se BoT-SORT não vencer, ele fortalece a pesquisa porque mostra empiricamente se features de aparência ajudam nesse domínio.

### 3. RT-DETR + ByteTrack

RT-DETR é a alternativa baseada em transformer para detecção.

Ela é interessante porque testa uma hipótese diferente da família YOLO: detecção end-to-end, sem depender do mesmo tipo de pipeline de NMS. Para a pesquisa, isso dá diversidade arquitetural.

Vantagens:

- comparação técnica mais rica;
- pode lidar melhor com objetos difíceis em alguns cenários;
- já está previsto na metodologia.

Limitação:

- treinamento e inferência tendem a ser mais pesados que YOLO;
- provavelmente não é a opção mais rápida para uma entrega curta.

Para esta semana, RT-DETR pode ficar como terceira abordagem ou experimento planejado, não como bloqueador.

### 4. Pipeline geométrico + detector leve

O experimento com SAM 3 mostrou que a parte geométrica é valiosa: detectar o dohyo por threshold na tawara, criar ROI dinâmico e filtrar objetos por posição já resolve uma parte importante do problema.

Essa lógica pode ser reaproveitada com YOLO:

1. detectar o dohyo por visão clássica;
2. recortar ou priorizar a ROI;
3. rodar o detector de robôs;
4. filtrar detecções fora da arena;
5. passar as detecções para o tracker;
6. extrair eventos por regras geométricas.

Vantagens:

- reduz falsos positivos;
- incorpora conhecimento do domínio;
- mantém a inferência leve;
- melhora explicabilidade da pipeline.

Limitação:

- a detecção do dohyo precisa ser robusta a iluminação, perspectiva e oclusão.

Esse caminho é provavelmente o melhor compromisso para a entrega: deep learning onde precisa reconhecer robôs, visão clássica onde a geometria do esporte já é conhecida.

### 5. Anotação semi-automática com revisão manual

Para construir o dataset, ainda faz sentido usar SAM 3, SAM 2, Roboflow, CVAT ou Label Studio como ferramentas auxiliares.

O ponto é separar anotação de inferência.

Na anotação, podemos aceitar uma ferramenta pesada, lenta e semi-manual, porque o objetivo é produzir labels revisados. Na inferência final, precisamos de uma pipeline rápida, reprodutível e avaliável.

Fluxo sugerido:

1. selecionar clips representativos;
2. usar SAM 3/HuggingFace ou anotação manual para gerar máscaras iniciais;
3. converter máscaras para bounding boxes YOLO;
4. revisar manualmente os frames problemáticos;
5. treinar YOLO;
6. avaliar tracking e eventos.

Esse fluxo preserva o valor do SAM 3 sem deixar a pesquisa dependente dele.

## Entrega recomendada para esta semana

Para uma entrega curta, o escopo mais defensável é:

1. **Assumir oficialmente que SAM 3 saiu do caminho crítico.**
   Ele fica documentado como POC e ferramenta auxiliar.

2. **Implementar ou preparar o baseline YOLO + ByteTrack.**
   Mesmo com dataset pequeno, já permite medir FPS, detecção e tracking.

3. **Reaproveitar a detecção dinâmica do dohyo.**
   A ROI por frame e o filtro geométrico são resultados concretos do experimento SAM 3 e continuam úteis.

4. **Anotar um subconjunto pequeno, mas consistente.**
   Melhor ter 10 a 20 clips bem anotados e avaliáveis do que tentar processar 150 vídeos com uma pipeline instável.

5. **Reportar métricas simples.**
   Para esta fase: FPS, número de ID switches observados, exemplos de acerto/erro, e mAP se já houver split mínimo.

6. **Manter RT-DETR e BoT-SORT como próximos experimentos.**
   Eles entram quando o baseline estiver rodando.

## Conclusão

SAM 3 foi a escolha certa para explorar o problema, mas não é a escolha certa para fechar a primeira versão do Kanshigan.

Ele confirmou que os robôs são segmentáveis, que tracking temporal é possível e que a ROI do dohyo é uma peça importante da pipeline. Ao mesmo tempo, expôs limites claros: VRAM, FPS, dependência de prompt, falsos positivos, custo de manutenção e baixa previsibilidade em vídeos fora do cenário inicial.

A decisão técnica mais segura é usar SAM 3 como ferramenta de anotação semi-automática e migrar a inferência principal para uma arquitetura mais leve e treinável: YOLO + ByteTrack como baseline, YOLO + BoT-SORT como comparação de tracking, e RT-DETR + ByteTrack como alternativa baseada em transformer.

Isso deixa a pesquisa mais alinhada com a pergunta original: comparar arquiteturas de visão computacional não apenas por acurácia, mas também por viabilidade prática.

## Referências úteis

- [SAM 3 no GitHub](https://github.com/facebookresearch/sam3): documentação oficial, requisitos e exemplos de uso.
- [SAM 3 Video no HuggingFace Transformers](https://huggingface.co/docs/transformers/model_doc/sam3_video): API alternativa para inferência de vídeo com prompt textual.
- [Post da Meta sobre SAM 3 e SAM 3.1](https://ai.meta.com/blog/segment-anything-model-3/): contexto oficial do modelo e da atualização Object Multiplex.
- [Issue sobre backend-agnostic inference no repositório oficial](https://github.com/facebookresearch/sam3/issues/164): discussão sobre limitações de device no SAM 3 oficial.
