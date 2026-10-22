# Versão final do artigo SBC: crítica em rodadas e o que ela derrubou

## Contexto

O artigo SBC ainda era o texto da pré-banca: seção "Resultados Parciais", introdução chamando a si mesma de "entrega de meio de percurso", conclusão prometendo "a base experimental para o trabalho final". A tarefa era transformá-lo na versão final do TCC.

Não existe rubrica pública da Inteli para a banca final. Usamos a da pré-banca (diário 10) como melhor aproximação: contribuição científica 35%, validação e dados 25%, metodologia e execução 25%, posicionamento científico 15%. As issues abertas da banca simulada (#8, #10, #11, #12, #14, #15) entraram como lista de perguntas a responder.

## Método: crítica antes de reescrever

Antes de mexer no texto, quatro revisores independentes leram o artigo, cada um com uma lente: banca com a rubrica, auditoria de números contra `results/`, revisor de evento SBC e editor de português. Os quatro convergiram nos mesmos bloqueios, o que deu confiança na lista. Nota ponderada estimada da versão de pré-banca: cerca de 6,2.

O que nenhum de nós tinha visto, e os revisores acharam olhando o repositório:

- O "acima de 100 FPS" foi medido num clip já recortado (1558 x 506), e o `infer.py` deixava a decodificação fora do cronômetro. O artigo dizia o contrário nas duas coisas.
- O "pico de VRAM do processo" era o cache do alocador do PyTorch, sem o contexto CUDA.
- A referência de identidade tem 59 quadros a 10 Hz, não "quadro a quadro", e 6 das 118 caixas ficam fora do recorte.
- A referência BR é uma semifinal entre dois robôs que estão no treino; a JP sai do mesmo vídeo do treino JP.
- O 0,026 do baseline COCO media, em parte, a classe errada (só "pessoa").
- "A maioria dos rounds termina em menos de um segundo" nunca foi medido.
- A metodologia descrevia uma calibração de limiares de eventos que nunca aconteceu e uma regra de início de round ("ambos os robôs") que o código não implementa ("qualquer um").
- O artigo dizia que pipeline e conjunto de dados eram públicos. O código é; dados e pesos não estão no repositório.
- A citação do SportsMOT usava um ponto de ablação como "melhor método".

## Medições que mudaram o texto

Várias críticas só se resolviam com número, e todas eram computacionais. Rodamos as medições sem anotar nada novo (diário 23). Três delas mudaram a narrativa, não só os números:

- **C2 caiu no round avaliado.** A explicação do empate entre rastreadores era "robôs idênticos, aparência sem sinal". Medido, o OSNet separa os dois robôs com AUC de 0,94, e até um histograma de cor separa. O empate vem de o movimento já bastar. O artigo agora diz isso e declara que o round não estressa C2.
- **O detector "compacto" não é o mais rápido na GPU.** Com decodificação e vídeo nativo, os dois empatam perto de 90 a 97 FPS; no recorte, o YOLOv8s é mais rápido. Na CPU, o YOLO26n dobra a velocidade. A resposta à pergunta passou a depender do hardware.
- **A fase decisiva dura cerca de um segundo, não o round.** A primeira conta deu mediana de 0,75 s do contato à saída em 14 rounds JP. A terceira rodada de crítica notou que 5 desses rounds têm rótulos de treino nunca revisados. Só nos 9 revisados, a mediana é 1,1 s (0,2 a 2,2 s), e é esse o número do artigo.

## O que mudou no artigo

- Título sem a promessa de "extração de métricas"; resumo e abstract reescritos sem eventos nem cinemática como resultado.
- Pergunta respondida explicitamente na introdução, na discussão e na conclusão.
- Metodologia com desenho de E1 a E6, métricas citadas (CLEAR MOT, IDF1, HOTA), parâmetros reais do código e a declaração de que o treino não foi revisado.
- Trabalhos relacionados com a literatura de alvos idênticos da biologia (idtracker.ai, TRex, SLEAP) e sucessores do DanceTrack, todos com DOI conferido.
- Resultados com a grade 2 x 4 com HOTA, custo por estágio, avaliação sem rótulos nos 23 rounds RC e a ablação da arena.
- Ameaças à validade como seção própria.
- Template: pt-BR, citações autor-ano, legendas de tabela acima, figuras flutuantes. Testamos três estilos antes de acertar: o Chicago duplica o nome do evento nas referências de conferência em português, e o APA deixa uma vírgula solta na data, "(2024,)", em entradas sem DOI. Ficou o Harvard (`harvard-cite-them-right`), que renderiza limpo.

## Terceira rodada

Um revisor novo, com o histórico das duas rodadas, conferiu mais de 100 números contra `results/` e achou uma discrepância (a mediana acima). Os outros achados eram omissões que um arguidor encontra no repositório: a tabela de E5 tinha perdido a coluna de falsos positivos dentro da arena, que mostra a semente 0 com 28 e 70 no BR; o E1 BR mediu o modo de imagem com o conceito "robot", não o "toy" usado no treino; as trajetórias cruas têm até duas trocas de identidade, não uma; e dois clipes da auditoria vêm de JSONs antigos cujo vídeo não achamos. Todos entraram no texto. Veredito: aceitar com ressalvas, nota ponderada estimada de 7,3, contra cerca de 6,2 da versão de pré-banca.

## Erros desta sessão

- Usei `*termo*` para itálico. Em Typst isso é negrito; todo termo estrangeiro saiu em negrito até a inspeção visual do PDF.
- A primeira reescrita do resumo e da conclusão dizia que o treino foi "revisado à mão". Não foi: só a referência e os 14 rounds de E5. A segunda rodada de crítica pegou.
- Repeti do diário 14 o "84.º All Japan Robot Sumo". O torneio começou em 1989; o título do vídeo diz "All Japan Robot Sumo Tournament 2025, Grand Final".
- Tamanho dos pesos calculado como parâmetros vezes 4 bytes (44,5 e 10 MB). Em disco são 22,5 e 5,4 MB.

## Pendências que dependem dos autores

- Publicar rótulos e pesos (release ou Zenodo) se quisermos chamar o conjunto de público. O texto atual não afirma isso.
- Regerar o `identity.txt` com o recorte corrigido é mudança de rótulo.
- E-mails dos autores no cabeçalho, se a banca pedir.
- O README ainda diz que os dados estão "frozen under DVC", o que não é verdade.

## Versão IEEE em inglês

O `docs/paper/` ainda era o rascunho do começo do projeto: inglês, TODOs no resumo, "first publicly available dataset" e três abordagens que nunca rodaram (RT-DETR + ByteTrack). Portamos o texto final da SBC para ele, em inglês, no template `charged-ieee` de duas colunas: 10 páginas, contra 19 da SBC, com a mesma bibliografia. Tabelas largas (a matriz C1 a C6, a grade detector × rastreador e a escala JP) atravessam as duas colunas; ficaram no rodapé da página para não aparecerem antes das tabelas de número menor. A figura da pipeline e a da final japonesa ganharam cópias com rótulos em inglês em `docs/paper/figures/`, e o rótulo do recorte foi corrigido para "retângulo fixo", que é o que o código faz. O CI passou a compilar a versão IEEE com `--root .`, porque ela usa caminhos absolutos do repositório.

Um revisor comparou as duas versões número a número: todos batiam. Ele achou, porém, três frases do resumo em inglês mais fortes que a fonte ("shows where the pipeline breaks"), dois caveats que a tradução perdeu (a fonte BR sem mediana de duração; a semente que fragmenta um robô) e "slow motion" lido como câmera lenta. Corrigidos.

Ao trocar os rótulos da figura da final, o placar da transmissão mostra "84th 3kg AUTO": é daí que veio o "84.º" do diário 14, que o artigo não usa mais.
