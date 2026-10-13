# Studio: um lugar para ver e editar o gold

## Contexto

A revisão dos rounds JP (diário 21) foi feita num editor web de um arquivo só, escrito às pressas para funcionar pelo Tailscale. Ele resolveu aquele lote, mas deixou claro o que faltava: o gold está espalhado em quatro pastas, não há onde ver o conjunto inteiro, identidade A/B só existe para um round, eventos nunca foram anotados e trazer um vídeo novo exige editar YAML e rodar três scripts na ordem certa.

O Studio é a resposta: uma aplicação em `studio/` para ver, corrigir e ampliar o conjunto de dados. Esta entrada registra as decisões e o que foi verificado.

## O que os dados impuseram

Antes de desenhar qualquer tela, levantamos como os clips estão no disco. Três fatos definiram o modelo:

- **O mesmo clip aparece em vários conjuntos.** O `gold_zb01` está em cinco pastas. A chave de um item é conjunto mais clip, não o clip.
- **Só existem imagens dos quadros que o SAM rotulou.** Há buracos. Mas todo clip tem o vídeo nativo, então o Studio decodifica qualquer quadro sob demanda. Conferimos o alinhamento: o quadro extraído da posição 10 bate com o `_0010` gravado pelo SAM (PSNR de 46 dB, contra 33 a 35 dos vizinhos). Dá para anotar exatamente os quadros que o SAM pulou.
- **Os candidatos a gold BR estão em quadro inteiro.** As seis partidas `br01` a `br06` são 1920×1080, sem recorte no dohyo. O gold de avaliação é recortado. Promover uma delas exige recortar por round, e isso depende de marcas de round que agora dá para fazer.

## Decisões

**Sem banco de dados.** Os rótulos YOLO continuam em `data/annotations/<conjunto>/{images,labels}`, porque `train.py` e `evaluate.py` leem essas pastas. O que um rótulo YOLO não guarda (revisão, identidade por caixa, eventos, rounds) vai para um arquivo por clip, `studio/<clip>.json`. Um quadro rejeitado é movido para `rejected/`, e não só marcado: o treino lê o diretório, então a rejeição precisa ter efeito no sistema de arquivos.

**O que já existia é importado, não migrado.** Na leitura, o Studio lê o `review.json` do editor antigo, as marcas do `mark_rounds.py` e o `identity.txt` do gold BR. Só grava o arquivo próprio quando alguém salva. Abrir um clip não muda nada no disco.

**Os arquivos da avaliação são derivados.** A cada salvamento o Studio regenera `exports/<clip>.identity.txt` (formato MOT), `.events.json` e `.rounds.json`, nos formatos que o `evaluate.py` e o `mark_rounds.py` já usam.

**O pipeline Python não foi reescrito.** O servidor chama `segment_rounds.py`, `prep_clips.py` e `annotate_to_yolo.py` como subprocessos, um por vez por causa da GPU. A única mudança neles foi aceitar `KANSHIGAN_DATA_ROOT`, para os testes rodarem contra uma cópia dos dados.

**Stack.** Bun, Hono e Zod no servidor; React 19, TanStack Router e Query no front, a partir do template de projetos web do grupo. Os schemas são compartilhados e o cliente deriva os tipos das rotas.

**Visual.** A linguagem veio do monitor do Micras: superfície escura neutra, janelas com barra de título, números em mono tabular, chips. O layout é próprio da tarefa: biblioteca com o gold como a maior janela, e uma bancada com o quadro, a linha do tempo do clip inteiro e o inspetor.

## O que a bancada faz que o editor antigo não fazia

- **Linha do tempo do clip inteiro.** Uma coluna por quadro, uma faixa por tipo de anotação. Os buracos do SAM e os quadros sem dois robôs aparecem de relance, e `N` pula para o próximo.
- **Identidade A/B com proposta automática.** Ao avançar, a identidade do quadro anterior é proposta pelo centro mais próximo, a mesma hipótese de continuidade do `identity_gold.py`. Num cruzamento ela pode trocar os dois, e `Q` desfaz. `I` propaga até o fim do clip.
- **Eventos e rounds.** Início, primeiro contato e ring-out no quadro atual, e o intervalo de cada round.
- **Quadros sem rótulo.** Qualquer posição do vídeo pode ser anotada, não só as que o SAM gravou.

## Acesso

O servidor só escuta em `127.0.0.1`. A exposição é pelo Tailscale, ligada e desligada por comando. Duas lições do diário 21 viraram código:

- O script de exposição lê `tailscale serve status` e recusa tomar uma porta que já serve outra aplicação. Testamos contra a configuração real: ele recusou a 443.
- O Funnel não abre sem senha. O script pergunta ao servidor se ele exige login e recusa se não exigir.

Com senha, quem chega pela tailnet entra direto: o Tailscale Serve acrescenta `Tailscale-User-Login` para usuários da tailnet e remove esse cabeçalho de qualquer outro pedido, inclusive do Funnel. Conferimos na documentação e depois no tráfego real, com um servidor que só imprime os cabeçalhos. Um pedido da tailnet chega com a identidade. Um pedido de fora chega sem ela e com `Tailscale-Funnel-Request`, que a documentação não cita. O servidor usa os dois: sem senha configurada, ele recusa o que vem pelo Funnel.

Esse teste desfez um erro do diário 21. Lá a confirmação de que o link público abria "de fora" foi feita com uma ferramenta de busca que, descobrimos agora, roda de dentro da tailnet: o pedido dela chega com a identidade do dono. Aquele teste não provava acesso externo. O teste externo de verdade passou a ser feito por um serviço de terceiros.

## O que as revisões acharam

Antes do PR, o código passou por duas revisões independentes, uma de segurança e uma de correção. Nenhuma achou falha crítica, mas as duas acharam coisas que os testes não pegavam:

- **Um salvamento que falhava era esquecido.** O salvamento seguinte fazia a tela voltar a dizer "salvo". Agora o que falhou fica na frente da fila, é reenviado antes de qualquer coisa nova, e o erro fica visível com um botão para tentar de novo.
- **A identidade do gold BR sumiria ao promover um segundo clip.** A importação do `identity.txt` dependia de o conjunto ter um clip só. Agora ela é amarrada ao clip pelo nome.
- **As marcas de round antigas eram lidas na unidade errada.** O primeiro formato do `mark_rounds.py` conta quadros nativos, não quadros de anotação. Os rounds do `br01` apareciam seis vezes mais longos e bloqueavam qualquer marcação nova.
- **Navegar gravava.** Uma identidade proposta era salva só por passar pelo quadro. Agora só seguir para o próximo quadro aceita a proposta.
- **O link do YouTube podia ser burlado.** `http://youtube.com\@127.0.0.1/` passava no filtro e o `yt-dlp` buscava o outro endereço. O servidor agora reconstrói o link a partir do ID do vídeo.
- **Limites que faltavam:** duração da janela de corte, tamanho da fila, tamanho do upload, checagem do cabeçalho Host.

Cada um virou um teste.

## Verificação

- 172 testes do servidor e do front (272 contando os das regras de lint e dos scripts): leitura e escrita de clips contra um diretório temporário, importação dos formatos antigos, movimentação entre conjuntos, autenticação, fila de tarefas, lógica de edição e identidade.
- 12 testes de ponta a ponta com Playwright, contra uma cópia de uma biblioteca de exemplo versionada, incluindo varredura de acessibilidade WCAG 2.1 AA nas três telas.
- Fluxo de vídeo novo exercitado de verdade numa cópia dos dados: a detecção de rounds devolveu a mesma janela registrada no `clips.yaml` (6,8 a 17,2 s), e o corte com SAM 3 produziu 30 quadros rotulados num clip que apareceu na biblioteca como candidato.
- Com os dados reais, a biblioteca lista os 45 clips e 2.468 quadros anotados, e importa a identidade do gold BR em 55 dos 59 quadros.

## Limites

- Eventos e rounds são marcados nos quadros de anotação, a 10 por segundo. A resolução é de 100 ms; a tolerância do `evaluate.py` é de 150 ms. Marcar no vídeo nativo fica para depois.
- Os candidatos BR em quadro inteiro ainda não podem virar gold de avaliação. Falta o recorte por round.
- A lista de tarefas vive na memória do servidor.

## Pendências

- Recorte por round dos candidatos `br01` a `br06`, a partir das marcas de round, para poderem ser promovidos.
- Marcação de eventos com resolução de quadro nativo.
- Os dados continuam sem versionamento (pendência do diário 21): o Studio grava em `data/`, que o DVC ainda não rastreia.
