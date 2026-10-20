# Kanshigan Studio

Onde o conjunto de dados do Kanshigan é visto, corrigido e ampliado: o gold, os candidatos a gold, o treino e a validação, num lugar só.

Roda na máquina que tem os dados e a GPU. Não fica no ar o tempo todo: você sobe quando vai anotar e desliga depois.

## Subir

```bash
cd studio
bun install
bun run start        # builda o front e sobe em http://127.0.0.1:4310
```

O servidor só escuta em `127.0.0.1`. Para acessar de outra máquina, ligue a exposição pelo Tailscale:

```bash
bun run tailnet      # só para a tailnet, em https://<maquina>.<tailnet>.ts.net:8443
bun run funnel       # internet pública, pede senha
bun run expose:off   # tira do ar
```

O `funnel` recusa se o servidor estiver sem senha. Defina `STUDIO_PASSWORD` no `.env` (veja `.env.example`) e reinicie. Com senha, quem chega pela tailnet entra direto, porque o Tailscale Serve identifica o usuário; todo o resto passa pela tela de login.

O próprio servidor também se protege: sem senha, ele recusa qualquer pedido que chegue pelo Funnel. Um Funnel esquecido ligado não expõe um servidor aberto.

Os dois comandos recusam tomar uma porta que já serve outra aplicação. Troque com `STUDIO_PUBLIC_PORT`.

Para desenvolver: `bun run dev` sobe a API com reload e o Vite com proxy para ela.

## O que dá para fazer

**Biblioteca.** Todos os clips, agrupados por papel. O gold é a janela maior, com o total de quadros, quantos têm os dois robôs, quantos têm identidade A/B, eventos e rounds.

**Bancada.** Um clip aberto, com o quadro, a linha do tempo do clip inteiro e o inspetor.

- Caixas: arraste para mover, puxe um canto para redimensionar, arraste no vazio para criar.
- Identidade: marque qual caixa é o robô A e qual é o B. Ao avançar, o studio propõe a identidade a partir do quadro anterior, pelo centro mais próximo. Seguir para o próximo quadro aceita a proposta; voltar, pular ou sair a descarta. Num cruzamento a proposta pode trocar os dois: `Q` desfaz a troca e `I` refaz a identidade dali até o fim do clip.
- Eventos e rounds: início, primeiro contato e ring-out no quadro atual, e o intervalo de cada round.
- Quadros que o SAM 3 pulou: o studio decodifica qualquer quadro do vídeo do clip sob demanda, então dá para anotar os buracos.
- Revisão: cada quadro é pendente, aprovado ou rejeitado. Um quadro rejeitado sai de `images/` e vai para `rejected/`, então o treino não o lê.

**Novo vídeo.** Envie um arquivo ou cole um link do YouTube, detecte os rounds (ou marque no vídeo), e mande cortar e anotar. O studio corta cada round, recorta no dohyo, roda o SAM 3 e registra os clips como candidatos. As tarefas rodam uma por vez, por causa da GPU.

**Mover.** Um clip muda de conjunto pelo botão Mover na bancada. Gold, treino e validação só aceitam clips sem quadro pendente. O gold exige também o recorte do dohyo e o conjunto da fonte certa (`gold_crop` para BR, `gold_jp` para JP).

### Atalhos da bancada

| Tecla           | Faz                                                    |
| --------------- | ------------------------------------------------------ |
| `Espaço`        | play e pause                                           |
| `←` `→`         | quadro anterior e próximo (`Shift` anda 10)            |
| `Tab`           | troca a caixa selecionada                              |
| `W` `A` `S` `D` | move a caixa (`Shift` muda o tamanho)                  |
| `Delete`        | apaga a caixa                                          |
| `1` `2`         | marca a caixa como robô A ou B                         |
| `Q`             | troca A e B                                            |
| `I`             | propaga a identidade até o fim do clip                 |
| `Enter` `X`     | aprova ou rejeita o quadro e avança                    |
| `G` `N`         | próximo quadro pendente, próximo quadro sem dois robôs |
| `H` `T` `O`     | marca início, contato, ring-out                        |
| `R`             | abre e fecha um round                                  |
| `C` `Y` `V`     | limpa, copia e cola as caixas                          |

## Onde os dados ficam

O studio não tem banco. Ele lê e grava as mesmas pastas que o treino e a avaliação usam, em `data/annotations/<conjunto>/`:

| Caminho                       | Conteúdo                                                                            |
| ----------------------------- | ----------------------------------------------------------------------------------- |
| `images/`, `labels/`          | Quadros e rótulos YOLO, como sempre. `train.py` e `evaluate.py` seguem funcionando. |
| `rejected/`                   | Quadros rejeitados, fora do alcance do treino.                                      |
| `studio/<clip>.json`          | O que um rótulo YOLO não guarda: revisão, identidade por caixa, eventos, rounds.    |
| `exports/<clip>.identity.txt` | Identidade em formato MOT, para `evaluate.py --gold-mot`.                           |
| `exports/<clip>.events.json`  | Tempos dos eventos em ms, para `evaluate.py --gold-events`.                         |
| `exports/<clip>.rounds.json`  | Limites de round, no formato do `mark_rounds.py`.                                   |

Os arquivos de `exports/` são regenerados a cada salvamento.

Na primeira vez que abre um clip antigo, o studio importa o que já existia: `review.json` do editor web, as marcas de `round_marks/` e o `identity.txt` do gold BR. Ele só passa a gravar o próprio arquivo quando você salva algo.

Os quadros decodificados sob demanda ficam em `data/studio/frame-cache/` e os logs das tarefas em `data/studio/jobs/`. Os dois podem ser apagados.

## Limites conhecidos

- Eventos e rounds são marcados nos quadros de anotação, a 10 quadros por segundo. A resolução é de 100 ms; a tolerância do `evaluate.py` é de 150 ms.
- Os candidatos a gold BR antigos (`br01` a `br06`) estão em quadro inteiro e não podem ir para o gold de avaliação. Falta recortar cada round no dohyo a partir das marcas de round.
- A lista de tarefas vive na memória do servidor. Reiniciar esquece a lista, não os logs.
- Cada round cortado tem no máximo 60 s, porque o `prep_clips.py` decodifica a janela inteira na memória.
- Cinco senhas erradas em um minuto bloqueiam o login daquele endereço por um minuto.
- A identidade exportada do gold BR não é idêntica ao `identity.txt` antigo: o antigo guarda caixas que passam da borda do recorte, e os rótulos do `gold_crop` são cortados na borda. São 110 linhas contra 118. O `tracker_comparison.py` continua lendo o `identity.txt` antigo, então os números do artigo não mudam; trocar de arquivo exige recalcular.
- Dois navegadores no mesmo clip sobrescrevem um ao outro, quadro a quadro para as caixas e por inteiro para eventos e rounds.

## Stack

Bun, Hono e Zod no servidor; React 19, TanStack Router e Query, Tailwind v4 e shadcn/ui no front. Os schemas em `src/shared/` valem para os dois lados, e o cliente deriva seus tipos das rotas do servidor. O pipeline de vídeo continua em Python: o servidor chama `segment_rounds.py`, `prep_clips.py` e `annotate_to_yolo.py` como subprocessos.

## Comandos

| Comando                                     | Faz                                                                                   |
| ------------------------------------------- | ------------------------------------------------------------------------------------- |
| `bun run start`                             | Build do front e servidor                                                             |
| `bun run dev`                               | API com reload e Vite                                                                 |
| `bun run doctor`                            | Lint, formatação, tipos, código morto e testes unitários                              |
| `bun run test:e2e`                          | Playwright contra uma cópia da biblioteca de exemplo, com varredura de acessibilidade |
| `bun run tailnet` / `funnel` / `expose:off` | Exposição pelo Tailscale                                                              |

Todo texto do projeto segue o `docs/anti-slop-guidelines.md`.
