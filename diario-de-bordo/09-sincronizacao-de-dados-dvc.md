# Sincronização de Dados com DVC

## Contexto

Os vídeos das partidas — tanto os brutos quanto os processados (cropados, resampleados, anotados) — estavam fora de qualquer fluxo de versionamento. Cada autor mantinha cópias locais, sem garantia de que estávamos olhando para o mesmo arquivo num mesmo experimento. Sem sincronização, qualquer comparação entre runs de inferência fica comprometida: dois "annotated_rumble_ironcup.mp4" podem ter origens diferentes.

A pasta de vídeos no Google Drive ([link](https://drive.google.com/drive/folders/1NxH5M58lXbKg1zyfgppBcai-rWzo17Zl)) já existia como armazenamento informal, mas nada no repositório apontava pra ela.

## Opções consideradas

| Opção | Versionamento | Casa com git | Custo | Veredito |
|---|---|---|---|---|
| Git LFS | Sim | Nativo | GitHub: 1GB free + 1GB/mês banda | Estoura limite rápido |
| rclone + Google Drive | Não | Manual | Free | Funciona mas sem rastreio por commit |
| Google Drive Desktop | Não | Não | Free | Erro humano alto, sem auditoria |
| DVC + Google Drive | Sim, atrelado a commits | Sim | Free (reaproveita o GDrive) | Escolhido |

O `.gitignore` original já tinha o comentário `# Data (tracked via DVC)` apontando pra `data/raw/` e `data/processed/`, então a direção já estava semi-decidida — só não tinha sido configurada.

## Por que DVC

- **Reprodutibilidade.** Cada commit aponta pra uma versão exata dos dados via arquivos `.dvc` (pequenos, JSON com hash SHA256). Voltar num commit antigo e rodar `dvc checkout` materializa os vídeos exatos daquela versão.
- **Reaproveita o Google Drive.** O remote do DVC pode ser exatamente a pasta que já usávamos. Não trocamos de storage, ganhamos uma camada de versionamento.
- **Workflow declarativo.** `git pull && dvc pull` traz código + dados juntos. `dvc add` + `dvc push` versionam um vídeo novo da mesma forma que `git add` + `git push` versionam código.
- **Sem cap de banda.** Diferente do GitHub LFS, o Google Drive não tem cota de bandwidth pra leitura, só limite de API por OAuth client.

## Estrutura criada

```
.dvc/
├── config              # remote gdrive default → folder ID 1NxH5M58lXbKg1zyfgppBcai-rWzo17Zl
├── .gitignore          # ignora cache/tmp/config.local
└── tmp/
.dvcignore
data/
├── README.md
├── raw/                # vídeos originais (imutáveis)
└── processed/          # derivados (cropados, anotados, resampleados)
```

O `.gitignore` original tinha `*.dvc` na lista de ignorados — exatamente o oposto do necessário, já que os arquivos `.dvc` são justamente os pointers que precisam ir pro git. Foi corrigido pra ignorar o conteúdo de `data/` mas manter os pointers e o `.gitignore` interno gerado pelo DVC:

```
/data/raw/**
/data/processed/**
!/data/raw/.gitkeep
!/data/processed/.gitkeep
!/data/**/*.dvc
!/data/**/.gitignore
```

## Setup (uma vez por colaborador)

### 1. Instalar o DVC

Como CLI standalone via uv (o repo tem múltiplos `pyproject.toml` aninhados, então não vale colocar como dependência de um deles):

```bash
uv tool install 'dvc[gdrive]'
dvc --version    # confirma 3.x
```

Alternativas: `pipx install 'dvc[gdrive]'` ou `brew install dvc` (com `pip install pydrive2` separado).

### 2. Criar OAuth client no Google Cloud

O OAuth client público padrão do DVC tem rate-limit baixo demais pra uso real. Doc oficial pra referência: [DVC – Using a custom Google Cloud project](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended).

Só **um de nós** precisa fazer esse passo — o `client_id` e `client_secret` resultantes não são segredos no fluxo de Desktop app e podem ser compartilhados via canal privado pra equipe colar no próprio `.dvc/config.local`. A autenticação real acontece no login Google de cada um (passo 4).

Caminho atual no console:

#### 2.1. Criar projeto

1. Acessa https://console.cloud.google.com/.
2. Topo da página, ao lado do logo "Google Cloud", clica no seletor de projeto → **Novo Projeto**.
3. Nome: `kanshigan-dvc` (ou qualquer coisa). Cria.
4. Garante que o projeto novo tá selecionado no seletor antes de seguir.

#### 2.2. Habilitar Google Drive API

1. Menu lateral → **APIs e serviços** → **Biblioteca**.
2. Busca por "Google Drive API", clica no resultado.
3. Botão **Ativar**.

#### 2.3. Criar credenciais (wizard)

1. Menu lateral → **APIs e serviços** → **Credenciais**.
2. Topo, **+ Criar credenciais** → **ID do cliente OAuth**.
3. O console força configurar a tela de consentimento primeiro. Clica em **Configurar tela de consentimento**.

Na sequência o wizard pergunta:

**API que você está usando** → seleciona **Google Drive API**.

**Que dados você acessará** → **Dados do usuário** (cria OAuth client).
> Não escolhe "Dados do aplicativo". Aquilo cria uma Service Account, que só funciona com Shared Drives de Google Workspace. A pasta de vídeos é Drive pessoal, então precisa de fluxo OAuth de usuário.

**Informações do app:**
- Nome do app: `kanshigan-dvc` (aparece na tela de consentimento na hora do login — pode ser qualquer coisa).
- Email de suporte do usuário: o seu.
- Email do desenvolvedor: o seu.

**Escopos:** pula (clica **Salvar e continuar**). O DVC pede os escopos em runtime.

**Tipo de cliente OAuth:** **Aplicativo para computador**. Nome: qualquer (ex. `dvc-cli`).

Tela final mostra **ID do cliente** e **Chave secreta do cliente**. Copia os dois.

#### 2.4. Adicionar test users

O app fica em modo "Testing" (não precisa publicar). Só emails listados como test user conseguem autenticar — sem isso, o login retorna "access blocked".

1. **APIs e serviços** → **Tela de permissão OAuth** (ou "OAuth consent screen").
2. Seção **Usuários de teste** → **+ Adicionar usuários**.
3. Adiciona os emails Google de todo mundo do time (incluindo o seu).

#### 2.5. Configurar localmente

Cada colaborador, na própria máquina:

```bash
dvc remote modify gdrive --local gdrive_client_id     '<client-id>'
dvc remote modify gdrive --local gdrive_client_secret '<client-secret>'
```

Vai pra `.dvc/config.local`, que é gitignored.

### 3. Instalar os git hooks

```bash
dvc install
```

Configura três hooks no `.git/hooks/`:

| Hook | Comportamento | Por que |
|---|---|---|
| `post-checkout` | Roda `dvc checkout` após `git checkout` | Trocar de branch já sincroniza os dados |
| `post-merge` | Roda `dvc checkout` após `git pull`/`merge` | Mesmo motivo, no fluxo de pull |
| `pre-push` | Roda `dvc push` antes de `git push` | Evita pointer no git apontando pra dado que ainda não tá no remote |

Hooks são locais — cada colaborador roda `dvc install` na sua clone uma vez.

Decisão consciente: o `post-merge` roda `dvc checkout` (relinka do cache local, instantâneo), **não** `dvc pull` (baixa do remote, lento e consome quota). Quando o `git pull` traz `.dvc` files novos ou modificados, é responsabilidade de quem puxou rodar `dvc pull` manual. O motivo: `dvc pull` automático em toda merge derruba o limite de API do OAuth client e quebra workflow offline.

### 4. Primeiro pull

```bash
dvc pull
```

Abre o navegador pro fluxo OAuth na primeira vez. Token fica cacheado em `~/.cache/pydrive2fs/` e os comandos seguintes ficam silenciosos.

## Workflow do dia a dia

### Pull

```bash
git pull            # post-merge hook roda dvc checkout sozinho
dvc pull            # só se vc viu .dvc files novos/modificados na merge
```

### Adicionar um vídeo novo

```bash
cp ~/Downloads/match_ironcup_final.mp4 data/raw/
dvc add data/raw/match_ironcup_final.mp4
git add data/raw/match_ironcup_final.mp4.dvc data/raw/.gitignore
git commit -m "data(raw): add ironcup final match recording"
git push            # pre-push hook roda dvc push sozinho antes
```

O `.dvc` pointer (algumas centenas de bytes) é o que entra no git. O vídeo em si vai pro Google Drive.

### Trocar de branch ou voltar num commit antigo

O `post-checkout` hook chama `dvc checkout` automaticamente. Se faltar arquivo no cache local (porque foi adicionado em outra máquina e nunca foi puxado aqui), aí roda `dvc pull` na mão.

## Convenções

- `data/raw/` é imutável. Vídeo "mudou" → trata como arquivo novo com nome diferente.
- `data/processed/` guarda só o que é caro recomputar ou importante pra reprodutibilidade.
- Mudanças de código e dos `.dvc` pointers vão no mesmo commit quando se referem ao mesmo experimento.

## Decisões pendentes

- **Vídeos atuais em `experiments/sam3-poc/output/`**: são derivados regeneráveis ou queremos versionar? Duas opções: ignorar a pasta (cada um regenera localmente) ou mover pra `data/processed/` e `dvc add`.
- **Vídeos brutos**: precisam ser uploadados pra estrutura `data/raw/` antes do primeiro `dvc push` valer alguma coisa pros colegas.

## Troubleshooting

- **`Failed to authenticate GDrive`** → credenciais OAuth ausentes ou inválidas. Refaz o passo 2 do setup.
- **`Quota exceeded`** → rate-limit do seu OAuth client. Espera alguns minutos ou cria um novo.
- **`dvc pull` reclama de "missing data"** → o arquivo não foi pushado ainda, ou você não tem acesso à [pasta compartilhada](https://drive.google.com/drive/folders/1NxH5M58lXbKg1zyfgppBcai-rWzo17Zl).
