# Pestevision

Reprodução e melhoria do artigo *"Classification of Agricultural Pests Through Digital Images Using Deep Learning"* (RITA, 2025) — projeto da disciplina de Processamento Digital de Imagens (PDI).

O que já está pronto, o que falta e as ideias para a melhoria estão em [`docs/roadmap.md`](docs/roadmap.md).

Este guia assume que você **nunca usou o `uv` antes**. Siga do início ao fim.

## 1. Instalar o `uv`

O `uv` é a ferramenta que usamos no lugar do `pip` + `venv`. Ele gerencia a versão do Python e todas as dependências do projeto.

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Feche e reabra o terminal e confira:
```bash
uv --version
```
Se aparecer "command not found", a instalação não terminou: reabra o terminal e tente de novo.

## 2. Clonar o repositório

```bash
git clone <repo-url>
cd Pestevision
```

## 3. Configurar o ambiente

Você **não** precisa instalar o Python nem criar um ambiente virtual: o `uv` faz os dois com base no `pyproject.toml`, no `.python-version` e no `uv.lock`.

```bash
uv sync
```

Isso baixa o Python 3.12 (se necessário), cria a pasta `.venv/` (não mexa nela) e instala as dependências nas **versões exatas do `uv.lock`**, as mesmas que o resto do grupo usa. O PyTorch vem do índice **CUDA 12.6** (configurado no `pyproject.toml`); na primeira vez o download tem alguns GB.

## 4. Conferir a GPU

```bash
uv run python scripts/verify_cuda.py
```

O esperado é uma versão terminando em `+cu126`, `CUDA available: True`, o nome da sua placa e **nenhum** bloco longo de avisos. Isso foi verificado numa GTX 1060; quem tem outra placa deve rodar o comando e avisar o grupo se algo diferente aparecer.

- Versão terminando em `+cu130`, ou aviso dizendo que a GPU "não é compatível": rode `uv sync` de novo e avise o grupo.
- `CUDA available: False`: o treino roda na CPU, o que é muito lento. Serve só para depurar.

## 5. O dataset

As imagens já estão versionadas em `data/raw/` (uma pasta por classe, 12 classes, 5.494 imagens). Não é preciso baixar nada.

A divisão treino/teste está em `data/split/manifest_seed42.json` (4.395 treino / 1.099 teste, com as mesmas contagens da Tabela 1 do artigo). O manifest guarda só nomes de arquivo, não cópias das imagens.

O manifest **não** tem validação. Os 20% de validação (estratificados, tirados do treino, como na Seção 3.4 do artigo) são calculados ao carregar os dados, em `src/pestevision/data.py`, com seed 42: o resultado é sempre o mesmo, em qualquer máquina (3.517 treino / 878 validação).

Para conferir **os arquivos** (a partir da raiz do repositório; leva alguns segundos):
```bash
uv run python scripts/check_data.py
```
Ele confere que todo arquivo do manifest existe em `data/raw` (com a grafia exata, inclusive maiúsculas e minúsculas), que não há arquivos sobrando (como `Thumbs.db`), que as contagens batem com a Tabela 1 do artigo e que todas as imagens abrem sem erro. Termina com "All checks passed." ou lista os problemas. Rode depois de clonar, ou se o treino se comportar de forma estranha.

Para conferir **a divisão treino/validação/teste**:
```bash
uv run python -m pestevision.data
```
Saída esperada: `PestDataset(split='train', n_samples=3517 ...)`, `val ... 878`, `test ... 1099`, `train/val overlap: 0` e uma amostra com shape `(3, 224, 224)`.

**Não** gere o manifest de novo: ele é a referência de todos os experimentos. Para só conferir que o script de geração ainda reproduz o manifest versionado:
```bash
uv run python scripts/build_manifest.py --check
```
(O script se recusa a sobrescrever um manifest existente, a menos que use `--force`.)

## 6. Testes

```bash
uv run python -m unittest discover -s tests -v
```
Devem passar os 18 testes (treino/avaliação com dados falsos, transformações de imagem e previsões por imagem). Não precisam de GPU nem do dataset.

## 7. Treinando

Sempre a partir da raiz do repositório, com o prefixo `uv run`.

**a) Teste de encanamento, com dados falsos (segundos):**
```bash
uv run python scripts/run_training.py --provider dummy --epochs 2 --arch alexnet --strategy no_da
```
A acurácia deve ficar em ~0,083 (1 em 12): os dados são aleatórios.

**b) Execução real curta, para ver se aprende:**
```bash
uv run python scripts/run_training.py --arch resnet50 --strategy no_da --epochs 3
```

**c) Uma execução real completa (só para quando o early stopping decidir):**
```bash
uv run python scripts/run_training.py --arch resnet50 --strategy no_da
```

**d) O grid completo, 4 arquiteturas × 4 estratégias = 16 execuções:**
```bash
uv run python scripts/run_training.py
```

| Opção | O que faz |
|---|---|
| `--arch A [B ...]` | só estas arquiteturas: `alexnet`, `resnet50`, `efficientnet_b4`, `vit_b_16` |
| `--strategy S [T ...]` | só estas estratégias: `no_da`, `da1`, `da2`, `da3` |
| `--device cuda` / `cpu` | padrão: `cuda` se disponível |
| `--epochs N` | limita as épocas (**execução de teste**, ver abaixo) |
| `--batch-size N` | troca o batch size da Tabela 2 (use se faltar memória da GPU; é um desvio do artigo, registre no relatório) |
| `--num-workers N` | processos que carregam as imagens (padrão 4) |
| `--force` | refaz uma execução mesmo que já exista resultado |
| `--summary-only` | só reconstrói o `summary.csv` |

**Hiperparâmetros.** Não rodamos a busca em grade do artigo. Usamos os melhores valores publicados na Tabela 2, que estão em `config/base.yaml` (`hyperparameters`): AlexNet (lr 1e-5, batch 128), ResNet-50 (1e-5, 16), EfficientNet-B4 (1e-4, 16), ViT-B/16 (1e-5, 32). Otimizador Adam, LR reduzido após 10 épocas sem melhora na loss de validação (o PyTorch reduz na 11ª época ruim), early stopping após 21 épocas sem melhora, e os melhores pesos são restaurados ao final.

**Onde ficam os resultados**
- `outputs/results/<arch>__<estrategia>/` com `best.pt`, `history.json`, `results.json` (métricas de validação e teste, matriz de confusão) e `predictions_val.csv` / `predictions_test.csv`.
- `outputs/results/summary.csv`: uma linha por execução, reconstruída a cada execução.
- Execuções de **teste** (`--provider dummy` ou `--epochs`) vão para `outputs/scratch/` e nunca se misturam com os resultados reais. A pasta `outputs/` não vai para o git.
- Uma execução cujo `results.json` já existe é **pulada**; é assim que se reinicia um grid interrompido. Retomar *dentro* de uma execução não é suportado.
- Se a GPU ficar sem memória, o script avisa, pula aquela execução e segue com as outras.

**Dividindo o grid entre máquinas.** Cada pessoa roda uma parte (por exemplo `--arch resnet50`). Depois, copie as pastas `<arch>__<estrategia>` para um único `outputs/results/` e rode `uv run python scripts/run_training.py --summary-only`.

**Previsões por imagem.** Cada `predictions_*.csv` tem uma linha por imagem: caminho (relativo a `data/raw`), classe verdadeira, classe prevista, `correct` (0/1), `confidence` e a probabilidade de cada uma das 12 classes. Servem para listar as imagens que o modelo errou e para desenhar curvas precisão-revocação (Figura 3 do artigo). Abra com `pandas.read_csv`.

**Avaliar um checkpoint já treinado** (também serve para gerar as previsões de uma execução antiga, feita antes deste recurso):
```bash
uv run python -m pestevision.evaluate outputs/results/resnet50__no_da/best.pt --split test --predictions outputs/results/resnet50__no_da/predictions_test.csv
```

## 8. Estrutura do projeto

```
Pestevision
├── config
│   ├── base.yaml              # caminhos, treino, Tabela 2 (lr e batch por arquitetura)
│   └── augmentation           # no_da / da1 / da2 / da3 (um YAML por estratégia)
├── data
│   ├── raw/<12 classes>/      # imagens originais
│   └── split/manifest_seed42.json
├── docs                       # roadmap.md e o guia dos testes de treino
├── scripts
│   ├── run_training.py        # roda o grid de experimentos
│   ├── build_manifest.py      # confere (--check) / gera manifests
│   ├── check_data.py          # confere data/raw contra o manifest e a Tabela 1
│   └── verify_cuda.py
├── src/pestevision
│   ├── data.py                # Dataset, split de validação, DataLoaders
│   ├── augmentation/          # builder, config e get_transform(strategy, split)
│   ├── models.py              # get_model(nome, num_classes)
│   ├── train.py               # loop de treino (LR plateau, early stopping)
│   ├── evaluate.py            # métricas (acc, precision, recall, F1; macro e weighted)
│   ├── predictions.py         # previsões por imagem (CSV com probabilidades)
│   ├── interfaces.py          # liga dados + modelo + transformações reais ou dummy
│   └── dummy.py               # dados falsos para testes de encanamento
└── tests
```

## 9. Adicionando uma nova dependência

Não use `pip install`. Use:
```bash
uv add <nome-do-pacote>
```
Isso instala o pacote **e** registra a versão em `pyproject.toml` / `uv.lock`. Faça commit dos dois arquivos para o grupo todo usar a mesma versão.

## 10. Resumo dos comandos

| O que você quer fazer | Comando |
|---|---|
| Configurar / atualizar o ambiente | `uv sync` |
| Conferir a GPU | `uv run python scripts/verify_cuda.py` |
| Rodar os testes | `uv run python -m unittest discover -s tests -v` |
| Conferir os arquivos do dataset | `uv run python scripts/check_data.py` |
| Conferir a divisão treino/val/teste | `uv run python -m pestevision.data` |
| Treinar (um experimento) | `uv run python scripts/run_training.py --arch resnet50 --strategy no_da` |
| Treinar (grid completo) | `uv run python scripts/run_training.py` |
| Rodar um script qualquer | `uv run python caminho/script.py` |
| Adicionar dependência | `uv add <pacote>` |

## 11. Onde nos afastamos do artigo

- **Versões:** o artigo usou Python 3.9, PyTorch 2.0.1 e CUDA 10.1, que não são mais instaláveis. Usamos Python 3.12 e PyTorch com CUDA 12.6 (versões exatas no `uv.lock`).
- **Sem busca em grade:** valores fixos da Tabela 2 (seção 7).
- **Validação:** os 20% são tirados do treino em tempo de carga, com seed fixa (seção 5), porque o manifest só tem treino e teste.
- **Pré-processamento (o artigo não detalha):** treino usa as transformações da estratégia e depois normalização ImageNet; validação e teste usam apenas redimensionamento para 224×224 e a mesma normalização. O `no_da` usa `RandomResizedCrop` com escala 0,8–1,0.

## Problemas comuns

- **`uv: command not found`** → reabra o terminal ou repita o passo 1.
- **`uv sync` falha ao resolver** → copie o erro inteiro e mande para o grupo.
- **Aviso enorme sobre "compute capability" / GPU não compatível** → veja o passo 4.
- **`CUDA out of memory`** → rode com `--batch-size` menor (e registre o desvio), ou use a máquina com mais memória de vídeo.
- **`Ran 0 tests` ou `No module named 'pestevision'`** → rode a partir da raiz do repositório e com `uv run`.
- **O resultado de um teste "sumiu" ou foi pulado** → execuções com `--epochs`/`--provider dummy` ficam em `outputs/scratch/`; execuções reais já concluídas são puladas (use `--force`).
