# Pestevision

Reprodução e melhoria do artigo *"Classification of Agricultural Pests Through Digital Images Using Deep Learning"* — projeto da disciplina de Processamento Digital de Imagens (PDI).

Este guia assume que você **nunca usou o `uv` antes**. Siga do início ao fim.

## 1. Instalar o `uv`

O `uv` é a ferramenta que usamos no lugar do `pip` + `venv`. Ele gerencia a versão do Python e todas as dependências do projeto automaticamente.

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Depois que terminar, feche e reabra o terminal (ou rode `source ~/.bashrc` / `source ~/.zshrc`) para o comando `uv` ficar disponível.

**Confira se funcionou:**
```bash
uv --version
```
Você deve ver algo como `uv 0.x.x`. Se aparecer "command not found", a instalação não terminou corretamente — reabra o terminal e tente de novo.

## 2. Clonar o repositório

```bash
git clone <repo-url>
cd Pestevision
```

## 3. Configurar o ambiente Python

Você **não** precisa instalar o Python manualmente, e **não** precisa criar um ambiente virtual na mão. O `uv` faz os dois automaticamente com base nos arquivos de configuração do projeto (`pyproject.toml` e `.python-version`).

Basta rodar:
```bash
uv sync
```

Isso vai:
- Baixar a versão correta do Python, caso você ainda não tenha
- Criar um ambiente virtual na pasta `.venv/` (pode ignorar essa pasta, não mexa nela)
- Instalar todas as dependências do projeto, nas versões exatas que o restante do grupo está usando

## 4. Rodando os scripts

Toda vez que for rodar um script Python do projeto, use o prefixo `uv run`. Isso garante que você está usando o ambiente do projeto, e não o Python do seu sistema.

```bash
uv run python scripts/setup_dataset.py
```

Você **não** precisa ativar nada manualmente (não precisa rodar `source .venv/bin/activate`) — o `uv run` já cuida disso.

## 5. Baixar o dataset

As imagens do dataset não ficam salvas no git (são muito grandes). Baixe e prepare os dados com:

```bash
uv run python scripts/setup_dataset.py
```

Esse comando baixa o zip do dataset, extrai em `data/raw/` e gera o manifest da divisão treino/teste em `data/splits/manifest.json`.

## 6. Adicionando uma nova dependência (se precisar)

Se o código precisar de uma nova biblioteca, não use `pip install` manualmente. Em vez disso:

```bash
uv add <nome-do-pacote>
```

Isso instala o pacote **e** registra a versão em `pyproject.toml` / `uv.lock`, para que todo mundo do grupo use a mesma versão ao rodar `uv sync` depois.

## Resumo dos comandos

| O que você quer fazer                  | Comando                          |
|------------------------------------------|-----------------------------------|
| Configurar / atualizar o ambiente         | `uv sync`                        |
| Rodar um script                           | `uv run python caminho/script.py`|
| Adicionar uma nova dependência            | `uv add <pacote>`                |
| Adicionar dependência só de desenvolvimento | `uv add --dev <pacote>`        |
| Abrir um shell Python dentro do ambiente  | `uv run python`                  |

## Problemas comuns

- **"uv: command not found"** → reabra o terminal, ou rode de novo o comando de instalação do Passo 1.
- **`uv sync` falha ao resolver dependências** → confira se está usando um sistema operacional/CPU suportado e tente rodar `uv sync` de novo; se persistir, copie o erro e peça ajuda.
- **Scripts não encontram o dataset** → confira se você rodou o Passo 5 (`setup_dataset.py`) antes de qualquer outra coisa.

## Track C: training/evaluation scaffold

The default providers use fixed seeded `torch.randn` images and random labels.
These runs validate the pipeline only; their metrics are not research results.

```bash
uv run python -m pestevision.train --epochs 3 --output-dir outputs/dummy
uv run python -m pestevision.evaluate outputs/dummy/best.pt --output outputs/dummy/test.json
uv run python -m unittest discover -s tests -v
```

Use `--config configs/base.yaml` for the shared settings and `--device cuda`
for GPU execution. Defaults are Adam (LR 0.001), at most 200 epochs,
`ReduceLROnPlateau(patience=10, factor=0.1)` and early stopping after 21
consecutive epochs without a strictly lower validation loss. PyTorch reduces
LR when the bad-epoch count **exceeds** patience (the 11th bad epoch).
LR and factor are scaffold defaults, not architecture-specific paper settings;
override `training.lr` and `training.lr_factor` in YAML.

`best.pt` contains the best validation-loss model, configuration, epoch, and
optimizer/scheduler states. `history.json` records split metrics and learning
rates per epoch. Training restores the best weights and never evaluates the
test split; the separate evaluation command loads the saved best weights.
Checkpoint resume is not implemented. Use a distinct output directory per run.

Metrics implement standard multiclass accuracy/precision/recall/F1 definitions;
exact agreement with Section 3.6 still requires checking the article.
Precision, recall and F1 have macro and true-support-weighted averages; accuracy
is a single overall multiclass fraction. Metrics use the whole split's confusion
matrix, with all configured classes included in macro averages and undefined
ratios set to zero. Loss is averaged by sample count, including partial batches.

Provider imports live in `src/pestevision/interfaces.py`; replace one line per
interface when each real implementation is ready:

```python
from .data import build_dataloaders       # already available
from .models import build_model          # contract: build_model(cfg) -> nn.Module
from .augmentations import get_transform # contract: get_transform(strategy, split)
```

The loader contract is `build_dataloaders(cfg, transforms)` returning `train`,
`val`, and `test` loaders of `(images, int64_labels)` batches. Models return raw
`[batch, num_classes]` logits. Set `num_classes` (default 12) and `strategy`
(default `no_da`) in the config. Keep class indices identical across splits and
between training and evaluation. Keep provider selection consistent when loading
a checkpoint; weights are validated by `load_state_dict`. An eventual model
factory with another signature needs a thin adapter at this boundary only.

Optional `training.min_delta` (default `0.0`) sets the minimum validation-loss
improvement that resets early stopping. Smaller improvements still update the
best checkpoint; the LR scheduler continues to use any strictly lower loss.
Invalid labels, empty batches/loaders, inconsistent batch sizes, and fractional
confusion counts raise explicit errors before producing a misleading report.
