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