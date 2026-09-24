# Atividade de Fixação 05 — Aplicação para análise de dados

Pipeline de Engenharia/Ciência de Dados para uma Fintech: limpeza, engenharia
temporal, filtros vetorizados bitwise, cruzamento de dados, detecção de
outliers via Z-Score e visualização.

## Estrutura

- `gerar_dados.py` — script fornecido pelo professor; gera `transacoes.csv` e `cotacoes.csv`.
- `app.py` — aplicação principal com todo o pipeline solicitado.
- `requirements.txt` — dependências.

## Como executar

```bash
pip install -r requirements.txt
python gerar_dados.py   # gera os CSVs de exemplo
python app.py           # executa o pipeline completo
```

Ao final, é gerado `relatorio_transacoes.png` com o gráfico de valor total
diário de transações e a média móvel de 7 dias.

## O que o `app.py` faz

1. **Ingestão e limpeza** — lê `transacoes.csv` (encoding `latin1`), imputa
   valores ausentes de `valor` pela **mediana por estado**, e adiciona a
   coluna estática `plataforma = 'Mobile'`.
2. **Engenharia temporal** — converte `data_transacao` para
   `datetime64[ns]`, aplica o fuso `America/Sao_Paulo`, cria `dia_semana` e
   `mes` via `.dt`, e remove duplicatas mantendo a primeira ocorrência.
3. **Filtro vetorizado bitwise** — seleciona transações de setembro, nos
   estados SP **ou** RJ, com valor **maior que R$5.000**, usando `&`/`|`.
4. **Cruzamento e agregação** — mapeia `id_cliente` para nível de risco via
   `.map()` e monta uma `pivot_table` (mês × nível de risco) com subtotais
   (`margins=True`).
5. **Z-Score vetorizado** — calcula o Z-Score do valor por estado
   (`groupby` + `transform`, sem laços `for`) e filtra transações com
   `|Z| > 2.5` como potenciais anomalias/fraudes.
6. **Visualização OO (Matplotlib)** — `fig, ax = plt.subplots()` com o valor
   total diário e a média móvel de 7 dias, eixo Y começando em zero, título
   e legendas.
