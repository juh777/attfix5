"""
Atividade de Fixação 05 - Aplicação para análise de dados
Fintech: limpeza, engenharia de dados, filtros vetorizados, detecção de
outliers (Z-Score) e visualização de métricas de transações financeiras.

Autor: Rubens Lemos da Cruz Junior
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# 1. INGESTÃO, PERFORMANCE E LIMPEZA
# --------------------------------------------------------------------------
def carregar_e_limpar(caminho_csv: str = "transacoes.csv") -> pd.DataFrame:
    """Carrega o CSV de transações, trata encoding legado e imputa NaN."""

    # O arquivo foi gerado com encoding latin1 -> leitura correta sem
    # corromper caracteres acentuados.
    df = pd.read_csv(
        caminho_csv,
        encoding="latin1",
        usecols=["id_cliente", "data_transacao", "valor", "origem", "estado_cliente"],
    )

    # Imputação de valores ausentes em 'valor' pela MEDIANA por estado
    # (vetorizado via groupby + transform, sem laços for).
    df["valor"] = df["valor"].fillna(
        df.groupby("estado_cliente")["valor"].transform("median")
    )

    # Nova coluna estática para rastreabilidade do pipeline.
    df["plataforma"] = "Mobile"

    return df


# --------------------------------------------------------------------------
# 2. ENGENHARIA DE DADOS & ALINHAMENTO TEMPORAL
# --------------------------------------------------------------------------
def enriquecer_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Converte datas, aplica fuso horário e cria colunas derivadas."""

    df = df.copy()

    # datetime64[ns] + fuso horário America/Sao_Paulo.
    df["data_transacao"] = pd.to_datetime(df["data_transacao"])
    df["data_transacao"] = df["data_transacao"].dt.tz_localize("America/Sao_Paulo")

    # Colunas derivadas via acessor .dt (vetorizado).
    df["dia_semana"] = df["data_transacao"].dt.day_name()
    df["mes"] = df["data_transacao"].dt.month

    # Remove duplicatas exatas, mantendo apenas a primeira ocorrência.
    df = df.drop_duplicates(keep="first")

    return df


# --------------------------------------------------------------------------
# 3. OPERAÇÕES VETORIZADAS E FILTROS BITWISE
# --------------------------------------------------------------------------
def filtrar_setembro_sp_rj_alto_valor(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra transações de setembro, em SP ou RJ, com valor > R$5000,
    usando obrigatoriamente operadores bitwise (&, |) de forma vetorizada."""

    filtro = (
        (df["mes"] == 9)
        & ((df["estado_cliente"] == "SP") | (df["estado_cliente"] == "RJ"))
        & (df["valor"] > 5000.00)
    )
    return df[filtro]


# --------------------------------------------------------------------------
# 4. CRUZAMENTO DE DADOS & AGREGAÇÃO
# --------------------------------------------------------------------------
# Dicionário de risco por cliente (id_cliente -> nível de risco).
RISCO_DICT = {
    "C100": "Baixo",
    "C101": "Alto",
    "C102": "Médio",
    "C103": "Baixo",
    "C104": "Alto",
}


def aplicar_risco_e_pivot(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Atribui nível de risco via .map() e cria tabela dinâmica
    (mês x nível de risco), somando o valor, com subtotais (margins=True)."""

    df = df.copy()
    df["nivel_risco"] = df["id_cliente"].map(RISCO_DICT)

    pivot = pd.pivot_table(
        df,
        values="valor",
        index="mes",
        columns="nivel_risco",
        aggfunc="sum",
        margins=True,
        margins_name="Total",
    )
    return df, pivot


# --------------------------------------------------------------------------
# 5. DETECÇÃO ESTATÍSTICA DE OUTLIERS (Z-SCORE)
# --------------------------------------------------------------------------
def calcular_zscore_por_estado(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula o Z-Score do valor da transação por estado, de forma
    totalmente vetorizada (sem laços for), usando groupby + transform."""

    df = df.copy()
    media_estado = df.groupby("estado_cliente")["valor"].transform("mean")
    desvio_estado = df.groupby("estado_cliente")["valor"].transform("std")

    df["zscore"] = (df["valor"] - media_estado) / desvio_estado
    return df


def detectar_anomalias(df: pd.DataFrame, limiar: float = 2.5) -> pd.DataFrame:
    """Filtra transações cujo |Z-Score| > limiar (potenciais fraudes)."""
    return df[df["zscore"].abs() > limiar]


# --------------------------------------------------------------------------
# 6. VISUALIZAÇÃO GRÁFICA (Matplotlib OO)
# --------------------------------------------------------------------------
def gerar_grafico(df: pd.DataFrame, caminho_saida: str = "relatorio_transacoes.png"):
    """Gera gráfico com o valor total diário de transações e a média
    móvel de 7 dias, usando a API Orientada a Objetos do Matplotlib."""

    serie_diaria = (
        df.set_index("data_transacao")["valor"]
        .resample("D")
        .sum()
        .sort_index()
    )
    media_movel_7d = serie_diaria.rolling(7).mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(serie_diaria.index, serie_diaria.values,
            label="Valor total diário", color="#4C72B0", linewidth=1.2)
    ax.plot(media_movel_7d.index, media_movel_7d.values,
            label="Média móvel (7 dias)", color="#C44E52", linewidth=2)

    ax.set_ylim(bottom=0)
    ax.set_title("Valor Total de Transações Diárias e Média Móvel (7 dias)")
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor total (R$)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)

    return caminho_saida


# --------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# --------------------------------------------------------------------------
def main():
    print("=== 1. Ingestão e limpeza ===")
    df = carregar_e_limpar("transacoes.csv")
    print(f"Linhas carregadas: {len(df)} | Nulos em 'valor': {df['valor'].isna().sum()}")

    print("\n=== 2. Engenharia temporal ===")
    df = enriquecer_temporal(df)
    print(f"Linhas após remoção de duplicatas: {len(df)}")
    print(df[["data_transacao", "dia_semana", "mes"]].head())

    print("\n=== 3. Filtro bitwise (setembro, SP/RJ, valor > 5000) ===")
    df_filtrado = filtrar_setembro_sp_rj_alto_valor(df)
    print(f"Transações que atendem ao filtro: {len(df_filtrado)}")

    print("\n=== 4. Nível de risco + tabela dinâmica ===")
    df, pivot = aplicar_risco_e_pivot(df)
    print(pivot)

    print("\n=== 5. Z-Score e detecção de anomalias ===")
    df = calcular_zscore_por_estado(df)
    df_anomalias = detectar_anomalias(df, limiar=2.5)
    print(f"Transações anômalas (|Z| > 2.5): {len(df_anomalias)}")
    print(df_anomalias[["id_cliente", "estado_cliente", "valor", "zscore"]].head())

    print("\n=== 6. Gráfico ===")
    caminho = gerar_grafico(df)
    print(f"Gráfico salvo em: {caminho}")

    return {
        "df": df,
        "df_filtrado": df_filtrado,
        "pivot": pivot,
        "df_anomalias": df_anomalias,
    }


if __name__ == "__main__":
    main()
