import numpy as np
import pandas as pd

def profile_dataframe(df: pd.DataFrame) -> dict:

    # 1. Общая информация о таблице
    result = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {}
    }

    # 2. Проходим по каждому столбцу
    for name in df.columns:
        col = df[name]
        clean = col.dropna()

        # 3. Определяем тип данных
        if pd.api.types.is_bool_dtype(col.dtype):
            kind = "boolean"
        elif pd.api.types.is_numeric_dtype(col.dtype):
            kind = "numeric"
        else:
            kind = "categorical"

        # 4. Основные характеристики
        missing_pct = round(float(col.isna().mean() * 100), 2) if len(col) else 0.0
        n_unique = int(col.nunique())
        flags = []

        profile = {
            "kind": kind,
            "missing_pct": missing_pct,
            "n_unique": n_unique
        }

        # 5. Проверяем основные проблемы
        if n_unique <= 1:
            flags.append("constant")

        if missing_pct > 50:
            flags.append("high_missing")

        if kind == "categorical":
            if len(clean) >= 5 and n_unique > 0.9 * len(clean):
                flags.append("high_cardinality")

        # 6. Анализ числовых столбцов
        if kind == "numeric":

            if len(clean) > 0:

                q1 = col.quantile(0.25)
                q3 = col.quantile(0.75)

                iqr = q3 - q1

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                n_outliers = int(
                    ((clean < lower) | (clean > upper)).sum()
                )

                profile["mean"] = round(float(col.mean()), 4)
                profile["std"] = round(float(col.std()), 4) if len(clean) >= 2 else None
                profile["min"] = round(float(col.min()), 4)
                profile["median"] = round(float(col.median()), 4)
                profile["max"] = round(float(col.max()), 4)
                profile["n_outliers"] = n_outliers

                if n_outliers > 0:
                    flags.append("outliers")

            else:
                profile["mean"] = None
                profile["std"] = None
                profile["min"] = None
                profile["median"] = None
                profile["max"] = None
                profile["n_outliers"] = 0

        # 7. Анализ категориальных и логических столбцов
        else:
            if len(clean) > 0:
                counts = col.value_counts()
                top_freq = int(counts.max())

                for value in clean:
                    if counts[value] == top_freq:
                        top = value
                        break

                if isinstance(top, np.generic):
                    top = top.item()

                profile["top"] = top
                profile["top_freq"] = top_freq

            else:
                profile["top"] = None
                profile["top_freq"] = 0

        # 8. Сохраняем профиль столбца
        profile["flags"] = flags
        result["columns"][name] = profile

    return result