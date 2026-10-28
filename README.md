# Лабораторная №2 — вариант 7–1–4

Основная работа — [lab2.ipynb](lab2.ipynb): выполненный эксперимент, графики, таблицы и выводы.
Методы: Sequential Monte Carlo, зануление входа, статистика Колмогорова — Смирнова.

- [LAB2_REPORT.md](LAB2_REPORT.md) — теория и подробные результаты.
- [lab2_latex/lab2_presentation.pdf](lab2_latex/lab2_presentation.pdf) — презентация.
- [lab2_latex/lab2_presenter.pdf](lab2_latex/lab2_presenter.pdf) — слайды с заметками докладчика.
- [lab2_latex/SPEAKER_NOTES.md](lab2_latex/SPEAKER_NOTES.md) — сценарий выступления.
- `lab2_latex/lab2_presentation.tex` и `lab2_latex/figures/` — исходники LaTeX; [инструкция сборки](lab2_latex/README.md).
- `lab2_analysis.py`, `lab2_report.py`, `oscillators.py` — код, необходимый ноутбуку.
- `data/invar_datasets.zip` — исходные данные для запуска без интернета.
- `lab2_results/` — результаты и диагностика; нужны также для пересборки презентации.
- `tools/` — сборка ноутбука и LaTeX из результатов.
- `test_lab2_analysis.py` — проверки SMC и анализа.

## Запуск

Из каталога `lab1_stepanov`:

```bash
pip install -r requirements.txt
jupyter lab lab2.ipynb
```

В Jupyter выполните **Run All**. Альтернатива:

```bash
OPENBLAS_NUM_THREADS=1 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 lab2.ipynb
pytest -q test_lab2_analysis.py
```

Повторный запуск обновляет отчёт, результаты, LaTeX-исходник и сценарий.
PDF нужно собрать отдельно по инструкции в `lab2_latex/README.md`.
Только пересобрать LaTeX-исходник из сохранённых результатов:

```bash
python tools/build_lab2_latex.py
```

`tools/build_lab2.py` пересоздаёт ноутбук и очищает его выводы; запускайте его только при изменении структуры работы.
Перед сдачей заполните ФИО и группу в отчёте и на первом слайде.
