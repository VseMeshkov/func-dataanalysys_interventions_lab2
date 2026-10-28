# Презентация ЛР2 в LaTeX

Beamer, тема Madrid, цветовая схема seahorse, 16:9 — по предоставленному примеру.
15 слайдов; сценарий встроен в `\note{...}` после каждого слайда.
ФИО и группу замените в `\author` и `\institute`.

## Сборка

```bash
pdflatex -interaction=nonstopmode -halt-on-error lab2_presentation.tex
pdflatex -interaction=nonstopmode -halt-on-error lab2_presentation.tex
```

В Overleaf загрузите содержимое папки `lab2_latex`, выберите **pdfLaTeX** и основной файл
`lab2_presentation.tex`. Сохраните папку `figures` рядом с исходником.
Для русского языка используются inputenc, T2A и babel, как в примере.
Также поддерживается LuaLaTeX с fontspec и шрифтом DejaVu Sans.

Версия для докладчика со сценарием справа:

```bash
pdflatex -jobname=lab2_presenter '\def\shownotes{1}\input{lab2_presentation.tex}'
pdflatex -jobname=lab2_presenter '\def\shownotes{1}\input{lab2_presentation.tex}'
```

`SPEAKER_NOTES.md` — отдельный сценарий примерно на 10 минут.
Числа взяты из результатов лабораторной; модель из примера про волны не переносилась.
