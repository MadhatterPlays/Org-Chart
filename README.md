# Reporting lines: org chart inference

Turns a flat list of people and job titles into an org chart, with a confidence on every reporting line.
Where a manager can't be told apart from titles, the person is placed under the nearest leader that every
plausible manager shares (a dashed group), or left unlinked with a reason. A wrong line is treated as worse than no line.

## Run the app (no build needed)

`web/index.html` is a single self-contained file with all 520 companies inlined.

```bash
./serve.sh            # then open http://localhost:8000
```

Or just double-click `web/index.html`. It loads D3 from cdnjs and falls back to `web/vendor/d3.min.js` offline.
Needs a current Chrome, Edge, Firefox or Safari (uses the built-in gzip decompressor).

## Rebuild everything from scratch

Python 3.10+ and numpy.

```bash
pip3 install -r requirements.txt
./run_pipeline.sh     # generate data, fit, calibrate, score, rebuild web/index.html (a few minutes)
```

The run is seeded, so it reproduces the same companies and numbers.

## Layout

```
pipeline/
  generate.py   520 synthetic companies with a hidden true org chart and messy records -> data/*.csv
  parse.py      reads one record: cleans the title, expands abbreviations, fixes typos,
                works out seniority, function, team and region, and keeps notes on how it read it
  infer.py      candidate managers per person, 34 pair features, conditional-logit model,
                isotonic calibration, placement (line / roll-up / unlinked) at every cut-off 50-99%
  evaluate.py   picks the default cut-off on training companies, scores held-out companies -> data/metrics.json
  build.py      packs results into web/index.html
web/
  template.html the app (vanilla JS + D3); build.py inlines the data into index.html
  index.html    the built app
  vendor/       offline copy of D3 7.8.5
data/
  profiles.csv  what the model sees: name, title, department, location, start date
  truth.csv     hidden truth: real manager, function, team, level, clean title, mess applied
  companies.csv company attributes
  metrics.json  all scores shown on the accuracy tab
```

`data/inference.pkl` (about 400 MB) is created by `infer.py` and is not shipped.

## Splits

520 companies: 300 fit the model, 100 calibrate its confidence, 120 are held out and used only for scoring.
The default cut-off (82%) is the lowest that gave at least 95% line precision on the 400 training companies.

## Held-out results at the default cut-off

| | |
|---|---|
| Drawn lines that are correct | 94.7% |
| People with a drawn line | 39.6% |
| People rolled up under a leader | 47.5% (99.1% under a true leader) |
| People left unlinked | 12.9% |
| Who runs the &lt;function&gt; team? | 96.6% correct, answered 98% |
| Who does the head of &lt;function&gt; report to? | 97.6% correct, answered 42% |
| Economic buyer above &lt;name&gt; | 98.8% correct, answered 67% |
