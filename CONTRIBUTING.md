# Contributing example code

Example code in this repo is maintained by the Union team.

Before contributing tutorial material, run `pip install -r requirements.txt` and `pre-commit install` to ensure proper
formatting of the examples.

This repo is organized in the following structure:

```
examples/
  _blogs/  # blog post example code
  guides/  # exmaple code for product documentation
  tutorials/  # example code for end-to-end use cases
```

> [!NOTE]
> The `_blogs` directory is a temporary
> space for example code to be used in the Union blog posts. Once we've
> matured the testing and development process in this repo, contributors will
> ideally start developing example code in the `tutorials` or `guides`
> directories directly.

## Tutorial example format

Tutorials are end-to-end examples that focus on user-oriented applications. These
examples are meant to showcase what you can get done with Union, for example:

- training language models, time series models, classification and regression models, etc.
- processing datasets with frameworks like Spark, Polars, etc.
- scheduling retraining jobs for keeping models up-to-date.
- analyzing, modeling, and visualizing bioinformatics datasets.
- processing image data and other unstructured datasets.
- performing batch inference to generate, process, or analyze image and video data.

The ideal structure of a tutorial example is as follows:

- The example code should be in a directory like `tutorials/timeseries_forecasting`.
- The example script should be a single Python script like `timeseries_forecasting.py`.
- The example script should be in the `jupytext` [`light` format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-light-format). This script is consumed by the `unionai/docs`
  repo and converted into Markdown format according to the `light` format standard.
- The tutorial examples follow the [literate programming](https://en.wikipedia.org/wiki/Literate_programming#:~:text=Literate%20programming%20is%20a%20programming,source%20code%20can%20be%20generated) principle of interleaving code with natural language
  explanations so that the example can be consumed as-is in both the docs or by
  users who clone this repo.
- The workflow in the example script should specify reasonable default inputs so that
  it's runnable without having to specify inputs in the `unionai run` command.

> [!NOTE]
> The ideal structure above may not always be possible, but contributors should
> do their best to adhere to these guidelines.

## Run commands

Examples that we want to include in the Union documentation can be pulled
into the docs build system explicitly in the `docs/sitemap.json` file (see
the docs repo [README](https://github.com/unionai/docs/blob/main/README.md)
for more details).

For example pages that require instructions on how to run them, the
`run_commands.yaml` file needs to be updated like so:

```
<path/to/example.py>:
  - git clone https://github.com/unionai/examples
  - cd examples
  - unionai run --remote <path/to/example.py> <workflow_name> <input_flags>
```

Adding an entry like the one above will add a dropdown element on the docs
example page that tells the user how to run the code. This dropdown element
will be inserted after the first Markdown element in the `.py` example file.
