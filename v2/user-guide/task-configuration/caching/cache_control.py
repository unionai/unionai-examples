# {{docs-fragment cache-control}}
# Disable caching for this specific execution
run = flyte.with_runcontext(overwrite_cache=True).run(my_cached_task, data="test")
# {{/docs-fragment cache-control}}