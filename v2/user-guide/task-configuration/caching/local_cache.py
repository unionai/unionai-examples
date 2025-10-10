# {{docs-fragment local-cache}}
# Local execution uses ~/.flyte/local-cache/
flyte.init()  # Local mode
result = flyte.run(my_cached_task, data="test")
# {{/docs-fragment local-cache}}
