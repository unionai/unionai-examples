# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b25",
# ]
# main = "main"
# params = ""
# ///

import flyte
import flyte.report

env = flyte.TaskEnvironment(name="reports_example")


@env.task(report=True)
async def task1():
    await flyte.report.replace.aio("<p>The quick, brown fox jumps over a lazy dog.</p>")
    tab2 = flyte.report.get_tab("Tab 2")
    tab2.log("<p>The quick, brown dog jumps over a lazy fox.</p>")
    await flyte.report.flush.aio()


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(task1)
    print(r.name)
    print(r.url)
    r.wait()