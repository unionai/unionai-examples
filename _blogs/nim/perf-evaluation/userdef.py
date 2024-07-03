################################################
# Source: https://github.com/bentoml/llm-bench #
################################################
class UserDef:
    BASE_URL = ""

    @classmethod
    def ping_url(cls):
        return f"{cls.BASE_URL}/health/ready"

    @staticmethod
    async def rest():
        import asyncio

        await asyncio.sleep(0.01)
