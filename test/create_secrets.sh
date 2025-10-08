#!/bin/bash
uvx --prerelease=allow flyte --config ./config.flyte.yaml create secret openai_api_key --value $OPENAI_API_KEY
uvx --prerelease=allow flyte --config ./config.flyte.yaml create secret tavily_api_key --value $TAVILY_API_KEY
uvx --prerelease=allow flyte --config ./config.flyte.yaml create secret finnhub_api_key --value $FINNHUB_API_KEY
uvx --prerelease=allow flyte --config ./config.flyte.yaml create secret together_api_key --value $TOGETHER_API_KEY