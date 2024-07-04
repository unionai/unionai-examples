# # Reddit Slack Bot
#
# This tutorial demonstrates how to set up a simple workflow to search for
# Reddit posts of interest and send them to a Slack channel. We will make use
# of `secrets` to store Reddit and Slack credentials, and `LaunchPlans` to schedule
# workflows at constant interval, so we never miss a Reddit post we might be
# interested in.

# ## Creating Secrets to Access Reddit and Slack
#
# For our workflow to access Reddit and Slack, we need to set up some credentials
# with Slack and Reddit, and share them with our workflow in a safe way. For Reddit,
# we create a [developer app](https://www.reddit.com/prefs/apps) which will give
# us a client ID and secret. We will refer to these as `reddit_client_id` and `reddit_secret_key`.
# Once we have collected our `reddit_client_id` and `reddit_secret_key`,
# we securely store them using the `unionai` CLI tool. To add `reddit_client_id` we run:
# ```bash
# unionai create secret reddit_client_id
# ```
# and paste the client ID when prompted.
#
# After, we can do the same for `reddit_secret_key`.

# For Slack, we will need a Slack API bot token which can be configured and created for
# a specific channel by following
# [Slack's documentation](https://api.slack.com/tutorials/tracks/getting-a-token). We
# will call this token `slack_token`.  Similar to how we added `reddit_client_id` and `reddit_secret_key`,
# we securely store the `slack_token` in the CLI using:
# ```bash
# unionai create secret slack_token
# ```

# Once our secrets are set up, we continue by importing some of the workflow dependencies
# and setting some constants like the frequency in which we want to run the workflow,
# the Slack channel we want to post to, and the reference names of our secrets:

import os
from datetime import datetime, timedelta
from typing import List, Dict
from flytekit import (
    task,
    workflow,
    ImageSpec,
    Secret,
    current_context,
    LaunchPlan,
    CronSchedule,
)
import requests
from requests.auth import HTTPBasicAuth

DAYS_BETWEEN_RUNS = 1
SLACK_CHANNEL_NAME = "#reddit-posts"
REDDIT_CLIENT_ID = "reddit_client_id"
REDDIT_SECRET_KEY = "reddit_secret_key"
SLACK_TOKEN = "slack_token"

# ## Defining a Container Image
#
# The previously imported packages are either in the Python Standard Library or included by
# default in the base flyte image used by Union. However, if we want to include additional
# packages we define an `ImageSpec` which will create an image under the hood, so we don't
# have to worry about writing a `Dockerfile`.

image = ImageSpec(
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"), packages=["slack_sdk==3.28.0"]
)

# ## Collecting Reddit Posts
#
# For our task, to use `reddit_client_id` and `reddit_secret_key`, we
# specify them in the `@task` decorator using the `secret_requests` argument. Then all we need
# to do is format a request and parst the result. We will only return posts since the last
# time we ran the workflow; specified by `DAYS_BETWEEN_RUNS`.

# We also make use of caching by setting `cache=True` in the `@task` decorator. By adding
# `kickoff_time` and `lookback_days` as arguments to the task we keep the function code
# independent of the current time. Now if `get_posts` ever gets called multiple times, we will
# not make unnecessary requets to the reddit API.


@task(
    secret_requests=[
        Secret(key=REDDIT_CLIENT_ID),
        Secret(key=REDDIT_SECRET_KEY),
    ],
    cache=True,
    cache_version="1.0",
)
def get_posts(
    kickoff_time: datetime, lookback_days: int, search_terms: List[str]
) -> List[Dict[str, str]]:
    """Query Reddit API for certain search terms over a specified time period.

    :param datetime kickoff_time: The time the workflow was kicked off. This represents
    the most recent time for which Reddit posts are queried.
    :param int lookback_days: Number of days before kickoff_time for which we query
    Reddit posts.
    :param List[str] search_terms: List of terms we want the Reddit posts to include.
    :return: List of recent posts.
    """
    # Load Secrets and Authenticate
    reddit_client_id = current_context().secrets.get(REDDIT_CLIENT_ID)
    reddit_secret_key = current_context().secrets.get(REDDIT_SECRET_KEY)
    auth = HTTPBasicAuth(reddit_client_id, reddit_secret_key)

    # Format and run request for reddit posts
    query_string = "+".join(search_terms)
    result = requests.get(
        f"https://www.reddit.com/search.json?q={query_string}&sort=new", auth=auth
    )
    result_json = result.json()
    posts = result_json["data"]["children"]

    # Get only recent posts
    days_ago_datetime = kickoff_time - timedelta(days=lookback_days)
    days_ago_timestamp = days_ago_datetime.timestamp()
    recent_posts = [
        {
            "title": post["data"]["title"],
            "description": post["data"]["selftext"],
            "link": post["data"]["url"],
        }
        for post in posts
        if post["data"]["created_utc"] >= days_ago_timestamp
    ]

    return recent_posts


# ## Posting to Slack
#
# Given the reddit posts returned by our previous task, we define a helper function to format them
# in a more readable format for Slack.


def format_posts(posts: List[Dict[str, str]]) -> str:
    """Format Reddit posts for readability in Slack.

    :param List[Dict[str, str]] posts: List of Reddit posts.
    :return: Human-readable string.
    """
    formatted_message = "*Recent Posts:*\n\n"
    for idx, post in enumerate(posts, 1):
        formatted_message += f"*{idx}. <{post['link']}|{post['title']}>*\n"
        formatted_message += f"{post['description']}\n\n"
    return formatted_message


# We define another task to send our posts to slack. This time we use the `ImageSpec`
# we previously wrote that contains the `slack_sdk` package.


@task(container_image=image, secret_requests=[Secret(key=SLACK_TOKEN)])
def post_slack_message(recent_posts: List[Dict[str, str]]):
    """Format Reddit posts and send them to Slack.

    :param List[Dict[str, str]] recent_posts: List of Reddit posts.
    """
    from slack_sdk import WebClient

    slack_token = current_context().secrets.get(SLACK_TOKEN)
    client = WebClient(token=slack_token)

    response = client.chat_postMessage(
        channel=SLACK_CHANNEL_NAME, text=format_posts(recent_posts)
    )
    assert response["message"]["text"]


# ## Creating the Workflow
#
# Finally, we chain these tasks together into a simple two-step workflow with some default inputs.


@workflow
def reddit_wf(
    kickoff_time: datetime = datetime(2024, 1, 1),
    lookback_days: int = DAYS_BETWEEN_RUNS,
    search_terms: List[str] = ["flyte", "ml"],
):
    """Workflow to query recent Reddit posts and send them to Slack.

    :param datetime kickoff_time: The time the workflow was kicked off. This represents
    the most recent time for which Reddit posts are queried.
    :param int lookback_days: Number of days before kickoff_time for which we query
    Reddit posts.
    :param List[str] search_terms: List of terms we want the Reddit posts to include.
    """
    recent_posts = get_posts(
        kickoff_time=kickoff_time, lookback_days=lookback_days, search_terms=search_terms
    )
    post_slack_message(recent_posts=recent_posts)


# ## Defining a Schedule
#
# Using our previously created workflow, we define a `LaunchPlan` that will
# run on a schedule defined by a standard `cron` expression. In this example we have a
# `LaunchPlan` querying for posts containing "flyte" and "ml".

LaunchPlan.get_or_create(
    reddit_wf,
    name="flyte_reddit_posts",
    default_inputs={"lookback_days": DAYS_BETWEEN_RUNS, "search_terms": ["flyte", "ml"]},
    schedule=CronSchedule(
        schedule=f"0 0 */{DAYS_BETWEEN_RUNS} * *",
        kickoff_time_input_arg="kickoff_time",
    ),
)

# To register and activate this `LaunchPLan` we run:
# ```bash
# unionai register tutorials/reddit_slack_bot/
# unionai launchplan flyte_reddit_posts --activate
# ```

# Our workflow will now run on the configured schedule until we deactivate the `LaunchPlan` either in the UI or using:
# ```bash
# unionai launchplan flyte_reddit_posts --deactivate
# ```
