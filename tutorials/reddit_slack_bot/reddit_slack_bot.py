# # Reddit Slack Bot
#
# This tutorial demonstrates how to set up a simple workflow to search for
# Reddit posts of interest and send them to a Slack channel. We will make use
# of `secrets` to store Reddit and Slack credentials, and `LaunchPlans` to schedule
# workflows at constant interval, so we never miss a Reddit post we might be
# interested in.

# ## Overview
#
# For our workflow to access Reddit and Slack, we need to set up some credentials
# with Slack and Reddit, and share them with our workflow in a safe way. For Reddit,
# we can create a [developer app](https://www.reddit.com/prefs/apps) which will give
# us a client ID and secret. We will refer to these as `reddit_client_id` and `reddit_secret_key`.
# For Slack, we will need a Slack API bot token which can be configured and created for
# a specific channel by following
# [Slack's documentation](https://api.slack.com/tutorials/tracks/getting-a-token). We
# will call this token `reddit_slack_token`.
#
# Once we have collected our `reddit_client_id`, `reddit_secret_key`, and `reddit_slack_token`,
# we can securely store them using the `unionai` CLI tool. To add `reddit_client_id` we can run:
# ```bash
# unionai create secret reddit_client_id
# ```
# and paste the client ID when prompted.
#
# The same can then be done for `reddit_secret_key`, and `reddit_slack_token`.
#
# Once there are set up, can can continue by importing some of the workflow dependencies
# and setting some constatnts like the frequency in which we want to run the workflow and
# the Slack channel we want to post to:

from datetime import datetime, timedelta
from typing import List, Dict

from flytekit import task, workflow, ImageSpec, Secret, current_context, LaunchPlan, CronSchedule
import requests
from requests.auth import HTTPBasicAuth

WEEKS_BETWEEN_RUNS = 1
SLACK_CHANNEL_NAME = '#reddit-posts'

# ## Defining a Container Image
#
# The previously imported packages are either in the Python Standard Library or included by
# default in the base flyte image used by Union. However, if we want to include additional
# packages we can define an `ImageSpec` which will create an image under the hood, so we don't
# have to worry about writing a `Dockerfile`.

image = ImageSpec(
    builder="unionai",
    packages=[
            "slack_sdk==3.28.0",
        ]
)

# ## Collecting Reddit Posts
#
# To allow for our task to use our `reddit_client_id` and `reddit_secret_key`, we simply
# specify them in the `@task` decorator using the `secret_requests` argument. Then all we need
# to do is format a request and parst the result. We will only return posts since the last
# time we ran the workflow; specified by `WEEKS_BETWEEN_RUNS`.

@task(
    secret_requests=[
        Secret(key="reddit_client_id"),
        Secret(key="reddit_secret_key"),
    ]
)
def get_posts(search_terms: List[str]) -> List[Dict[str, str]]:
    # Load Secrets and Authenticate
    reddit_client_id = current_context().secrets.get("reddit_client_id")
    reddit_secret_key = current_context().secrets.get("reddit_secret_key")
    auth = HTTPBasicAuth(reddit_client_id, reddit_secret_key)

    # Format and run request for reddit posts
    query_string = '+'.join(search_terms)
    result = requests.get(f'https://www.reddit.com/search.json?q={query_string}&sort=new', auth=auth)
    result_json = result.json()
    posts = result_json['data']['children']

    # Get only recent posts
    weeks_ago_datetime = datetime.utcnow() - timedelta(weeks=WEEKS_BETWEEN_RUNS)
    weeks_ago_timestamp = weeks_ago_datetime.timestamp()
    recent_posts = [{
        'title': post['data']['title'],
        'description': post['data']['selftext'],
        'link': post['data']['url']
    } for post in posts if post['data']['created_utc'] >= weeks_ago_timestamp]

    return recent_posts

# ## Posting to Slack
#
# Given the reddit posts returned by our previous task, we can define a helper function to format them
# in a more readable format for Slack.

def format_posts(posts: List[Dict[str, str]]):
    formatted_message = "*Recent Posts:*\n\n"
    for idx, post in enumerate(posts, 1):
        formatted_message += f"*{idx}. <{post['link']}|{post['title']}>*\n"
        formatted_message += f"{post['description']}\n\n"
    return formatted_message

# We can then define another task to send our posts to slack. This time we can make use of our `ImageSpec`
# we previously defined that contains our `slack_sdk` package.

@task(
    container_image=image,
    secret_requests=[Secret(key="reddit_slack_token")]
)
def post_slack_message(recent_posts: List[Dict[str, str]]):
    from slack_sdk import WebClient

    reddit_slack_token = current_context().secrets.get("reddit_slack_token")
    client = WebClient(token=reddit_slack_token)

    response = client.chat_postMessage(
        channel=SLACK_CHANNEL_NAME,
        text=format_posts(recent_posts)
    )
    assert response["message"]["text"]

# ## Creating the Workflow
#
# Finally, we can chain these tasks together int a simple two-step workflow with some default inputs.

@workflow
def reddit_wf(search_terms: List[str]= ["flyte", "ml"]):
    recent_posts = get_posts(search_terms=search_terms)
    post_slack_message(recent_posts=recent_posts)

# ## Defining a Schedule
#
# Using our previously created workflow, we can define multiple `LaunchPlans` that will
# run on a schedule defined by standard `cron` expression. In this example we have one
# `LaunchPlan` querying for posts containing "flyte" and "ml", and another `LaunchPlan`
# querying for posts containing "union.ai".

LaunchPlan.get_or_create(
    reddit_wf,
    name="flyte_reddit_posts",
    default_inputs={"search_terms": ["flyte", "ml"]},
    schedule=CronSchedule(
        schedule=f"0 0 */{WEEKS_BETWEEN_RUNS*7} * *",
    )
)

LaunchPlan.get_or_create(
    reddit_wf,
    name="union_reddit_posts",
    default_inputs={"search_terms": ["union.ai"]},
    schedule=CronSchedule(
        schedule=f"0 0 */{WEEKS_BETWEEN_RUNS*7} * *",
    )
)

# After the `LaunchPlan` has been (registered)[https://docs.union.ai/serverless/development-cycle/registering-workflows#remote-cluster-using-fast-registration]
# using `unionai register <path/to/redit_slack_bot/>`, it must be [activated](https://docs.union.ai/byoc/core-concepts/launch-plans/activating-and-deactivating)
# either using the UI, CLI, or `UnionRemote`.


