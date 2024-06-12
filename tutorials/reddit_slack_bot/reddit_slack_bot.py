import time
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from flytekit import task, workflow, ImageSpec, Secret, current_context, LaunchPlan, CronSchedule
import requests
from requests.auth import HTTPBasicAuth

WEEKS_BETWEEN_RUNS = 1
SLACK_CHANNEL_NAME = '#reddit-posts'


image = ImageSpec(
    builder="unionai",
    packages=[
            "slack_sdk==3.28.0",
            "pandas"  # TODO: remove pandas when we switch to a dict (waiting on release with this fix: https://github.com/flyteorg/flytekit/pull/2469)
        ]
)


def format_posts(posts):
    formatted_message = "*Recent Posts:*\n\n"
    for idx, post in enumerate(posts, 1):
        formatted_message += f"*{idx}. <{post['link']}|{post['title']}>*\n"
        formatted_message += f"{post['description']}\n\n"
    return formatted_message

@task(
    container_image=image,
    secret_requests=[
        Secret(key="reddit_client_id"),
        Secret(key="reddit_secret_key"),
    ]
)
def get_posts(search_terms: List[str]) -> pd.DataFrame:  # TODO: return a dict rather than DataFrame
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

    return pd.DataFrame(recent_posts)


@task(
    container_image=image,
    secret_requests=[Secret(key="reddit_slack_token")]
)
def post_slack_message(df: pd.DataFrame):  # TODO: take a dict rather than DataFrame
    from slack_sdk import WebClient

    recent_posts = df.to_dict(orient='records')
    reddit_slack_token = current_context().secrets.get("reddit_slack_token")
    client = WebClient(token=reddit_slack_token)

    response = client.chat_postMessage(
        channel=SLACK_CHANNEL_NAME,
        text=format_posts(recent_posts)
    )
    assert response["message"]["text"]


@workflow
def reddit_wf(search_terms: List[str]):
    recent_posts = get_posts(search_terms=search_terms)
    post_slack_message(df=recent_posts)


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
