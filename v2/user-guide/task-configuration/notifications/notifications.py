# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = ""
# ///

from flyte import notify
from flyte.models import ActionPhase

# {{docs-fragment email-short-notification}}
notify.Email(
    on_phase=(ActionPhase.FAILED, ActionPhase.ABORTED),
    recipients=["oncall@example.com"],
    subject="Alert: Run completed with phase {{.Phase}}",
    body="Run: {{.Run.Name}}\nError: {{.Error}}",
)
# {{/docs-fragment email-short-notification}}

# {{docs-fragment slack-notification}}
notify.Slack(
    on_phase=ActionPhase.FAILED,
    webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    message="Run {{.Run.Name}} failed in {{.Run.Project}}/{{.Run.Domain}}: {{.Error}}",
)
# {{/docs-fragment slack-notification}}

# {{docs-fragment notification-blocks}}
notify.Slack(
    on_phase=ActionPhase.SUCCEEDED,
    webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    blocks=[
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Task Succeeded"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": "*Run:*\n{{.Run.Name}}"},
                {"type": "mrkdwn", "text": "*Phase:*\n{{.Phase}}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": "{{.Run.Project}}/{{.Run.Domain}}"},
            ],
        },
    ],
)
# {{/docs-fragment notification-blocks}}

# {{docs-fragment email-extended-notification}}
notify.Email(
    on_phase=ActionPhase.FAILED,
    recipients=["oncall@example.com"],
    cc=["team-lead@example.com"],
    subject="ALERT: Run {{.Run.Name}} failed",
    body="Run: {{.Run.Name}}\nError: {{.Error}}",
    html_body="<b>Error:</b> {{.Error}}<br>",
)
# {{/docs-fragment email-extended-notification}}

# {{docs-fragment teams-notification}}
notify.Teams(
    on_phase=ActionPhase.FAILED,
    webhook_url="https://outlook.office.com/webhook/YOUR_WEBHOOK_URL",
    title="Task Failed",
    message="Run {{.Run.Name}} failed: {{.Error}}\n",
)
# {{/docs-fragment teams-notification}}

# {{docs-fragment webhook-notification}}
notify.Webhook(
    on_phase=ActionPhase.SUCCEEDED,
    url="https://api.example.com/events",
    method="POST",
    headers={"Authorization": "Bearer my-token"},
    body={
        "event": "task_succeeded",
        "run": "{{.Run.Name}}",
    },
)
# {{/docs-fragment webhook-notification}}
