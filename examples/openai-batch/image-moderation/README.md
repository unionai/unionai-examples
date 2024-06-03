# Image Moderation with OpenAI Batch API

Suppose you want to build an image moderation system that:

- Fetches images from a cloud blob storage every x hours,
- Triggers a downstream workflow to send requests to a GPT-4 model for image moderation, and
- Triggers another workflow to read the output file and send an email notification.

Union enables you to implement this seamlessly as follows:

- Define a launch plan with a cron schedule to fetch images every x hours.
- Use artifacts and reactive workflows to trigger downstream workflow.
- Send JSON requests to GPT-4 using the OpenAI Batch agent.
- Wait for the batch processing to complete.
- Send an email notification announcing the completion status.
