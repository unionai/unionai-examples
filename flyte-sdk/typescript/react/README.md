# React Application -- Accessing Union V2

This sample React application, built with [Vite][vite], showcases how to access Union V2 APIs.

## Requirements

- Union V2 URL: The URL of the Union V2 instance that you are using.
- OAuth Client ID: The client ID of the OAuth client that you created in Union V2.
- Organization ID: The ID of the organization that you created in Union V2.

As any OAuth2 application, the authorization server must allow the redirect back to the application. In this example the path is:

    /oauth2/callback

If your application is deployed on a custom domain, you need to configure the redirect URI to include the domain. For example:

    https://<your-domain>/oauth2/callback

For this sample, the redirect URI is:

    http://localhost:5173/oauth2/callback

> Please reach out to Union support to get a client ID for your application and redirect URI.

## Configuration

You need to configure these three variables in `union.config.ts`:

    const UNION_ORG_URL = 'https://<your-organization-url>'
    const UNION_CLIENT_ID = '<client-id>'
    const UNION_ORG_ID = '<organization-id>'

Example:

    const UNION_ORG_URL = 'https://tryv2.hosted.unionai.cloud'
    const UNION_CLIENT_ID = '0oaqfip14wHD4EuT35d7'
    const UNION_ORG_ID = 'tryv2'

## Running

To run you simple run as a regular Vite application:

    pnpm install
    pnpm run dev

## Deploying

### As a Union Application

TBD

### On your own infrastructure

To deploy this application on your own domain, you need to configure the server that's hosting the application to redirect requests to Union V2.

This example uses the following services:

        /.well-known/openid-configuration
        /flyteidl.service.AdminService
        /cloudidl.workflow.RunService

As this is a pure-client React app, it does not provide a server-side component. Please ensure the server forwards those requests to your organization's Union V2 instance, same as `UNION_ORG_URL`.

[vite]: https://vitejs.dev/
