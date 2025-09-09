import { defineConfig, ViteDevServer } from 'vite'
import react from '@vitejs/plugin-react'
import { createProxyMiddleware } from 'http-proxy-middleware'
import path from 'path'
import { UNION_CLIENT_ID, UNION_ORG_ID, UNION_ORG_URL } from './union.config'

const proxy = (
  server: ViteDevServer,
  serviceName: string,
  appendPath: boolean = true
) => {
  server.middlewares.use(
    serviceName,
    createProxyMiddleware({
      changeOrigin: true,
      router: () => UNION_ORG_URL,
      on: {
        proxyReq: (proxyReq, _req, _res) => {
          if (appendPath) {
            proxyReq.path = serviceName + proxyReq.path
          } else {
            proxyReq.path = serviceName
          }
          proxyReq.setHeader('Content-Type', 'application/json')
          console.log(
            `Proxying to: ${proxyReq.getHeader('host')}${proxyReq.path}`
          )
        },
        error: (err, _req, res) => {
          console.error('Proxy error:', err.message)
          if (!res.headersSent) {
            res.writeHead(400, {
              'Content-Type': 'application/json',
            })
            res.end(JSON.stringify({ error: err.message }))
          }
        },
      },
    } as any)
  )
}

export default defineConfig({
  define: {
    'process.env': {
      UNION_ORG_URL: UNION_ORG_URL,
      UNION_CLIENT_ID: UNION_CLIENT_ID,
      UNION_ORG_ID: UNION_ORG_ID,
    },
  },
  resolve: {
    alias: {
      '@flyteorg/flyteidl2': path.resolve(__dirname, 'gen'),
    },
  },
  plugins: [
    react(),
    {
      name: 'flyte-proxy',
      configureServer(server) {
        proxy(server, '/.well-known/openid-configuration', false)
        proxy(server, '/flyteidl.service.AdminService')
        proxy(server, '/cloudidl.workflow.RunService')
      },
    },
  ],
})
