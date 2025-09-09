import { createConnectTransport } from '@connectrpc/connect-web'

export const createTransport = () =>
  createConnectTransport({
    baseUrl: window.origin,
    fetch: (input, init) => {
      const token = localStorage.getItem('ROCP_token')?.replaceAll('"', '')

      // Only add authorization header if token exists
      if (!token) {
        alert('Token is missing. This will not work. Please log in.')
        return Promise.reject(new Error('Token is missing'))
      }

      return fetch(input, {
        ...init,
        headers: {
          ...init?.headers,
          Authorization: `Bearer ${token}`,
        },
        credentials: 'include',
      })
    },
  })
