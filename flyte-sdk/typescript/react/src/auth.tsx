import {
  createContext,
  useContext,
  useEffect,
  useLayoutEffect,
  useState,
  type ReactNode,
} from 'react'
import {
  AuthContext,
  AuthProvider,
  type IAuthContext,
  type TAuthConfig,
} from 'react-oauth2-code-pkce'

interface ContextData {
  isLoggedIn: boolean
  login: VoidFunction
  logout: VoidFunction
  userInfo?: UserInfo
}

interface UserInfo {
  sub: string
  name: string
  email: string
}

const redirectUrl = new URL(document.location.origin + '/oauth/callback')

export const Context = createContext<ContextData>({
  isLoggedIn: false,
  login: () => {},
  logout: () => {},
})

const Network: React.FC<{ children: ReactNode }> = ({ children }) => {
  const { token, tokenData, logIn, logOut } =
    useContext<IAuthContext>(AuthContext)
  const [userInfo, setUserInfo] = useState<UserInfo>()

  useEffect(() => {
    if (tokenData !== undefined && userInfo === undefined) {
      fetch(`${tokenData.iss}/v1/userinfo`, {
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: 'application/json',
          'Content-Type': 'application/json',
        },
      }).then((response) => {
        if (response.ok) {
          response.json().then((user) => {
            setUserInfo(user)
          })
        }
      })
    }
  }, [tokenData, userInfo])

  return (
    <Context.Provider
      value={{
        isLoggedIn: token !== undefined && token.length > 0,
        login: () => {
          logIn(Date.now().toString(), undefined, 'redirect')
        },
        logout: logOut,
        userInfo,
      }}
    >
      {children}
    </Context.Provider>
  )
}

export const Provider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [authConfig, setAuthConfig] = useState<TAuthConfig>()

  useLayoutEffect(() => {
    fetch(`/.well-known/openid-configuration`).then((response) => {
      if (response.ok) {
        response.json().then((data) => {
          setAuthConfig({
            autoLogin: false,
            loginMethod: 'redirect',
            clientId: process.env.UNION_CLIENT_ID!,
            authorizationEndpoint: data.authorization_endpoint,
            tokenEndpoint: data.token_endpoint,
            redirectUri: redirectUrl.toString(),
            scope: 'openid profile email',
            state: Date.now().toString(),
          })
        })
      }
    })
  }, [])

  if (authConfig === undefined) {
    return <div>Loading...</div>
  }

  return (
    <div>
      <AuthProvider authConfig={authConfig}>
        <Network>{children}</Network>
      </AuthProvider>
    </div>
  )
}
