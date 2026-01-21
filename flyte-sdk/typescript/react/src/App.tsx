import { useContext, useState } from 'react'
import react from './assets/react.svg'
import { Context as AuthContext } from './auth'
import { Projects } from './projects'
import union from '/union.png'

function App() {
  const { isLoggedIn, login, logout, userInfo } = useContext(AuthContext)
  const [loggingIn, setLoggingIn] = useState(false)

  const loginHandler = () => {
    setLoggingIn(true)
    login()
  }

  if (
    document.location.pathname === '/oauth/callback' &&
    document.location.search === ''
  ) {
    document.location = '/' // Redirect to projects after auth
  }

  return (
    <>
      <h1
        style={{
          color: 'lightgray',
          display: 'flex',
          gap: 24,
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <img src={union} height={64} />
        +
        <img src={react} height={64} />
      </h1>
      <h1>Union 2.0 Typescript</h1>
      <div></div>
      <div>
        {loggingIn ? <p>Logging in...</p> : null}
        {!isLoggedIn || userInfo === undefined ? (
          <>
            <p>Org: {process.env.UNION_ORG_URL}</p>
            <button onClick={loginHandler} disabled={loggingIn}>
              Login
            </button>
          </>
        ) : (
          <>
            <p>
              Logged in as {userInfo === undefined ? 'NONE' : userInfo.name} (
              {userInfo === undefined ? 'NONE' : userInfo.email})
            </p>
            <button onClick={logout}>Logout</button>
          </>
        )}
      </div>
      <Projects />
    </>
  )
}

export default App
