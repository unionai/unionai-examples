import type { Run } from '@flyteorg/flyteidl2/workflow/run_definition_pb'
import { useContext, useEffect, useState } from 'react'
import { list } from './api/runs'
import { Context } from './auth'
import { Result } from './runDetails'

export const Runs: React.FC<{ project: string; domain: string }> = ({
  project,
  domain,
}) => {
  const { isLoggedIn } = useContext(Context)
  const [runs, setRuns] = useState<Run[]>()

  useEffect(() => {
    if (isLoggedIn && project !== undefined && domain !== undefined) {
      list(project, domain).then(setRuns)
    } else {
      setRuns(undefined)
    }
  }, [project, domain, isLoggedIn])

  if (runs === undefined) {
    return null
  }

  return (
    <div>
      <h2>Runs</h2>
      {runs.length > 0 ? (
        <>
          <table>
            <thead>
              <tr>
                <th rowSpan={2}>ID</th>
                <th rowSpan={2}>Name</th>
                <th rowSpan={2}>Status</th>
                <th rowSpan={2}>Duration</th>
                <th colSpan={2}>Output</th>
                <th rowSpan={2}>Logs</th>
              </tr>
              <tr>
                <th>Type</th>
                <th>Output</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <Result key={r.action?.id?.run?.name} data={r} />
              ))}
            </tbody>
          </table>
        </>
      ) : (
        <div>No runs found</div>
      )}
    </div>
  )
}
