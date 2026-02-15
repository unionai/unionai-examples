import type {
  AbortInfo,
  ActionDetails,
  ClusterEvent,
  ErrorInfo,
  Run,
  TaskActionMetadata,
} from '@flyteorg/flyteidl2/workflow/run_definition_pb'

import { useEffect, useState } from 'react'
import { get } from './api/runs'
import type { ActionIdentifier } from '@flyteorg/flyteidl2/common/identifier_pb'

export const Result: React.FC<{ data: Run }> = ({ data }) => {
  const [details, setDetails] = useState<ActionDetails>()
  const [logs, setLogs] = useState<ClusterEvent[]>()
  const [showResult, setShowResult] = useState(false)
  const [showLogs, setShowLogs] = useState(false)

  useEffect(() => {
    get(data.action?.id!).then((details) => {
      setDetails(details)
      setLogs(details.attempts[0].clusterEvents)
    })
  }, [data.action?.id])

  const error: ErrorInfo | undefined =
    details?.result.case === 'errorInfo' ? details?.result.value : undefined
  const abort: AbortInfo | undefined =
    details?.result.case === 'abortInfo' ? details?.result.value : undefined

  const cellClass = error !== undefined ? 'error' : 'success'

  return (
    <tr key={data.action?.id?.run?.name} className={cellClass}>
      <td className="id">{data.action?.id?.run?.name}</td>
      <td>
        {(data.action?.metadata?.spec.value as TaskActionMetadata).shortName}
      </td>
      <td>{data.action?.status?.phase}</td>
      <td>
        {data.action?.status?.endTime?.seconds! -
          data.action?.status?.startTime?.seconds!}
        s
      </td>
      <td>
        {error !== undefined
          ? 'ERROR'
          : abort !== undefined
          ? 'ABORT'
          : 'SUCCESS'}
      </td>
      {!showResult ? (
        <td>
          {error !== undefined || abort !== undefined ? (
            <button onClick={() => setShowResult(true)}>Get</button>
          ) : (
            'n/a'
          )}
        </td>
      ) : (
        <>
          <td
            style={{
              whiteSpace: 'nowrap',
              textAlign: 'left',
              verticalAlign: 'top',
            }}
          >
            {error !== undefined ? <pre>{error.message.trim()}</pre> : null}
            {abort !== undefined ? <pre>{abort.reason}</pre> : null}
            {error === undefined && abort === undefined ? 'TBD' : null}
          </td>
        </>
      )}
      {!showLogs ? (
        <td colSpan={2}>
          <button onClick={() => setShowLogs(true)}>Get</button>
        </td>
      ) : logs !== undefined ? (
        <td>
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Event</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((l, i) => {
                const ts = new Date(
                  parseInt(l.occurredAt!.seconds!.toString(10)) * 1000
                )
                return (
                  <tr key={i}>
                    <td style={{ whiteSpace: 'nowrap', textAlign: 'left' }}>
                      {ts.toLocaleDateString()} {ts.toLocaleTimeString()}
                    </td>
                    <td style={{ whiteSpace: 'nowrap', textAlign: 'left' }}>
                      {l.message}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </td>
      ) : (
        <td>No logs</td>
      )}
    </tr>
  )
}
