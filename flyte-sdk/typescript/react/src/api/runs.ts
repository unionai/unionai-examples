import type { ActionIdentifier } from '@flyteorg/flyteidl2/common/identifier_pb'
import type {
  ActionDetails,
  Run,
} from '@flyteorg/flyteidl2/workflow/run_definition_pb'
import { RunService } from '@flyteorg/flyteidl2/workflow/run_service_pb'
import { client } from './client'

export const list = async (project: string, domain: string): Promise<Run[]> => {
  console.log('Listing runs for project', project, 'domain', domain)
  const response = await client(RunService).listRuns({
    request: {
      limit: 200,
    },
    scopeBy: {
      case: 'projectId',
      value: {
        name: project,
        domain: domain,
        organization: process.env.UNION_ORG_ID as string,
      },
    },
  })
  return response.runs
}

export const get = async (
  actionId: ActionIdentifier
): Promise<ActionDetails> => {
  console.log('Fetching run details for action', actionId)
  const response = await client(RunService).getActionDetails({
    actionId,
  })
  if (response.details === undefined) {
    throw new Error('Action details not found')
  }
  return response.details
}
