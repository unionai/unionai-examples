import type { Project } from '@flyteorg/flyteidl2/flyteidl/admin/project_pb'
import { AdminService } from '@flyteorg/flyteidl2/flyteidl/service/admin_pb'
import { client } from './client'

export const list = async (): Promise<Project[]> => {
  console.log('Fetching projects')
  try {
    const response = await client(AdminService).listProjects({
      limit: 200,
    })
    return response.projects
  } catch (error) {
    console.error(error)
    throw error
  }
}
