import type { Project } from '@flyteorg/flyteidl2/flyteidl/admin/project_pb'

import { useContext, useEffect, useState, type ReactNode } from 'react'
import { list } from './api/projects'
import { Context } from './auth'
import { Runs } from './runs'

export const Projects = () => {
  const { isLoggedIn } = useContext(Context)
  const [projects, setProjects] = useState<Project[]>()
  const [selectedProjectId, setSelectedProjectId] = useState<string>()
  const [selectedDomainId, setSelectedDomainId] = useState<string>()

  useEffect(() => {
    if (isLoggedIn) {
      list().then((projects) => {
        setProjects(projects)
      })
    } else {
      setProjects(undefined)
    }
  }, [isLoggedIn])

  const options: ReactNode[] = []

  if (projects !== undefined) {
    options.push(<option key={'select'}>Select a project</option>)
    for (const p of projects) {
      options.push(
        <option key={`${p.name}`} value={`${p.name}`} disabled>
          --- {p.name} ---
        </option>
      )
      for (const d of p.domains) {
        options.push(
          <option key={`${p.name}-${d.name}`} value={`${p.name}-${d.name}`}>
            {p.name} - {d.name}
          </option>
        )
      }
    }
  }

  const hasSelectedProjectAndDomain =
    selectedProjectId !== undefined &&
    selectedDomainId !== undefined &&
    selectedDomainId.length > 0 &&
    selectedDomainId.length > 0
  const consoleUrl = hasSelectedProjectAndDomain
    ? `${process.env.UNION_ORG_URL}/v2/runs/project/${selectedProjectId}/domain/${selectedDomainId}`
    : `${process.env.UNION_ORG_URL}/v2`

  return (
    <>
      {isLoggedIn ? (
        <p>
          Console:{' '}
          <a href={consoleUrl} target="_blank">
            {consoleUrl}
          </a>
        </p>
      ) : null}
      {projects !== undefined ? (
        <select
          onChange={(e) => {
            const value = e.target.value.split('-')
            const project = value[0]
            const domain = value[1]
            console.log('Selecting', value, project, domain)
            setSelectedProjectId(project)
            setSelectedDomainId(domain)
          }}
        >
          {options}
        </select>
      ) : null}
      {hasSelectedProjectAndDomain ? (
        <Runs project={selectedProjectId} domain={selectedDomainId} />
      ) : null}
    </>
  )
}
