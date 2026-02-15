import { createClient, type Client } from '@connectrpc/connect'
import { createTransport } from './transport'
import type { DescService } from '@bufbuild/protobuf'

export const client = <T extends DescService>(service: T): Client<T> => {
  const client = createClient(service, createTransport())
  return client
}
