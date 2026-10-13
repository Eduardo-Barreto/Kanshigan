import type { ContentfulStatusCode } from 'hono/utils/http-status'

/** An error the API reports to the client as-is: a status code and a message safe to show. */
export class Problem extends Error {
  readonly status: ContentfulStatusCode

  constructor(status: ContentfulStatusCode, message: string) {
    super(message)
    this.name = 'Problem'
    this.status = status
  }
}

const HTTP_NOT_FOUND = 404
const HTTP_CONFLICT = 409
const HTTP_UNPROCESSABLE = 422

/** The requested dataset, clip or frame does not exist. */
export const notFound = (message: string) => new Problem(HTTP_NOT_FOUND, message)

/** The request is well-formed but cannot be applied to the current state on disk. */
export const conflict = (message: string) => new Problem(HTTP_CONFLICT, message)

/** The request is well-formed but asks for something the data cannot support. */
export const unprocessable = (message: string) => new Problem(HTTP_UNPROCESSABLE, message)
