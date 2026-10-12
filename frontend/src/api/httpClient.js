export class ApiError extends Error {
  constructor(message, statusCode, retryAfterSeconds) {
    super(message);
    this.statusCode = statusCode;
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

async function getErrorMessage(response) {
  try {
    const errorBody = await response.json();
    return typeof errorBody.detail === "string" ? errorBody.detail : JSON.stringify(errorBody.detail ?? errorBody);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

export async function sendRequest(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const retryAfterSeconds = Number(response.headers.get("Retry-After")) || null;
    throw new ApiError(await getErrorMessage(response), response.status, retryAfterSeconds);
  }
  return response.status === 204 ? null : response.json();
}

export const getJson = (url) => sendRequest(url);

export const postJson = (url, body) =>
  sendRequest(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });

export const postForm = (url, formData) => sendRequest(url, { method: "POST", body: formData });

export const deleteResource = (url) => sendRequest(url, { method: "DELETE" });
