import Foundation
import StreamCore



class DefaultAPI: DefaultAPIEndpoints, @unchecked Sendable {
    internal var transport: DefaultAPITransport
    internal var middlewares: [DefaultAPIClientMiddleware]
    internal var basePath: String
    internal var jsonDecoder: JSONDecoder
    internal var jsonEncoder: JSONEncoder

    init(
        basePath: String,
        transport: DefaultAPITransport,
        middlewares: [DefaultAPIClientMiddleware],
        jsonDecoder: JSONDecoder = JSONDecoder.default,
        jsonEncoder: JSONEncoder = JSONEncoder.default
    ) {
        self.basePath = basePath
        self.transport = transport
        self.middlewares = middlewares
        self.jsonDecoder = jsonDecoder
        self.jsonEncoder = jsonEncoder
    }

    func send<Response: Codable>(
        request: Request,
        deserializer: (Data) throws -> Response
    ) async throws -> Response {

        // TODO: make this a bit nicer and create an API error to make it easier to handle stuff
        func makeError(_ error: Error) -> Error {
            return error
        }

        func wrappingErrors<R>(
            work: () async throws -> R,
            mapError: (Error) -> Error
        ) async throws -> R {
            do {
                return try await work()
            } catch {
                throw mapError(error)
            }
        }

        let (data, _) = try await wrappingErrors {
            var next: (Request) async throws -> (Data, URLResponse) = { _request in
                try await wrappingErrors {
                    try await self.transport.execute(request: _request)
                } mapError: { error in
                    makeError(error)
                }
            }
            for middleware in middlewares.reversed() {
                let tmp = next
                next = {
                    try await middleware.intercept(
                        $0,
                        next: tmp
                    )
                }
            }
            return try await next(request)
        } mapError: { error in
            makeError(error)
        }

        return try await wrappingErrors {
            try deserializer(data)
        } mapError: { error in
            makeError(error)
        }
    }

    func makeRequest(
        uriPath: String,
        queryParams: [URLQueryItem] = [],
        httpMethod: String
    ) throws -> Request {
        let url = URL(string: basePath + uriPath)!
        return Request(
            url: url,
            method: .init(stringValue: httpMethod),
            queryParams: queryParams,
            headers: ["Content-Type": "application/json"]
        )
    }

    func makeRequest<T: Encodable>(
        uriPath: String,
        queryParams: [URLQueryItem] = [],
        httpMethod: String,
        request: T
    ) throws -> Request {
        var r = try makeRequest(uriPath: uriPath, queryParams: queryParams, httpMethod: httpMethod)
        r.body = try jsonEncoder.encode(request)
        return r
    }
    func createGuestUser(guestUserRequest: GuestUserRequest) async throws -> GuestUser {
        let path = "/v1/agents/guests"


        let urlRequest = try makeRequest(
            uriPath: path,
            httpMethod: "POST",
            request: guestUserRequest
        )
        return try await send(request: urlRequest) {
            try self.jsonDecoder.decode(GuestUser.self, from: $0)
        }
    }

    func createSession(createSessionRequest: CreateSessionRequest) async throws -> SessionPayload {
        let path = "/v1/agents/sessions"


        let urlRequest = try makeRequest(
            uriPath: path,
            httpMethod: "POST",
            request: createSessionRequest
        )
        return try await send(request: urlRequest) {
            try self.jsonDecoder.decode(SessionPayload.self, from: $0)
        }
    }

    func querySessions(sessionQueryPayload: SessionQueryPayload) async throws -> SessionPage {
        let path = "/v1/agents/sessions/query"


        let urlRequest = try makeRequest(
            uriPath: path,
            httpMethod: "POST",
            request: sessionQueryPayload
        )
        return try await send(request: urlRequest) {
            try self.jsonDecoder.decode(SessionPage.self, from: $0)
        }
    }

    func deleteSession(id: String) async throws -> Void {
        var path = "/v1/agents/sessions/{id}"

                let idPreEscape = "\(APIHelper.mapValueToPathItem(id))"
                let idPostEscape = idPreEscape.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? ""
                path = path.replacingOccurrences(of: String(format: "{%@}", "id"), with: idPostEscape, options: .literal, range: nil)

        let urlRequest = try makeRequest(
            uriPath: path,
            httpMethod: "DELETE"
        )
        _ = try await send(request: urlRequest) {
            try self.jsonDecoder.decode(EmptyResponse.self, from: $0)
        }
    }

    func getSession(id: String) async throws -> SessionPayload {
        var path = "/v1/agents/sessions/{id}"

                let idPreEscape = "\(APIHelper.mapValueToPathItem(id))"
                let idPostEscape = idPreEscape.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? ""
                path = path.replacingOccurrences(of: String(format: "{%@}", "id"), with: idPostEscape, options: .literal, range: nil)

        let urlRequest = try makeRequest(
            uriPath: path,
            httpMethod: "GET"
        )
        return try await send(request: urlRequest) {
            try self.jsonDecoder.decode(SessionPayload.self, from: $0)
        }
    }

    func updateSession(id: String, updateSessionRequest: UpdateSessionRequest) async throws -> SessionPayload {
        var path = "/v1/agents/sessions/{id}"

                let idPreEscape = "\(APIHelper.mapValueToPathItem(id))"
                let idPostEscape = idPreEscape.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? ""
                path = path.replacingOccurrences(of: String(format: "{%@}", "id"), with: idPostEscape, options: .literal, range: nil)

        let urlRequest = try makeRequest(
            uriPath: path,
            httpMethod: "PATCH",
            request: updateSessionRequest
        )
        return try await send(request: urlRequest) {
            try self.jsonDecoder.decode(SessionPayload.self, from: $0)
        }
    }

    func forkSession(id: String, forkSessionRequest: ForkSessionRequest) async throws -> SessionPayload {
        var path = "/v1/agents/sessions/{id}/fork"

                let idPreEscape = "\(APIHelper.mapValueToPathItem(id))"
                let idPostEscape = idPreEscape.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? ""
                path = path.replacingOccurrences(of: String(format: "{%@}", "id"), with: idPostEscape, options: .literal, range: nil)

        let urlRequest = try makeRequest(
            uriPath: path,
            httpMethod: "POST",
            request: forkSessionRequest
        )
        return try await send(request: urlRequest) {
            try self.jsonDecoder.decode(SessionPayload.self, from: $0)
        }
    }

    func listResponses(id: String, cursor: String?, limit: Int?) async throws -> AgentResponsePage {
        var path = "/v1/agents/sessions/{id}/responses"

                let idPreEscape = "\(APIHelper.mapValueToPathItem(id))"
                let idPostEscape = idPreEscape.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? ""
                path = path.replacingOccurrences(of: String(format: "{%@}", "id"), with: idPostEscape, options: .literal, range: nil)
                let queryParams = APIHelper.mapValuesToQueryItems([
                    "cursor": (wrappedValue: cursor?.encodeToJSON(), isExplode: true),
                    "limit": (wrappedValue: limit?.encodeToJSON(), isExplode: true)
                ])

        let urlRequest = try makeRequest(
            uriPath: path,
            queryParams: queryParams ?? [],
            httpMethod: "GET"
        )
        return try await send(request: urlRequest) {
            try self.jsonDecoder.decode(AgentResponsePage.self, from: $0)
        }
    }

    func createResponse(id: String, createResponseRequest: CreateResponseRequest) async throws -> AgentResponse {
        var path = "/v1/agents/sessions/{id}/responses"

                let idPreEscape = "\(APIHelper.mapValueToPathItem(id))"
                let idPostEscape = idPreEscape.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? ""
                path = path.replacingOccurrences(of: String(format: "{%@}", "id"), with: idPostEscape, options: .literal, range: nil)

        let urlRequest = try makeRequest(
            uriPath: path,
            httpMethod: "POST",
            request: createResponseRequest
        )
        return try await send(request: urlRequest) {
            try self.jsonDecoder.decode(AgentResponse.self, from: $0)
        }
    }

    func listResponseItems(id: String, cursor: String?, limit: Int?, responseId: String?) async throws -> AgentResponseItemPage {
        var path = "/v1/agents/sessions/{id}/responses/items"

                let idPreEscape = "\(APIHelper.mapValueToPathItem(id))"
                let idPostEscape = idPreEscape.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? ""
                path = path.replacingOccurrences(of: String(format: "{%@}", "id"), with: idPostEscape, options: .literal, range: nil)
                let queryParams = APIHelper.mapValuesToQueryItems([
                    "cursor": (wrappedValue: cursor?.encodeToJSON(), isExplode: true),
                    "limit": (wrappedValue: limit?.encodeToJSON(), isExplode: true),
                    "response_id": (wrappedValue: responseId?.encodeToJSON(), isExplode: true)
                ])

        let urlRequest = try makeRequest(
            uriPath: path,
            queryParams: queryParams ?? [],
            httpMethod: "GET"
        )
        return try await send(request: urlRequest) {
            try self.jsonDecoder.decode(AgentResponseItemPage.self, from: $0)
        }
    }

    func rewindSession(id: String, rewindSessionRequest: RewindSessionRequest) async throws -> Void {
        var path = "/v1/agents/sessions/{id}/rewind"

                let idPreEscape = "\(APIHelper.mapValueToPathItem(id))"
                let idPostEscape = idPreEscape.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? ""
                path = path.replacingOccurrences(of: String(format: "{%@}", "id"), with: idPostEscape, options: .literal, range: nil)

        let urlRequest = try makeRequest(
            uriPath: path,
            httpMethod: "POST",
            request: rewindSessionRequest
        )
        _ = try await send(request: urlRequest) {
            try self.jsonDecoder.decode(EmptyResponse.self, from: $0)
        }
    }

    func stopSession(id: String) async throws -> Void {
        var path = "/v1/agents/sessions/{id}/stop"

                let idPreEscape = "\(APIHelper.mapValueToPathItem(id))"
                let idPostEscape = idPreEscape.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? ""
                path = path.replacingOccurrences(of: String(format: "{%@}", "id"), with: idPostEscape, options: .literal, range: nil)

        let urlRequest = try makeRequest(
            uriPath: path,
            httpMethod: "POST"
        )
        _ = try await send(request: urlRequest) {
            try self.jsonDecoder.decode(EmptyResponse.self, from: $0)
        }
    }

    func stopSessionVoice(id: String) async throws -> SessionPayload {
        var path = "/v1/agents/sessions/{id}/voice"

                let idPreEscape = "\(APIHelper.mapValueToPathItem(id))"
                let idPostEscape = idPreEscape.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? ""
                path = path.replacingOccurrences(of: String(format: "{%@}", "id"), with: idPostEscape, options: .literal, range: nil)

        let urlRequest = try makeRequest(
            uriPath: path,
            httpMethod: "DELETE"
        )
        return try await send(request: urlRequest) {
            try self.jsonDecoder.decode(SessionPayload.self, from: $0)
        }
    }

    func startSessionVoice(id: String, startSessionVoiceRequest: StartSessionVoiceRequest) async throws -> SessionPayload {
        var path = "/v1/agents/sessions/{id}/voice"

                let idPreEscape = "\(APIHelper.mapValueToPathItem(id))"
                let idPostEscape = idPreEscape.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? ""
                path = path.replacingOccurrences(of: String(format: "{%@}", "id"), with: idPostEscape, options: .literal, range: nil)

        let urlRequest = try makeRequest(
            uriPath: path,
            httpMethod: "POST",
            request: startSessionVoiceRequest
        )
        return try await send(request: urlRequest) {
            try self.jsonDecoder.decode(SessionPayload.self, from: $0)
        }
    }

    func search(searchRequest: SearchRequest) async throws -> SearchAnswerPayload {
        let path = "/v1/search"


        let urlRequest = try makeRequest(
            uriPath: path,
            httpMethod: "POST",
            request: searchRequest
        )
        return try await send(request: urlRequest) {
            try self.jsonDecoder.decode(SearchAnswerPayload.self, from: $0)
        }
    }

}



protocol DefaultAPIEndpoints {
    func createGuestUser(guestUserRequest: GuestUserRequest) async throws -> GuestUser

    func createSession(createSessionRequest: CreateSessionRequest) async throws -> SessionPayload

    func querySessions(sessionQueryPayload: SessionQueryPayload) async throws -> SessionPage

    func deleteSession(id: String) async throws -> Void

    func getSession(id: String) async throws -> SessionPayload

    func updateSession(id: String, updateSessionRequest: UpdateSessionRequest) async throws -> SessionPayload

    func forkSession(id: String, forkSessionRequest: ForkSessionRequest) async throws -> SessionPayload

    func listResponses(id: String, cursor: String?, limit: Int?) async throws -> AgentResponsePage

    func createResponse(id: String, createResponseRequest: CreateResponseRequest) async throws -> AgentResponse

    func listResponseItems(id: String, cursor: String?, limit: Int?, responseId: String?) async throws -> AgentResponseItemPage

    func rewindSession(id: String, rewindSessionRequest: RewindSessionRequest) async throws -> Void

    func stopSession(id: String) async throws -> Void

    func stopSessionVoice(id: String) async throws -> SessionPayload

    func startSessionVoice(id: String, startSessionVoiceRequest: StartSessionVoiceRequest) async throws -> SessionPayload

    func search(searchRequest: SearchRequest) async throws -> SearchAnswerPayload

    
}