import AGUI
import SwiftUI
import VisionAgentsCore

/// What the agent is waiting to be allowed to do, and the two answers to it.
///
/// This is the AG-UI interrupt a run ended on, which for a tool the app declared with an
/// `approval` question is the question itself. Nothing happens until one of these buttons is
/// pressed: the tool has not run, and the agent is holding its turn open waiting to be told.
public struct ApprovalView: View {
    private let approval: Interrupt
    private let approve: () async -> Void
    private let decline: () async -> Void

    /// - Parameters:
    ///   - approval: what is being asked, from `AgentSession.pendingApprovals`.
    ///   - approve: what to do when it is allowed, usually `session.approve(approval)`.
    ///   - decline: what to do when it is not.
    public init(
        approval: Interrupt,
        approve: @escaping () async -> Void,
        decline: @escaping () async -> Void
    ) {
        self.approval = approval
        self.approve = approve
        self.decline = decline
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label {
                Text(approval.message ?? "Is this alright?")
            } icon: {
                Image(systemName: "hand.raised")
                    .foregroundStyle(.secondary)
            }
            .font(.callout)

            HStack {
                Button("Not now") {
                    Task { await decline() }
                }
                .buttonStyle(.bordered)

                Button("Approve") {
                    Task { await approve() }
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        // The card is the host's colours, not ours: a background it can read against in
        // either scheme, and whatever tint the app set on the button that carries the answer.
        .background(Color(.secondarySystemBackground), in: .rect(cornerRadius: 12))
        .accessibilityElement(children: .contain)
    }
}
